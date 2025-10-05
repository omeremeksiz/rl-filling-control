from __future__ import annotations

import os
import random
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import yaml

from utils.communication_utils import create_modbus_client, create_tcp_client, parse_live_payload_to_floats
from utils.logging_utils import setup_legacy_training_logger, get_legacy_output_paths
from utils.plotting_utils import (
    plot_qvalue_vs_state_from_pair_table,
    plot_switching_trajectory_with_exploration,
)
from utils.database_utils import DatabaseHandler
from utils.data_processing import DataProcessor, EpisodeMeta


def init_database_handler(cfg: Dict[str, Any], logger) -> Optional[DatabaseHandler]:
    db_section = cfg.get("database")
    if isinstance(db_section, dict) and db_section.get("enabled") is False:
        logger.info("Database logging disabled via configuration.")
        return None

    db_config: Optional[Dict[str, Any]] = None
    if isinstance(db_section, dict):
        candidate = {k: v for k, v in db_section.items() if k != "enabled"}
        nested = candidate.pop("config", None)
        if isinstance(nested, dict):
            candidate.update(nested)
        db_config = candidate or None

    try:
        handler = DatabaseHandler(db_config)
    except Exception as exc:  # pragma: no cover - defensive
        logger.error(f"Failed to initialise database handler: {exc}")
        return None

    if not handler.connect():
        logger.warning(
            "Database connection unavailable; continuing without persistence. "
            "(Check mysql-connector-python installation and DB credentials.)"
        )
        return None

    safe_host = handler.config.get("host", "?")
    safe_db = handler.config.get("name", "?")
    logger.info(f"Database logging enabled -> host={safe_host}, database={safe_db}")
    return handler


def persist_episode(
    handler: Optional[DatabaseHandler],
    logger,
    *,
    raw_data: str,
    weight_sequence: List[int],
    switch_point: int,
    episode_length: int,
    final_weight: int,
    safe_min: int,
    safe_max: int,
    q_values_snapshot: Optional[Dict[Tuple[int, int], float]] = None,
    meta: Optional[EpisodeMeta] = None,
) -> None:
    if handler is None:
        return

    overflow_amount = (
        meta.overflow_amount if meta and meta.overflow_amount is not None
        else max(0, final_weight - safe_max)
    )
    underflow_amount = (
        meta.underflow_amount if meta and meta.underflow_amount is not None
        else max(0, safe_min - final_weight)
    )

    coarse_time = meta.coarse_time if meta and meta.coarse_time is not None else episode_length
    fine_time = meta.fine_time if meta and meta.fine_time is not None else episode_length
    total_time = meta.total_time if meta and meta.total_time is not None else episode_length
    switch_state = meta.switch_point if meta and meta.switch_point is not None else switch_point

    try:
        original_id = handler.save_original_episode(
            raw_data or "",
            coarse_time,
            fine_time,
            total_time,
            switch_state,
            overflow_amount,
            underflow_amount,
        )
    except Exception as exc:  # pragma: no cover
        logger.error(f"Failed to save original episode: {exc}")
        return

    parsed_id: Optional[int] = None
    if original_id:
        try:
            parsed_id = handler.save_parsed_episode(
                original_id,
                weight_sequence,
                coarse_time,
                fine_time,
                total_time,
                switch_state,
                overflow_amount,
                underflow_amount,
            )
        except Exception as exc:  # pragma: no cover
            logger.error(f"Failed to save parsed episode: {exc}")

    if parsed_id and q_values_snapshot:
        try:
            handler.update_q_values(parsed_id, q_values_snapshot)
        except Exception as exc:  # pragma: no cover
            logger.error(f"Failed to persist Q-values: {exc}")

    stats_payload = {
        "valid_fillings": 1,
        "overflow_fillings": 1 if overflow_amount > 0 else 0,
        "underflow_fillings": 1 if underflow_amount > 0 else 0,
        "safe_fillings": 1 if overflow_amount == 0 and underflow_amount == 0 else 0,
    }

    try:
        handler.save_statistics(**stats_payload)
    except Exception as exc:  # pragma: no cover
        logger.error(f"Failed to update statistics: {exc}")


def load_config() -> Dict[str, Any]:
    with open(os.path.join("configs", "td_test.yaml"), "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def calc_reward_td(final_weight: int, safe_min: int = 80, safe_max: int = 120,
                   overflow_penalty_constant: float = -10.0, underflow_penalty_constant: float = -10.0) -> float:
    base_reward = 0.0
    if safe_min <= final_weight <= safe_max:
        penalty = 0.0
    elif final_weight > safe_max:
        penalty = (final_weight - safe_max) * overflow_penalty_constant
    else:
        penalty = (safe_min - final_weight) * underflow_penalty_constant
    return base_reward + penalty


def main() -> None:
    cfg = load_config()
    rng_seed = int(cfg.get("seed", 42))
    random.seed(rng_seed)
    np.random.seed(rng_seed)

    logger, output_dir, _ = setup_legacy_training_logger(base_dir="outputs")
    paths = get_legacy_output_paths(output_dir)

    test_cfg = cfg.get("testing", {})
    episodes = int(test_cfg.get("episodes", 50))
    max_steps = int(test_cfg.get("max_steps_per_episode", 100))

    hp = cfg.get("hyperparameters", {})
    alpha = float(hp.get("alpha", 0.1))
    gamma = float(hp.get("gamma", 0.99))
    safe_min = int(hp.get("safe_min", 80))
    safe_max = int(hp.get("safe_max", 120))

    comm_cfg = cfg.get("communication", {})
    tcp_cfg = comm_cfg.get("tcp", {})
    modbus_cfg = comm_cfg.get("modbus", {})

    tcp = create_tcp_client(tcp_cfg.get("host", "127.0.0.1"), int(tcp_cfg.get("port", 5051)), timeout=float(tcp_cfg.get("timeout", 2.0)))
    modbus = create_modbus_client(modbus_cfg.get("host", "127.0.0.1"), int(modbus_cfg.get("port", 502)), register=int(modbus_cfg.get("register", 40010)))
    tcp.connect()
    modbus.connect()

    # Online TD (SARSA) using live reward surrogate (final weight surrogate per step)
    value_map: Dict[Tuple[int, int], float] = {}
    current_switch_point = int(test_cfg.get("initial_switch_point", 500))
    observed_states: Set[int] = set()
    episode_indices: List[int] = []
    model_selected_list: List[int] = []
    explored_list: List[Optional[int]] = []

    db_handler = init_database_handler(cfg, logger)
    data_processor = DataProcessor()

    try:
        for ep in range(episodes):
            raw_payloads: List[str] = []
            weight_trace: List[int] = []
            final_weight = current_switch_point

            for _ in range(max_steps):
                payload = tcp.receive_data()
                if payload:
                    raw_payloads.append(payload)
                xs = parse_live_payload_to_floats(payload) if payload else []
                if xs:
                    weight_trace.extend(int(round(x)) for x in xs)
                final_weight_surrogate = int(max(0.0, float(np.mean(xs)) if xs else 0.0))
                final_weight = final_weight_surrogate
                observed_states.add(final_weight_surrogate)

                r = calc_reward_td(final_weight_surrogate, safe_min=safe_min, safe_max=safe_max)
                s_t = final_weight_surrogate
                a_t = 1 if s_t < current_switch_point else -1
                q_sa = value_map.get((s_t, a_t), 0.0)

                s_tp1 = s_t
                a_tp1 = 1 if s_tp1 < current_switch_point else -1
                q_next = value_map.get((s_tp1, a_tp1), 0.0)
                td_target = r + gamma * q_next
                value_map[(s_t, a_t)] = q_sa + alpha * (td_target - q_sa)

                modbus.send_switch_point(float(max(0.0, current_switch_point)))

            raw_combined = "".join(raw_payloads)
            session = None
            meta: Optional[EpisodeMeta] = None
            core_sequence: Optional[List[int]] = None
            if raw_combined:
                session, meta, core_sequence = data_processor.parse_real_episode(
                    raw_combined,
                    session_id=f"test_ep_{ep + 1}",
                )

            if meta and session:
                final_candidate = meta.final_weight if meta.final_weight is not None else session.final_weight
                if final_candidate is not None:
                    final_weight = final_candidate

                length_candidate = (
                    meta.episode_length if meta.episode_length is not None else session.episode_length
                )
                episode_length_db = length_candidate if length_candidate is not None else len(core_sequence or weight_trace)
                switch_for_storage = (
                    session.switch_point if session.switch_point is not None else current_switch_point
                )
            else:
                episode_length_db = len(weight_trace)
                switch_for_storage = current_switch_point
                if core_sequence is None:
                    core_sequence = weight_trace.copy()

            store_sequence = core_sequence if core_sequence else (weight_trace if weight_trace else [final_weight])
            if episode_length_db <= 0:
                episode_length_db = len(store_sequence)

            state_to_best: Dict[int, Tuple[int, float]] = {}
            for (state, action), value in value_map.items():
                best = state_to_best.get(state)
                if best is None or value > best[1]:
                    state_to_best[state] = (action, value)

            best_switch_point = current_switch_point
            for state in sorted(observed_states):
                best = state_to_best.get(state)
                if best is not None and best[0] == -1:
                    best_switch_point = state
                    break

            episode_indices.append(ep + 1)
            model_selected_list.append(best_switch_point)
            explored_list.append(None)

            logger.info(f"Test Episode {ep + 1}/{episodes} done | switch_point={best_switch_point}")

            persist_episode(
                db_handler,
                logger,
                raw_data=meta.raw_data if meta and meta.raw_data else raw_combined,
                weight_sequence=store_sequence,
                switch_point=int(switch_for_storage),
                episode_length=episode_length_db,
                final_weight=int(final_weight),
                safe_min=safe_min,
                safe_max=safe_max,
                q_values_snapshot=value_map.copy(),
                meta=meta,
            )

            current_switch_point = best_switch_point
    finally:
        tcp.close()
        modbus.close()
        if db_handler:
            db_handler.close()

    if value_map:
        plot_qvalue_vs_state_from_pair_table(value_map, paths['qvalue_vs_state_path'])
    plot_switching_trajectory_with_exploration(
        episode_indices,
        model_selected_list,
        explored_list,
        paths['switching_point_trajectory_path'],
    )


if __name__ == "__main__":
    main()

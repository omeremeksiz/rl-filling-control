from __future__ import annotations

import json
import os
import random
from typing import Any, Dict, List, Optional

import numpy as np
import yaml

from utils.communication_utils import create_modbus_client, create_tcp_client, parse_live_payload_to_floats
from utils.logging_utils import setup_legacy_training_logger, get_legacy_output_paths
from utils.plotting_utils import (
    plot_qvalue_vs_state_bandit,
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
        logger.warning("Database connection unavailable; continuing without persistence.")
        return None
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
    q_values_snapshot: Optional[Dict[int, float]] = None,
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
    except Exception as exc:  # pragma: no cover - legacy safeguard
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
    config_path = os.path.join("configs", "mab_test.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def calc_reward_mab(episode_length: int, final_weight: int, safe_min: int, safe_max: int,
                    overflow_penalty_constant: float, underflow_penalty_constant: float) -> float:
    base_reward = -episode_length
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
    episodes = int(test_cfg.get("episodes"))

    hp = cfg.get("hyperparameters", {})

    alpha = float(hp.get("alpha"))
    epsilon = float(hp.get("epsilon_start"))
    epsilon_min = float(hp.get("epsilon_min"))
    epsilon_decay = float(hp.get("epsilon_decay"))
    safe_min = int(hp.get("safe_min"))
    safe_max = int(hp.get("safe_max"))
    overflow_penalty_constant = float(hp.get("overflow_penalty_constant"))
    underflow_penalty_constant = float(hp.get("underflow_penalty_constant"))
    available_switch_points = list(range(safe_max + 1))  # 0 to safe_max inclusive
    starting_switch_point = int(hp.get("starting_switch_point"))

    comm_cfg = cfg.get("communication", {})
    tcp_cfg = comm_cfg.get("tcp", {})
    modbus_cfg = comm_cfg.get("modbus", {})

    tcp = create_tcp_client(
        tcp_cfg.get("host", "127.0.0.1"),
        int(tcp_cfg.get("port", 5051)),
        timeout=float(tcp_cfg.get("timeout", 2.0)),
    )
    modbus = create_modbus_client(
        modbus_cfg.get("host", "127.0.0.1"),
        int(modbus_cfg.get("port", 1502)),
        register=int(modbus_cfg.get("register", 40010)),
    )

    if not tcp.connect() or not modbus.connect():
        logger.error("Failed to connect to one of the communication endpoints; aborting test run.")
        return

    q_table: Dict[int, float] = {sp: 0.0 for sp in available_switch_points}
    current_switch_point = starting_switch_point

    db_handler = init_database_handler(cfg, logger)
    data_processor = DataProcessor()

    episode_indices: List[int] = []
    model_selected_list: List[int] = []
    explored_list: List[Optional[int]] = []
    episode_rewards: List[float] = []
    final_weights: List[int] = []

    try:
        for ep in range(episodes):
            logger.info(f"--- Episode {ep + 1}/{episodes} ---")
            logger.info(f"Dispatching switching point: {current_switch_point}")

            modbus.send_switch_point(float(current_switch_point))

            raw_payloads: List[str] = []
            weight_samples: List[int] = []

            payload = tcp.receive_data()
            if payload:
                raw_payloads.append(payload)
            values = parse_live_payload_to_floats(payload) if payload else []
            weight_samples.extend(int(round(val)) for val in values)

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
                final_candidate = (
                    meta.final_weight if meta.final_weight is not None else session.final_weight
                )
                if final_candidate is not None:
                    final_weight = final_candidate

                length_candidate = (
                    meta.episode_length if meta.episode_length is not None else session.episode_length
                )
                episode_length = length_candidate if length_candidate is not None else len(weight_samples)

                switch_for_storage = (
                    session.switch_point if session.switch_point is not None else current_switch_point
                )
            else:
                episode_length = len(weight_samples)
                final_weight = int(max(0.0, float(np.mean(weight_samples)) if weight_samples else 0.0))
                switch_for_storage = current_switch_point
                core_sequence = weight_samples.copy()
                meta = meta if meta else None

            store_sequence = core_sequence if core_sequence else weight_samples

            final_weights.append(final_weight)

            reward = calc_reward_mab(
                episode_length,
                final_weight,
                safe_min,
                safe_max,
                overflow_penalty_constant,
                underflow_penalty_constant,
            )
            episode_rewards.append(reward)

            q_table[current_switch_point] = q_table[current_switch_point] + alpha * (reward - q_table[current_switch_point])

            best_switch_point = max(q_table, key=q_table.get)
            termination = (
                "Normal" if safe_min <= final_weight <= safe_max
                else ("Underflow" if final_weight < safe_min else "Overflow")
            )
            logger.info(f"Termination Type: {termination}")
            logger.info(f"Observed final weight: {final_weight}")
            logger.info(f"Model-Selected Next Switching Point: {best_switch_point}")

            explored_choice: Optional[int] = None
            if random.random() < epsilon:
                idx = available_switch_points.index(current_switch_point)
                next_idx = min(idx + 1, len(available_switch_points) - 1)
                explored_choice = available_switch_points[next_idx]
                next_switch_point = explored_choice
            else:
                next_switch_point = best_switch_point

            epsilon = max(epsilon_min, epsilon * epsilon_decay)

            episode_indices.append(ep + 1)
            model_selected_list.append(best_switch_point)
            explored_list.append(explored_choice)

            persist_episode(
                db_handler,
                logger,
                raw_data=meta.raw_data if meta and meta.raw_data else raw_combined,
                weight_sequence=store_sequence or [final_weight],
                switch_point=int(switch_for_storage),
                episode_length=episode_length,
                final_weight=final_weight,
                safe_min=safe_min,
                safe_max=safe_max,
                q_values_snapshot=q_table.copy(),
                meta=meta,
            )

            current_switch_point = next_switch_point

    finally:
        tcp.close()
        modbus.close()
        if db_handler:
            db_handler.close()

    plot_qvalue_vs_state_bandit(q_table, paths['qvalue_vs_state_path'])
    plot_switching_trajectory_with_exploration(
        episode_indices,
        model_selected_list,
        explored_list,
        paths['switching_point_trajectory_path'],
    )

    metrics = {
        "episodes": episodes,
        "best_switch_point": int(max(q_table, key=q_table.get)) if q_table else None,
    }
    with open(os.path.join(output_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Finished testing. Metrics: %s", metrics)


if __name__ == "__main__":
    main()

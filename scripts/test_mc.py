from __future__ import annotations

import json
import os
import random
from typing import Any, Dict, List, Optional, Tuple, Set

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
    config_path = os.path.join("configs", "mc_test.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def calc_reward_mc(final_weight: int, safe_min: int, safe_max: int,
                   overflow_penalty_constant: float, underflow_penalty_constant: float) -> float:
    base_reward = 0.0
    if safe_min <= final_weight <= safe_max:
        penalty = 0.0
    elif final_weight > safe_max:
        penalty = (final_weight - safe_max) * overflow_penalty_constant
    else:
        penalty = (safe_min - final_weight) * underflow_penalty_constant
    return base_reward + penalty


def ensure_q_entries(q_table: Dict[Tuple[int, int], float], state: int, initial_q: float) -> None:
    if (state, 1) not in q_table:
        q_table[(state, 1)] = initial_q
    if (state, -1) not in q_table:
        q_table[(state, -1)] = initial_q


def build_mc_trajectory(
    weight_sequence: List[int],
    current_switch_point: int,
    final_weight: int,
    safe_min: int,
    safe_max: int,
    overflow_penalty_constant: float,
    underflow_penalty_constant: float,
) -> List[Tuple[int, int, float]]:
    trajectory: List[Tuple[int, int, float]] = []
    step_cost = -1.0
    for w in weight_sequence:
        if w in (-1, 300):
            continue
        action = 1 if w < current_switch_point else -1
        trajectory.append((w, action, step_cost))
    if trajectory:
        w_last, a_last, r_last = trajectory[-1]
        r_last += calc_reward_mc(final_weight, safe_min, safe_max,
                                 overflow_penalty_constant, underflow_penalty_constant)
        trajectory[-1] = (w_last, a_last, r_last)
    return trajectory


def main() -> None:
    cfg = load_config()
    rng_seed = int(cfg.get("seed", 42))
    random.seed(rng_seed)
    np.random.seed(rng_seed)

    logger, output_dir, _ = setup_legacy_training_logger(base_dir="outputs")
    paths = get_legacy_output_paths(output_dir)

    test_cfg = cfg.get("testing", {})
    episodes = int(test_cfg.get("episodes", 1))
    max_steps = int(test_cfg.get("max_steps_per_episode", 1))

    hp = cfg.get("hyperparameters", {})
    gamma = float(hp.get("gamma"))
    alpha = float(hp.get("alpha"))
    initial_q = float(hp.get("initial_q"))
    epsilon = float(hp.get("epsilon_start"))
    epsilon_min = float(hp.get("epsilon_min"))
    epsilon_decay = float(hp.get("epsilon_decay"))
    underflow_penalty_constant = float(hp.get("underflow_penalty_constant"))
    overflow_penalty_constant = float(hp.get("overflow_penalty_constant"))
    safe_min = int(hp.get("safe_min"))
    safe_max = int(hp.get("safe_max"))
    starting_sp = int(hp.get("starting_switch_point"))

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

    db_handler = init_database_handler(cfg, logger)
    data_processor = DataProcessor()

    q_table: Dict[Tuple[int, int], float] = {}
    positive_updates: Set[int] = set()
    known_sps: List[int] = [starting_sp]

    current_sp = starting_sp
    traj_ep: List[int] = []
    model_selected_list: List[int] = []
    explored_list: List[Optional[int]] = []

    try:
        for ep in range(episodes):
            logger.info(f"--- Episode {ep + 1}/{episodes} ---")
            logger.info(f"Dispatching switching point: {current_sp}")

            modbus.send_switch_point(float(current_sp))

            raw_payloads: List[str] = []
            weight_trace: List[int] = []

            for _ in range(max_steps):
                payload = tcp.receive_data()
                if payload:
                    raw_payloads.append(payload)
                values = parse_live_payload_to_floats(payload) if payload else []
                if values:
                    weight_trace.extend(int(round(v)) for v in values)

            raw_combined = "".join(raw_payloads)
            session = None
            meta: Optional[EpisodeMeta] = None
            core_sequence: Optional[List[int]] = None
            if raw_combined:
                session, meta, core_sequence = data_processor.parse_real_episode(
                    raw_combined,
                    session_id=f"test_ep_{ep + 1}",
                )

            if session and meta and core_sequence:
                weight_sequence = [w for w in session.weight_sequence[:-1] if w not in (-1, 300)]
                final_weight = meta.final_weight if meta.final_weight is not None else (session.final_weight or 0)
            else:
                weight_sequence = weight_trace.copy()
                final_weight = int(np.mean(weight_trace)) if weight_trace else 0
                meta = meta if meta else None

            for state in weight_sequence:
                ensure_q_entries(q_table, state, initial_q)
                if state not in known_sps:
                    known_sps.append(state)
            known_sps = sorted(set(known_sps))

            trajectory = build_mc_trajectory(
                weight_sequence,
                current_sp,
                final_weight,
                safe_min,
                safe_max,
                overflow_penalty_constant,
                underflow_penalty_constant,
            )

            G = 0.0
            for state, action, reward in reversed(trajectory):
                ensure_q_entries(q_table, state, initial_q)
                G = reward + gamma * G
                if action == -1 and state not in positive_updates:
                    continue
                q_sa = q_table[(state, action)]
                q_table[(state, action)] = q_sa + alpha * (G - q_sa)
                if action == 1:
                    positive_updates.add(state)

            state_to_best: Dict[int, Tuple[int, float]] = {}
            for (state, action), value in q_table.items():
                best = state_to_best.get(state)
                if best is None or value > best[1]:
                    state_to_best[state] = (action, value)

            best_sp = current_sp
            for state in sorted(state_to_best.keys()):
                if state_to_best[state][0] == -1:
                    best_sp = state
                    break

            explored_choice: Optional[int] = None
            if random.random() < epsilon and len(known_sps) > 1:
                if current_sp in known_sps:
                    idx = known_sps.index(current_sp)
                    target = min(idx + 1, len(known_sps) - 1)
                    explored_choice = known_sps[target]
                    next_sp = explored_choice
                else:
                    next_sp = best_sp
            else:
                next_sp = best_sp

            epsilon = max(epsilon_min, epsilon * epsilon_decay)

            termination_label = (
                "safe" if safe_min <= final_weight <= safe_max
                else ("underweight" if final_weight < safe_min else "overweight")
            )
            logger.info(f"Termination Type: {termination_label}")
            logger.info(f"Model-Selected Next Switching Point: {best_sp}")
            logger.info(f"Explored Switching Point: {explored_choice}")
            logger.info("")

            traj_ep.append(ep + 1)
            model_selected_list.append(best_sp)
            explored_list.append(explored_choice)

            persist_episode(
                db_handler,
                logger,
                raw_data=meta.raw_data if meta and meta.raw_data else raw_combined,
                weight_sequence=weight_sequence or (weight_trace if weight_trace else [final_weight]),
                switch_point=int(current_sp),
                episode_length=meta.episode_length if meta and meta.episode_length is not None else max(1, len(weight_sequence)),
                final_weight=int(final_weight),
                safe_min=safe_min,
                safe_max=safe_max,
                q_values_snapshot=q_table.copy(),
                meta=meta,
            )

            current_sp = next_sp

    finally:
        tcp.close()
        modbus.close()
        if db_handler:
            db_handler.close()

    if q_table:
        plot_qvalue_vs_state_from_pair_table(q_table, paths['qvalue_vs_state_path'])
    plot_switching_trajectory_with_exploration(
        traj_ep,
        model_selected_list,
        explored_list,
        paths['switching_point_trajectory_path'],
    )

    metrics = {
        "episodes": episodes,
        "best_switch_point": model_selected_list[-1] if model_selected_list else None,
    }
    with open(os.path.join(output_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Finished testing. Metrics: %s", metrics)


if __name__ == "__main__":
    main()

# scripts/test_q.py
from __future__ import annotations

import json
import os
import random
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Set

import numpy as np
import yaml

from utils.communication_utils import create_modbus_client, create_tcp_client, parse_live_payload_to_floats
from utils.excel_logging import write_qtable_to_excel
from utils.logging_utils import (
    setup_legacy_training_logger,
    get_legacy_output_paths,
    copy_config_to_output,
)
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


CONFIG_PATH = os.path.join("configs", "q_test.yaml")


def load_config() -> Dict[str, Any]:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def calc_reward_q(final_weight: int, safe_min: int, safe_max: int,
                         overflow_penalty_constant: float, underflow_penalty_constant: float) -> float:
    base_reward = 0.0
    if safe_min <= final_weight <= safe_max:
        penalty = 0.0
    elif final_weight > safe_max:
        penalty = (final_weight - safe_max) * overflow_penalty_constant
    else:
        penalty = (safe_min - final_weight) * underflow_penalty_constant
    return base_reward + penalty


def _pick_next_switch_point(
    best_switch_point: int,
    candidates: List[int],
    epsilon: float,
    weights: Optional[List[float]],
) -> Tuple[int, Optional[int]]:
    if random.random() >= epsilon: # no exploration, stay with best sp
        return best_switch_point, None

    base_idx = candidates.index(best_switch_point)
    
    total = sum(weights)
    if total <= 0.0:
        return best_switch_point, None

    roll = random.random() * total
    cumulative = 0.0
    offset = 1
    for step, weight in enumerate(weights, start=1):
        cumulative += weight
        if roll <= cumulative:
            offset = step
            break

    target_idx = min(base_idx + offset, len(candidates) - 1)
    if target_idx == base_idx:
        return best_switch_point, None

    next_sp = candidates[target_idx]
    return next_sp, next_sp


def ensure_q_entries(q_table: Dict[Tuple[int, int], float], state: int, initial_q: float) -> None:
    if (state, 1) not in q_table:
        q_table[(state, 1)] = initial_q
    if (state, -1) not in q_table:
        q_table[(state, -1)] = initial_q


def build_episode_sequences(
    weight_sequence: List[int],
    current_sp: int,
    final_weight: int,
    safe_min: int,
    safe_max: int,
    overflow_penalty_constant: float,
    underflow_penalty_constant: float,
) -> Tuple[List[int], List[int], List[float]]:
    states: List[int] = []
    actions: List[int] = []
    rewards: List[float] = []

    post_switch = False
    for w in weight_sequence:
        if w == -1:
            post_switch = True
            continue
        if w == 300:
            break
        states.append(w)
        actions.append(-1 if post_switch else 1)
        rewards.append(0.0)

    if states:
        rewards[-1] += calc_reward_q(
            final_weight,
            safe_min,
            safe_max,
            overflow_penalty_constant,
            underflow_penalty_constant,
        )
    return states, actions, rewards


def main() -> None:
    cfg = load_config()
    rng_seed = int(cfg.get("seed", 42))
    random.seed(rng_seed)
    np.random.seed(rng_seed)

    logger, output_dir, _ = setup_legacy_training_logger(base_dir="outputs")
    paths = get_legacy_output_paths(output_dir)
    copy_config_to_output(CONFIG_PATH, output_dir)

    test_cfg = cfg.get("testing", {})
    episodes = int(test_cfg.get("episodes", 50))
    max_steps = int(test_cfg.get("max_steps_per_episode", 100))

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
    step_weights = hp.get("exploration_step_weights")

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
    update_counts = defaultdict(int)
    episode_records: List[Dict[str, Any]] = []

    try:
        for ep in range(episodes):
            experienced_sp = current_sp
            logger.info(f"--- Episode {ep + 1}/{episodes} ---")
            logger.info(f"Dispatching switching point: {experienced_sp}")

            modbus.send_switch_point(float(experienced_sp))

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

            states, actions, rewards = build_episode_sequences(
                weight_sequence,
                experienced_sp,
                final_weight,
                safe_min,
                safe_max,
                overflow_penalty_constant,
                underflow_penalty_constant,
            )

            for idx in range(len(states)):
                state = states[idx]
                action = actions[idx]
                reward = rewards[idx]
                ensure_q_entries(q_table, state, initial_q)
                if action == -1 and state not in positive_updates:
                    continue
                if idx + 1 < len(states):
                    next_state = states[idx + 1]
                    ensure_q_entries(q_table, next_state, initial_q)
                    best_next = max(
                        q_table[(next_state, 1)],
                        q_table[(next_state, -1)],
                    )
                else:
                    best_next = 0.0
                q_sa = q_table[(state, action)]
                td_target = reward + gamma * best_next
                q_table[(state, action)] = q_sa + alpha * (td_target - q_sa)
                update_counts[(state, action)] += 1
                if action == 1:
                    positive_updates.add(state)

            state_to_best: Dict[int, Tuple[int, float]] = {}
            for (state, action), value in q_table.items():
                best = state_to_best.get(state)
                if best is None or value > best[1]:
                    state_to_best[state] = (action, value)

            best_sp = experienced_sp
            for state in sorted(state_to_best.keys()):
                if state_to_best[state][0] == -1:
                    best_sp = state
                    break

            next_sp, explored_choice = _pick_next_switch_point(
                best_sp,
                known_sps,
                epsilon,
                step_weights,
            )

            epsilon = max(epsilon_min, epsilon * epsilon_decay)

            if safe_min <= final_weight <= safe_max:
                termination_type = "Normal"
            elif final_weight < safe_min:
                termination_type = "Underflow"
            else:
                termination_type = "Overflow"

            logger.info(f"Termination Type: {termination_type}")
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
                switch_point=int(experienced_sp),
                episode_length=meta.episode_length if meta and meta.episode_length is not None else max(1, len(weight_sequence)),
                final_weight=int(final_weight),
                safe_min=safe_min,
                safe_max=safe_max,
                q_values_snapshot=q_table.copy(),
                meta=meta,
            )

            episode_records.append(
                {
                    "episode_num": ep,
                    "experienced_switching_point": experienced_sp,
                    "model_selected_switching_point": best_sp,
                    "explored_switching_point": explored_choice,
                    "termination_type": termination_type,
                    "q_table": dict(q_table),
                    "counts": dict(update_counts),
                }
            )

            current_sp = next_sp

    finally:
        tcp.close()
        modbus.close()
        if db_handler:
            db_handler.close()

    if q_table:
        plot_qvalue_vs_state_from_pair_table(q_table, paths['qvalue_vs_state_path'])

    sp_bounds = (min(known_sps), max(known_sps)) if known_sps else (0, 1)
    plot_switching_trajectory_with_exploration(
        traj_ep,
        model_selected_list,
        explored_list,
        paths['switching_point_trajectory_path'],
        switch_point_bounds=sp_bounds,
    )

    if episode_records:
        excel_output_path = os.path.join(output_dir, "qlearning_qvalue_updates.xlsx")
        write_qtable_to_excel(episode_records, excel_output_path)
        logger.info("Saved Q-learning Q-value updates to %s", excel_output_path)

    metrics = {
        "episodes": episodes,
        "best_switch_point": model_selected_list[-1] if model_selected_list else None,
    }
    with open(os.path.join(output_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Finished testing. Metrics: %s", metrics)


if __name__ == "__main__":
    main()

# scripts/train_q.py
from __future__ import annotations

import json
import os
import random
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Set

import numpy as np
import yaml

from utils.data_processing import DataProcessor
from utils.excel_logging import write_qtable_to_excel
from utils.logging_utils import setup_legacy_training_logger, get_legacy_output_paths
from utils.plotting_utils import (
    plot_qvalue_vs_state_from_pair_table,
    plot_switching_trajectory_with_exploration,
)

def load_config() -> Dict[str, Any]:
    with open(os.path.join("configs", "q_train.yaml"), "r", encoding="utf-8") as f:
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

def main() -> None:
    cfg = load_config()
    rng_seed = int(cfg.get("seed", 42))
    random.seed(rng_seed)
    np.random.seed(rng_seed)

    logger, output_dir, _ = setup_legacy_training_logger(base_dir="outputs")
    paths = get_legacy_output_paths(output_dir)

    dcfg = cfg.get("data")
    excel_path = dcfg.get("path", "")
    dp = DataProcessor()
    dp.load_excel(excel_path)

    train_cfg = cfg.get("training", {})
    episodes = int(train_cfg.get("episodes"))

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

    available_weights = dp.get_all_available_weights()
    if not available_weights:
        raise RuntimeError("No weights parsed from dataset.")
    q_table: Dict[Tuple[int,int], float] = {}
    for w in available_weights:
        q_table[(w, 1)] = initial_q
        q_table[(w, -1)] = initial_q

    # Choose an initial switch point from data clusters
    available_sps = dp.get_available_switch_points()
    current_sp = starting_sp
    traj_ep: List[int] = []
    model_selected_list: List[int] = []
    explored_list: List[Optional[int]] = []
    update_counts = defaultdict(int)
    episode_records: List[Dict[str, Any]] = []
    positive_updates: Set[int] = set()

    for ep in range(episodes):
        experienced_sp = current_sp
        unused = dp.get_unused_sessions_for_switch_point(experienced_sp)
        s = random.choice(unused)
        dp.mark_session_as_used(experienced_sp, s)

        states: List[int] = []
        actions: List[int] = []
        rewards: List[float] = []
        final_weight = s.final_weight if s.final_weight is not None else 0

        for i, w in enumerate(s.weight_sequence):
            if w in (-1, 300): 
                continue
            if i + 1 < len(s.weight_sequence) and s.weight_sequence[i + 1] == 300:
                continue
            a = 1 if w < experienced_sp else -1
            states.append(w)
            actions.append(a)
            rewards.append(-1.0)

        if states:
            rewards[-1] += calc_reward_q(final_weight, safe_min, safe_max,
                                 overflow_penalty_constant, underflow_penalty_constant)

        for t in range(len(states)):
            s_t = states[t]
            a_t = actions[t]
            r_t = rewards[t]
            # if a_t == -1 and s_t not in positive_updates: # for update after first fast action
            #     continue
            q_fast = q_table.get((s_t, 1), initial_q)
            q_slow = q_table.get((s_t, -1), initial_q)
            best_next = max(q_fast, q_slow)
            q_sa = q_table[(s_t, a_t)]
            td_target = r_t + gamma * best_next
            q_table[(s_t, a_t)] = q_sa + alpha * (td_target - q_sa)
            update_counts[(s_t, a_t)] += 1
            if a_t == 1:
                positive_updates.add(s_t)

        state_to_best = {}
        for (w, a), v in q_table.items():
            best = state_to_best.get(w)
            if best is None or v > best[1]:
                state_to_best[w] = (a, v)
        best_sp = None
        for w in sorted(state_to_best.keys()):
            if state_to_best[w][0] == -1 and w in available_sps:
                best_sp = w
                break
        if best_sp is None:
            best_sp = current_sp

        if safe_min <= final_weight <= safe_max:
            termination_type = "Normal"
        elif final_weight < safe_min:
            termination_type = "Underflow"
        else:
            termination_type = "Overflow"

        logger.info(f"--- Episode {ep + 1}/{episodes} ---")
        logger.info(f"Experienced Switching Point: {experienced_sp}")
        logger.info(f"Termination Type: {termination_type}")

        next_sp, explored_choice = _pick_next_switch_point(
            best_sp,
            available_sps,
            epsilon,
            step_weights,
        )
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        logger.info(f"Model-Selected Next Switching Point: {best_sp}")
        logger.info(f"Explored Switching Point: {explored_choice}")
        logger.info("")

        traj_ep.append(ep + 1)
        model_selected_list.append(best_sp)
        explored_list.append(explored_choice)
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

    plot_qvalue_vs_state_from_pair_table(q_table, paths['qvalue_vs_state_path'])
    plot_switching_trajectory_with_exploration(traj_ep, model_selected_list, explored_list, paths['switching_point_trajectory_path'])

    if episode_records:
        excel_output_path = os.path.join(output_dir, "qlearning_qvalue_updates.xlsx")
        write_qtable_to_excel(episode_records, excel_output_path)
        logger.info("Saved Q-learning Q-value updates to %s", excel_output_path)

    metrics = {
        "episodes": episodes,
        "best_switch_point": best_sp,
    }
    with open(os.path.join(output_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Finished testing. Metrics: %s", metrics)

if __name__ == "__main__":
    main()

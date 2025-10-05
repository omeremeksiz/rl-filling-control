from __future__ import annotations

import os
import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml

from utils.data_processing import DataProcessor, SWITCH_TOKEN, TERMINATION_TOKEN
from utils.logging_utils import setup_legacy_training_logger, get_legacy_output_paths
from utils.plotting_utils import (
    plot_qvalue_vs_state_from_pair_table,
    plot_switching_trajectory_with_exploration,
)


def load_config() -> Dict[str, Any]:
    with open(os.path.join("configs", "q_train.yaml"), "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def calc_reward_standard(final_weight: int, safe_min: int = 80, safe_max: int = 120,
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

    excel_path = "/Users/omeremeksiz/Desktop/filling-control/rl-filling-control/data/data.xlsx"
    dp = DataProcessor()
    dp.load_excel(excel_path)

    train_cfg = cfg.get("training", {})
    episodes = int(train_cfg.get("episodes", 300))

    hp = cfg.get("hyperparameters", {})
    alpha = float(hp.get("alpha", 0.1))
    gamma = float(hp.get("gamma", 0.99))
    epsilon = float(hp.get("epsilon_start", 0.2))
    epsilon_min = float(hp.get("epsilon_min", 0.01))
    epsilon_decay = float(hp.get("epsilon_decay", 0.995))

    available_weights = dp.get_all_available_weights()
    q_table: Dict[Tuple[int,int], float] = {(w, 1): 0.0 for w in available_weights}
    q_table.update({(w, -1): 0.0 for w in available_weights})
    switch_points = dp.get_available_switch_points()
    current_switch_point = random.choice(switch_points)

    traj_ep: List[int] = []
    model_selected_list: List[int] = []
    explored_list: List[Optional[int]] = []

    for ep in range(episodes):
        unused = dp.get_unused_sessions_for_switch_point(current_switch_point)
        s = random.choice(unused)
        dp.mark_session_as_used(current_switch_point, s)
        final_weight = s.final_weight if s.final_weight is not None else 0

        states: List[int] = []
        actions: List[int] = []
        rewards: List[float] = []

        for i, w in enumerate(s.weight_sequence):
            if w in (SWITCH_TOKEN, TERMINATION_TOKEN):
                continue
            if i + 1 < len(s.weight_sequence) and s.weight_sequence[i + 1] == TERMINATION_TOKEN:
                continue
            a = 1 if w < current_switch_point else -1
            states.append(w)
            actions.append(a)
            rewards.append(0.0)

        if states:
            rewards[-1] += calc_reward_standard(final_weight)

        for t in range(len(states)):
            s_t = states[t]
            a_t = actions[t]
            r_t = rewards[t]
            q_fast = q_table.get((s_t, 1), 0.0)
            q_slow = q_table.get((s_t, -1), 0.0)
            best_next = max(q_fast, q_slow)
            q_sa = q_table[(s_t, a_t)]
            td_target = r_t + gamma * best_next
            q_table[(s_t, a_t)] = q_sa + alpha * (td_target - q_sa)

        state_to_best = {}
        for (w, a), v in q_table.items():
            best = state_to_best.get(w)
            if best is None or v > best[1]:
                state_to_best[w] = (a, v)
        best_sp = None
        for w in sorted(state_to_best.keys()):
            if state_to_best[w][0] == -1 and w in switch_points:
                best_sp = w
                break
        if best_sp is None:
            best_sp = current_switch_point

        explored_choice = None
        if random.random() < epsilon:
            idx = switch_points.index(current_switch_point)
            next_sp = switch_points[min(idx + 1, len(switch_points) - 1)]
            explored_choice = next_sp
        else:
            next_sp = best_sp
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        traj_ep.append(ep + 1)
        model_selected_list.append(best_sp)
        explored_list.append(explored_choice)
        current_switch_point = next_sp

    plot_qvalue_vs_state_from_pair_table(q_table, paths['qvalue_vs_state_path'])
    plot_switching_trajectory_with_exploration(traj_ep, model_selected_list, explored_list, paths['switching_point_trajectory_path'])


if __name__ == "__main__":
    main()

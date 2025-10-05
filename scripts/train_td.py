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
    with open(os.path.join("configs", "td_train.yaml"), "r", encoding="utf-8") as f:
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

    # Load data
    excel_path = "/Users/omeremeksiz/Desktop/filling-control/rl-filling-control/data/data.xlsx"
    dp = DataProcessor()
    dp.load_excel(excel_path)

    train_cfg = cfg.get("training", {})
    episodes = int(train_cfg.get("episodes", 200))

    hp = cfg.get("hyperparameters", {})
    alpha = float(hp.get("alpha", 0.1))
    gamma = float(hp.get("gamma", 0.99))

    # Initialize Q(s,a) over filling-process weights and actions {1,-1}
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

        # Build (state, action, reward) trajectory
        trajectory: List[Tuple[int,int,float]] = []
        for i, w in enumerate(s.weight_sequence):
            if w in (SWITCH_TOKEN, TERMINATION_TOKEN):
                continue
            # skip final weight position (just before TERMINATION_TOKEN)
            if i + 1 < len(s.weight_sequence) and s.weight_sequence[i + 1] == TERMINATION_TOKEN:
                continue
            a = 1 if w < current_switch_point else -1
            step_reward = 0.0
            trajectory.append((w, a, step_reward))

        final_weight = s.final_weight if s.final_weight is not None else 0
        final_reward = calc_reward_td(final_weight)
        if trajectory:
            w_last, a_last, r_last = trajectory[-1]
            trajectory[-1] = (w_last, a_last, r_last + final_reward)

        # TD (SARSA) updates
        for i in range(len(trajectory) - 1):
            s_t, a_t, r_t = trajectory[i]
            s_tp1, a_tp1, _ = trajectory[i + 1]
            q_sa = q_table[(s_t, a_t)]
            q_next = q_table[(s_tp1, a_tp1)]
            td_target = r_t + gamma * q_next
            q_table[(s_t, a_t)] = q_sa + alpha * (td_target - q_sa)
        if trajectory:
            s_T, a_T, r_T = trajectory[-1]
            q_last = q_table[(s_T, a_T)]
            q_table[(s_T, a_T)] = q_last + alpha * (r_T - q_last)

        # Derive next switch point: first weight whose best action is -1
        state_to_best = {}
        for (w, a), v in q_table.items():
            best = state_to_best.get(w)
            if best is None or v > best[1]:
                state_to_best[w] = (a, v)
        best_next_sp = current_switch_point
        for w in sorted(state_to_best.keys()):
            if state_to_best[w][0] == -1 and w in switch_points:
                best_next_sp = w
                break

        logger.info(f"--- Episode {ep + 1}/{episodes} ---")
        logger.info(f"Experienced Switching Point: {current_switch_point}")
        term = "safe" if (80 <= final_weight <= 120) else ("underweight" if final_weight < 80 else "overweight")
        logger.info(f"Termination Type: {term}")
        logger.info(f"Model-Selected Next Switching Point: {best_next_sp}")
        logger.info("")

        traj_ep.append(ep + 1)
        model_selected_list.append(best_next_sp)
        explored_list.append(None)
        current_switch_point = best_next_sp

    plot_qvalue_vs_state_from_pair_table(q_table, paths['qvalue_vs_state_path'])
    plot_switching_trajectory_with_exploration(traj_ep, model_selected_list, explored_list, paths['switching_point_trajectory_path'])


if __name__ == "__main__":
    main()

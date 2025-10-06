from __future__ import annotations

import json
import os
import random
from typing import Any, Dict, List, Optional, Tuple, Set

import numpy as np
import yaml

from utils.data_processing import DataProcessor
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

    dcfg = cfg.get("data")
    excel_path = dcfg.get("path", "")
    dp = DataProcessor()
    dp.load_excel(excel_path)

    train_cfg = cfg.get("training", {})
    episodes = int(train_cfg.get("episodes", 200))

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

    available_weights = dp.get_all_available_weights()
    if not available_weights:
        raise RuntimeError("No weights parsed from dataset.")
    q_table: Dict[Tuple[int,int], float] = {}
    for w in available_weights:
        q_table[(w, 1)] = initial_q
        q_table[(w, -1)] = initial_q

    available_sps = dp.get_available_switch_points()
    current_sp = starting_sp
    traj_ep: List[int] = []
    model_selected_list: List[int] = []
    explored_list: List[Optional[int]] = []
    positive_updates: Set[int] = set()

    for ep in range(episodes):
        unused = dp.get_unused_sessions_for_switch_point(current_sp)
        s = random.choice(unused)
        dp.mark_session_as_used(current_sp, s)

        # Build (state, action, reward) trajectory
        trajectory: List[Tuple[int,int,float]] = []
        final_weight = s.final_weight if s.final_weight is not None else 0
        for i, w in enumerate(s.weight_sequence):
            if w in (-1, 300):
                continue
            # skip final weight position (just before TERMINATION_TOKEN)
            if i + 1 < len(s.weight_sequence) and s.weight_sequence[i + 1] == 300:
                continue
            a = 1 if w < current_sp else -1
            step_reward = 0.0
            trajectory.append((w, a, step_reward))

        final_reward = calc_reward_td(
            final_weight,
            safe_min=safe_min,
            safe_max=safe_max,
            overflow_penalty_constant=overflow_penalty_constant,
            underflow_penalty_constant=underflow_penalty_constant,
        )
        if trajectory:
            w_last, a_last, r_last = trajectory[-1]
            trajectory[-1] = (w_last, a_last, r_last + final_reward)

        # TD (SARSA) updates
        for t in range(len(trajectory)):
            s_t, a_t, r_t = trajectory[t]
            if a_t == -1 and s_t not in positive_updates:
                continue
            if t + 1 < len(trajectory):
                s_tp1, a_tp1, _ = trajectory[t + 1]
                q_next = q_table[(s_tp1, a_tp1)]
                td_target = r_t + gamma * q_next
            else:
                td_target = r_t
            q_sa = q_table[(s_t, a_t)]
            q_table[(s_t, a_t)] = q_sa + alpha * (td_target - q_sa)
            if a_t == 1:
                positive_updates.add(s_t)

        # Derive next switch point: first weight whose best action is -1
        state_to_best = {}
        for (w, a), v in q_table.items():
            best = state_to_best.get(w)
            if best is None or v > best[1]:
                state_to_best[w] = (a, v)
        best_sp = current_sp
        for w in sorted(state_to_best.keys()):
            if state_to_best[w][0] == -1 and w in available_sps:
                best_sp = w
                break

        logger.info(f"--- Episode {ep + 1}/{episodes} ---")
        logger.info(f"Experienced Switching Point: {current_sp}")
        termination_type = "safe" if (safe_min <= final_weight <= safe_max) else ("underweight" if final_weight < safe_min else "overweight")
        logger.info(f"Termination Type: {termination_type}")

        explored_choice: Optional[int] = None
        if random.random() < epsilon:
            idx = available_sps.index(current_sp)
            target = min(idx + 1, len(available_sps) - 1)
            next_sp = available_sps[target]
            explored_choice = next_sp
        else:
            next_sp = best_sp

        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        logger.info(f"Model-Selected Next Switching Point: {best_sp}")
        logger.info(f"Explored Switching Point: {explored_choice}")
        logger.info("")

        traj_ep.append(ep + 1)
        model_selected_list.append(best_sp)
        explored_list.append(explored_choice)
        current_sp = next_sp

    plot_qvalue_vs_state_from_pair_table(q_table, paths['qvalue_vs_state_path'])
    plot_switching_trajectory_with_exploration(traj_ep, model_selected_list, explored_list, paths['switching_point_trajectory_path'])

    metrics = {
        "episodes": episodes,
        "best_switch_point": best_sp,
    }
    with open(os.path.join(output_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Finished testing. Metrics: %s", metrics)

if __name__ == "__main__":
    main()

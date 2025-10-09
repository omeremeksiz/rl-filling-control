# scripts/train_mc.py
from __future__ import annotations

import json
import os
import random
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Set

import numpy as np
import yaml

from utils.data_processing import DataProcessor
from utils.excel_logging import write_standart_q_table
from utils.logging_utils import setup_legacy_training_logger, get_legacy_output_paths
from utils.plotting_utils import (
    plot_qvalue_vs_state_from_pair_table,
    plot_switching_trajectory_with_exploration,
)

def load_config() -> Dict[str, Any]:
    with open(os.path.join("configs", "mc_train.yaml"), "r", encoding="utf-8") as f:
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

    # Monte Carlo over filling-process states with actions {1,-1}
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
        # Build trajectory: fast (1) before switch, slow (-1) after
        unused = dp.get_unused_sessions_for_switch_point(experienced_sp)
        s = random.choice(unused)
        dp.mark_session_as_used(experienced_sp, s)

        # Construct episodic (state, action, step_reward)
        trajectory: List[Tuple[int,int,float]] = []
        # Use simple step cost -1 like original MC style
        step_cost = -1.0
        # Fill until the state just before TERMINATION_TOKEN
        for w in s.weight_sequence[:-1]:
            if w in (-1, 300):
                continue
            a = 1 if w < experienced_sp else -1
            trajectory.append((w, a, step_cost))

        # Append terminal reward from final weight
        final_weight = s.final_weight
        final_reward = calc_reward_mc(final_weight, safe_min, safe_max,
                                 overflow_penalty_constant, underflow_penalty_constant)
        if trajectory:
            s_last, a_last, r_last = trajectory[-1]
            trajectory[-1] = (s_last, a_last, r_last + final_reward)

        # Compute returns and update MC
        G = 0.0
        for t in reversed(range(len(trajectory))):
            s_t, a_t, r_t = trajectory[t]
            G = r_t + gamma * G
            if a_t == -1 and s_t not in positive_updates:
                continue
            q_sa = q_table[(s_t, a_t)]
            q_table[(s_t, a_t)] = q_sa + alpha * (G - q_sa)
            update_counts[(s_t, a_t)] += 1
            if a_t == 1:
                positive_updates.add(s_t)

        # Policy: flip point is first state whose best action becomes -1
        # Find best switch from policy derived from q_table
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

        if safe_min <= final_weight <= safe_max:
            termination_type = "Normal"
        elif final_weight < safe_min:
            termination_type = "Underflow"
        else:
            termination_type = "Overflow"

        logger.info(f"--- Episode {ep + 1}/{episodes} ---")
        logger.info(f"Experienced Switching Point: {experienced_sp}")
        logger.info(f"Termination Type: {termination_type}")

        explored_choice: Optional[int] = None
        if random.random() < epsilon:
            idx = available_sps.index(experienced_sp)
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
        excel_output_path = os.path.join(output_dir, "mc_qvalue_updates.xlsx")
        write_standart_q_table(episode_records, excel_output_path)
        logger.info("Saved MC Q-value updates to %s", excel_output_path)

    metrics = {
        "episodes": episodes,
        "best_switch_point": best_sp,
    }
    with open(os.path.join(output_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Finished testing. Metrics: %s", metrics)

if __name__ == "__main__":
    main()

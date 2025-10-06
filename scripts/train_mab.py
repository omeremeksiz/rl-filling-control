from __future__ import annotations

import json
import os
import random
from typing import Any, Dict, List, Optional

import numpy as np
import yaml
from utils.data_processing import DataProcessor
from utils.logging_utils import setup_legacy_training_logger, get_legacy_output_paths
from utils.plotting_utils import (
    plot_qvalue_vs_state_bandit,
    plot_switching_trajectory_with_exploration,
)

def load_config() -> Dict[str, Any]:
    with open(os.path.join("configs", "mab_train.yaml"), "r", encoding="utf-8") as f:
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

    # Load training data (absolute path enforced)
    dcfg = cfg.get("data")
    excel_path = dcfg.get("path", "")
    dp = DataProcessor()
    dp.load_excel(excel_path)

    train_cfg = cfg.get("training", {})
    episodes = int(train_cfg.get("episodes"))

    hp = cfg.get("hyperparameters", {})
    alpha = float(hp.get("alpha"))
    epsilon = float(hp.get("epsilon_start"))
    epsilon_min = float(hp.get("epsilon_min"))
    epsilon_decay = float(hp.get("epsilon_decay"))
    underflow_penalty_constant = float(hp.get("underflow_penalty_constant"))
    overflow_penalty_constant = float(hp.get("overflow_penalty_constant"))
    safe_min = int(hp.get("safe_min"))
    safe_max = int(hp.get("safe_max"))  
    starting_sp = int(hp.get("starting_switch_point"))

    # MAB logic: bandit update over switch points
    available_sps = dp.get_available_switch_points()
    if not available_sps:
        raise RuntimeError("No switch points available from dataset.")
    q_table: Dict[int, float] = {sp: 0.0 for sp in available_sps}

    current_sp = starting_sp
    trajectory_ep: List[int] = []
    model_selected_list: List[int] = []
    explored_list: List[Optional[int]] = []

    for ep in range(episodes):
        unused = dp.get_unused_sessions_for_switch_point(current_sp)
        selected = random.choice(unused)
        dp.mark_session_as_used(current_sp, selected)

        episode_length = selected.episode_length
        final_weight = selected.final_weight
        reward = calc_reward_mab(episode_length, final_weight, safe_min, safe_max,
                                 overflow_penalty_constant, underflow_penalty_constant)

        # Bandit update: Q(sp) <- Q(sp) + alpha * (reward - Q(sp))
        q_table[current_sp] = q_table[current_sp] + alpha * (reward - q_table[current_sp])

        best_sp = max(q_table, key=q_table.get)
        termination_type = "safe" if (safe_min <= final_weight <= safe_max) else ("underweight" if final_weight < safe_min else "overweight")
        logger.info(f"--- Episode {ep + 1}/{episodes} ---")
        logger.info(f"Experienced Switching Point: {current_sp}")
        logger.info(f"Termination Type: {termination_type}")

        # Epsilon-greedy next action: step +1 from current or best
        explored_choice = None
        if random.random() < epsilon:
            idx = available_sps.index(current_sp)
            target = min(idx + 1, len(available_sps) - 1)
            next_sp = available_sps[target]
            explored_choice = next_sp
        else:
            next_sp = best_sp

        logger.info(f"Model-Selected Next Switching Point: {best_sp}")
        logger.info(f"Explored Switching Point: {explored_choice}")
        logger.info("")

        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        trajectory_ep.append(ep + 1)
        model_selected_list.append(best_sp)
        explored_list.append(explored_choice)
        current_sp = next_sp

    # Legacy plots
    plot_qvalue_vs_state_bandit(q_table, paths['qvalue_vs_state_path'])
    plot_switching_trajectory_with_exploration(trajectory_ep, model_selected_list, explored_list, paths['switching_point_trajectory_path'])

    metrics = {
        "episodes": episodes,
        "best_switch_point": int(max(q_table, key=q_table.get)) if q_table else None,
    }
    with open(os.path.join(output_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Finished testing. Metrics: %s", metrics)


if __name__ == "__main__":
    main()

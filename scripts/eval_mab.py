"""Compare Multi-Armed Bandit configurations on the filling-control dataset."""
from __future__ import annotations

import copy
import json
import os
import random
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np
import yaml

from utils.data_processing import DataProcessor
from utils.logging_utils import (
    setup_legacy_training_logger,
    copy_config_to_output,
)
from utils.plotting_utils import (
    plot_multi_switching_trajectory,
    plot_multi_qvalue_vs_state,
)

CONFIG_PATH = os.path.join("configs", "mab_eval.yaml")


def load_config(path: str = CONFIG_PATH) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def calc_reward_mab(
    episode_length: int,
    final_weight: int,
    safe_min: int,
    safe_max: int,
    overflow_penalty_constant: float,
    underflow_penalty_constant: float,
) -> float:
    base_reward = -episode_length
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
    if random.random() >= epsilon or not candidates:
        return best_switch_point, None

    base_idx = candidates.index(best_switch_point)
    if not weights:
        return best_switch_point, None

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


def _merge_dicts(base: Mapping[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    result: Dict[str, Any] = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge_dicts(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _select_starting_sp(
    starting_sp: int,
    available_sps: List[int],
) -> int:
    if starting_sp in available_sps:
        return starting_sp
    if not available_sps:
        raise RuntimeError("No switch points available from dataset.")
    return min(available_sps, key=lambda sp: abs(sp - starting_sp))


def run_experiment(
    name: str,
    seed: int,
    episodes: int,
    dp: DataProcessor,
    hyperparameters: Mapping[str, Any],
    logger,
) -> Dict[str, Any]:
    random.seed(seed)
    np.random.seed(seed)

    available_sps = dp.get_available_switch_points()
    if not available_sps:
        raise RuntimeError("No switch points available from dataset.")
    available_sps = sorted(available_sps)
    q_table: Dict[int, float] = {sp: 0.0 for sp in available_sps}

    alpha = float(hyperparameters.get("alpha", 0.4))
    epsilon = float(hyperparameters.get("epsilon_start", 0.5))
    epsilon_min = float(hyperparameters.get("epsilon_min", 0.0))
    epsilon_decay = float(hyperparameters.get("epsilon_decay", 1.0))
    under_penalty = float(hyperparameters.get("underflow_penalty_constant", 0.0))
    over_penalty = float(hyperparameters.get("overflow_penalty_constant", 0.0))
    safe_min = int(hyperparameters.get("safe_min", 74))
    safe_max = int(hyperparameters.get("safe_max", 76))
    starting_sp = int(hyperparameters.get("starting_switch_point", available_sps[0]))
    step_weights = hyperparameters.get("exploration_step_weights", [])
    if isinstance(step_weights, list):
        step_weights = [float(x) for x in step_weights]
    else:
        step_weights = []

    current_sp = _select_starting_sp(starting_sp, available_sps)

    episode_nums: List[int] = []
    model_selected_list: List[int] = []
    update_counts = defaultdict(int)
    q_history: List[Dict[int, float]] = []

    for ep in range(episodes):
        experienced_sp = current_sp
        unused = dp.get_unused_sessions_for_switch_point(experienced_sp)
        selected_session = random.choice(unused)
        dp.mark_session_as_used(experienced_sp, selected_session)

        episode_length = selected_session.episode_length
        final_weight = selected_session.final_weight
        reward = calc_reward_mab(
            episode_length,
            final_weight,
            safe_min,
            safe_max,
            over_penalty,
            under_penalty,
        )

        update_counts[experienced_sp] += 1
        n = update_counts[experienced_sp]
        q_table[experienced_sp] = q_table[experienced_sp] + (1 / n) * (reward - q_table[experienced_sp])

        best_sp = max(q_table, key=q_table.get)

        next_sp, _ = _pick_next_switch_point(best_sp, available_sps, epsilon, step_weights)
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        episode_nums.append(ep + 1)
        model_selected_list.append(best_sp)
        q_history.append(dict(q_table))

        current_sp = next_sp

        logger.info(
            "[%s] Episode %d/%d - experienced=%s best=%s epsilon=%.4f",
            name,
            ep + 1,
            episodes,
            experienced_sp,
            best_sp,
            epsilon,
        )

    summary = {
        "name": name,
        "episode_numbers": episode_nums,
        "model_selected": model_selected_list,
        "q_table": q_table,
        "best_switch_point": max(q_table, key=q_table.get) if q_table else None,
        "q_history": q_history,
        "seed": seed,
    }
    return summary


def main() -> None:
    cfg = load_config()
    outputs_dir = cfg.get("outputs_dir", "outputs")

    logger, output_dir, _ = setup_legacy_training_logger(base_dir=outputs_dir)
    copy_config_to_output(CONFIG_PATH, output_dir, destination_name="mab_eval_config.yaml")

    data_cfg = cfg.get("data", {})
    data_path = data_cfg.get("path", "")
    if not data_path:
        raise RuntimeError("Dataset path must be provided in mab_eval.yaml under data.path")

    defaults = cfg.get("defaults", {})
    experiments_cfg = cfg.get("experiments")
    if not experiments_cfg:
        raise RuntimeError("No experiments defined in mab_eval.yaml")

    results: List[Dict[str, Any]] = []
    base_seed = int(cfg.get("seed", 42))
    dataset_bounds: Optional[Tuple[int, int]] = None

    for idx, experiment in enumerate(experiments_cfg, start=1):
        name = experiment.get("name", f"experiment_{idx}")
        merged_cfg = _merge_dicts(defaults, experiment)
        train_cfg = merged_cfg.get("training", {})
        hyperparameters = merged_cfg.get("hyperparameters", {})
        episodes = int(train_cfg.get("episodes", 0))
        if episodes <= 0:
            raise RuntimeError(f"Experiment '{name}' must specify training.episodes > 0")
        seed = int(merged_cfg.get("seed", base_seed))

        dp = DataProcessor()
        dp.load_excel(data_path)

        available_sps = dp.get_available_switch_points()
        if not available_sps:
            raise RuntimeError("No switch points available from dataset.")
        if dataset_bounds is None:
            dataset_bounds = (min(available_sps), max(available_sps))

        logger.info("=== Running experiment: %s (seed=%d) ===", name, seed)
        summary = run_experiment(
            name=name,
            seed=seed,
            episodes=episodes,
            dp=dp,
            hyperparameters=hyperparameters,
            logger=logger,
        )
        results.append(summary)
        logger.info(
            "=== Completed experiment: %s | best_switch_point=%s ===",
            name,
            summary["best_switch_point"],
        )
        logger.info("")

    trajectory_data = {
        res["name"]: (res["episode_numbers"], res["model_selected"]) for res in results
    }
    qvalue_data = {res["name"]: res["q_table"] for res in results}
    best_sp_map = {res["name"]: res.get("best_switch_point") for res in results}

    comparison_paths = {
        "switching": os.path.join(output_dir, "mab_eval_switching_trajectory.png"),
        "qvalues": os.path.join(output_dir, "mab_eval_qvalue_comparison.png"),
    }

    plot_multi_switching_trajectory(
        trajectory_data,
        comparison_paths["switching"],
        switch_point_bounds=dataset_bounds,
    )
    plot_multi_qvalue_vs_state(
        qvalue_data,
        comparison_paths["qvalues"],
        best_switch_points=best_sp_map,
    )

    metrics_payload = {
        "experiments": [
            {
                "name": res["name"],
                "seed": res["seed"],
                "episodes": len(res["episode_numbers"]),
                "best_switch_point": res["best_switch_point"],
                "final_q_values": {
                    str(sp): val for sp, val in sorted(res["q_table"].items())
                },
            }
            for res in results
        ],
        "artifacts": comparison_paths,
    }

    metrics_path = os.path.join(output_dir, "mab_eval_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as handle:
        json.dump(metrics_payload, handle, indent=2)

    logger.info("Saved comparison plots and metrics to %s", output_dir)


if __name__ == "__main__":
    main()

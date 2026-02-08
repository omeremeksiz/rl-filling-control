# scripts/penalty_sweep.py
from __future__ import annotations

import copy
import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np
import yaml

from utils.data_processing import DataProcessor
from utils.plotting_utils import plot_penalty_sweep_best_switch_points

from scripts import eval_mab, eval_mc

CONFIG_PATH = os.path.join("configs", "penalty_sweep.yaml")


def load_config(path: str = CONFIG_PATH) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _resolve_seed_list(value: Any, fallback: Iterable[int]) -> List[int]:
    if value is None:
        return [int(seed) for seed in fallback]
    if isinstance(value, (list, tuple, set)):
        seeds = [int(seed) for seed in value]
    else:
        seeds = [int(value)]
    if not seeds:
        return [int(seed) for seed in fallback]
    return seeds


def _resolve_plot_format(value: Any, default: str = "png") -> str:
    if value is None:
        return default
    fmt = str(value).strip().lower().lstrip(".")
    if fmt not in {"pdf", "png"}:
        return default
    return fmt


def _build_penalty_values(start: int, end: int, step: int) -> List[int]:
    if step == 0:
        raise ValueError("Sweep step cannot be zero.")
    values: List[int] = []
    if step > 0:
        current = start
        while current <= end:
            values.append(current)
            current += step
    else:
        current = start
        while current >= end:
            values.append(current)
            current += step
    return values


def _select_experiment(experiments: Sequence[Mapping[str, Any]], name: Optional[str]) -> Mapping[str, Any]:
    if not experiments:
        raise RuntimeError("No experiments available to run.")
    if name:
        for exp in experiments:
            if exp.get("name") == name:
                return exp
    return experiments[0]


def _run_method_sweep(
    *,
    method: str,
    cfg: Mapping[str, Any],
    seeds: Sequence[int],
    penalties: Sequence[int],
    penalty_target: str,
    experiment_name: Optional[str],
    episodes_override: Optional[int],
    logger: logging.Logger,
) -> Dict[int, List[float]]:
    data_cfg = cfg.get("data", {})
    data_path = data_cfg.get("path", "")
    if not data_path:
        raise RuntimeError(f"{method} config missing data.path")

    defaults = cfg.get("defaults", {})
    experiments_cfg = cfg.get("experiments", [])
    if not experiments_cfg:
        raise RuntimeError(f"{method} config missing experiments")

    experiment = _select_experiment(experiments_cfg, experiment_name)
    merged_cfg = eval_mab._merge_dicts(defaults, experiment)  # type: ignore[attr-defined]
    train_cfg = merged_cfg.get("training", {})
    hyperparameters = merged_cfg.get("hyperparameters", {})
    episodes = int(train_cfg.get("episodes", 0))
    if episodes_override is not None:
        episodes = int(episodes_override)
    if episodes <= 0:
        raise RuntimeError(f"{method} experiment must specify training.episodes > 0")

    results: Dict[int, List[float]] = {}
    for penalty in penalties:
        penalty_value = float(penalty)
        for seed in seeds:
            dp = DataProcessor()
            dp.load_excel(data_path)

            hp = copy.deepcopy(hyperparameters)
            if penalty_target in {"underflow", "both"}:
                hp["underflow_penalty_constant"] = penalty_value
            if penalty_target in {"overflow", "both"}:
                hp["overflow_penalty_constant"] = penalty_value

            run_name = f"{method}_{experiment.get('name', 'experiment')}_penalty_{penalty}_seed_{seed}"
            logger.info("[%s] penalty=%s seed=%s", method, penalty, seed)

            if method == "MAB":
                summary = eval_mab.run_experiment(  # type: ignore[attr-defined]
                    name=run_name,
                    seed=seed,
                    episodes=episodes,
                    dp=dp,
                    hyperparameters=hp,
                    logger=logger,
                )
            else:
                summary = eval_mc.run_experiment(  # type: ignore[attr-defined]
                    name=run_name,
                    seed=seed,
                    episodes=episodes,
                    dp=dp,
                    hyperparameters=hp,
                    logger=logger,
                )

            best_sp = summary.get("best_switch_point")
            if best_sp is None:
                continue
            results.setdefault(penalty, []).append(float(best_sp))
    return results


def main() -> None:
    cfg = load_config()
    outputs_dir = cfg.get("outputs_dir", "outputs")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(outputs_dir, f"penalty_sweep_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    log_path = os.path.join(output_dir, "penalty_sweep.log")
    logger = logging.getLogger(f"penalty_sweep_{timestamp}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    handler = logging.FileHandler(log_path)
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(handler)
    logger.propagate = False

    data_cfg = cfg.get("data", {})
    mab_config_path = data_cfg.get("mab_config", eval_mab.CONFIG_PATH)
    mc_config_path = data_cfg.get("mc_config", eval_mc.CONFIG_PATH)

    mab_cfg = eval_mab.load_config(mab_config_path)
    mc_cfg = eval_mc.load_config(mc_config_path)

    sweep_cfg = cfg.get("sweep", {})
    penalty_target = str(sweep_cfg.get("target", "both")).lower()
    if penalty_target not in {"underflow", "overflow", "both"}:
        raise RuntimeError("sweep.target must be one of: underflow, overflow, both")
    schedule = sweep_cfg.get("schedule")
    if schedule:
        penalties: List[int] = []
        for seg in schedule:
            seg_start = int(seg.get("start", 0))
            seg_end = int(seg.get("end", 0))
            seg_step = int(seg.get("step", -10))
            penalties.extend(_build_penalty_values(seg_start, seg_end, seg_step))
        if penalties:
            reverse = penalties[0] > penalties[-1] if len(penalties) > 1 else False
            penalties = sorted(set(penalties), reverse=reverse)
    else:
        start = int(sweep_cfg.get("start", 0))
        end = int(sweep_cfg.get("end", -3000))
        step = int(sweep_cfg.get("step", -10))
        penalties = _build_penalty_values(start, end, step)
    if not penalties:
        raise RuntimeError("No penalties generated; check sweep start/end/step.")

    episodes_override = sweep_cfg.get("episodes")

    sweep_seeds = cfg.get("seeds")
    mab_seeds = _resolve_seed_list(sweep_seeds, _resolve_seed_list(mab_cfg.get("seeds"), [int(mab_cfg.get("seed", 42))]))
    mc_seeds = _resolve_seed_list(sweep_seeds, _resolve_seed_list(mc_cfg.get("seeds"), [int(mc_cfg.get("seed", 42))]))

    mab_results = _run_method_sweep(
        method="MAB",
        cfg=mab_cfg,
        seeds=mab_seeds,
        penalties=penalties,
        penalty_target=penalty_target,
        experiment_name=None,
        episodes_override=episodes_override,
        logger=logger,
    )
    mc_results = _run_method_sweep(
        method="MC",
        cfg=mc_cfg,
        seeds=mc_seeds,
        penalties=penalties,
        penalty_target=penalty_target,
        experiment_name=None,
        episodes_override=episodes_override,
        logger=logger,
    )

    plot_format = _resolve_plot_format(cfg.get("plot_format"), default="png")
    plot_cfg = cfg.get("plot", {})
    x_tick_step = int(plot_cfg.get("x_tick_step", 0))
    x_tick_schedule = plot_cfg.get("x_tick_schedule")
    plot_path = os.path.join(output_dir, f"penalty_sweep_best_switch_points.{plot_format}")

    plot_penalty_sweep_best_switch_points(
        {
            "MAB": mab_results,
            "MC": mc_results,
        },
        plot_path,
        penalty_order=penalties,
        x_tick_step=x_tick_step,
        x_tick_schedule=x_tick_schedule,
        show_legend=True,
    )

    metrics = {
        "timestamp": timestamp,
        "penalties": penalties,
        "seeds": {
            "mab": mab_seeds,
            "mc": mc_seeds,
        },
        "target": penalty_target,
        "mab_results": mab_results,
        "mc_results": mc_results,
        "artifacts": {
            "plot": plot_path,
        },
    }
    metrics_path = os.path.join(output_dir, "penalty_sweep_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    logger.info("Saved penalty sweep outputs to %s", output_dir)


if __name__ == "__main__":
    main()

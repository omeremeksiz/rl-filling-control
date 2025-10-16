# scripts/eval_compare.py
from __future__ import annotations

import json
import os
import re
from datetime import datetime
from typing import Any, Dict, List, Mapping, Optional, Tuple

from utils.data_processing import DataProcessor
import logging

from utils.logging_utils import copy_config_to_output
from utils.plotting_utils import (
    plot_multi_switching_trajectory,
    plot_multi_qvalue_vs_state,
    plot_multi_qvalue_pair_tables,
    plot_compare_method_switch_points,
)

from scripts import eval_mab, eval_mc


def _slugify(label: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", label.strip())
    slug = re.sub(r"_+", "_", slug)
    slug = slug.strip("_").lower()
    return slug or "configuration"


def _normalise_config_name(name: str, method_key: str) -> Tuple[str, str]:
    method_key = method_key.lower()
    normalised = re.sub(rf"^(?:{method_key}[\s_\-]+)", "", name, flags=re.IGNORECASE).strip()
    if not normalised:
        normalised = name
    display = normalised.replace("_", " ").replace("-", " ").strip()
    display = display or name.replace("_", " ").replace("-", " ")
    slug = _slugify(normalised) if normalised else _slugify(name)
    return slug, display or name


def _run_mab(cfg: Dict[str, Any], logger, base_seed: int) -> Tuple[List[Dict[str, Any]], Tuple[int, int]]:
    outputs_dir = cfg.get("outputs_dir", "outputs")
    data_cfg = cfg.get("data", {})
    data_path = data_cfg.get("path", "")
    if not data_path:
        raise RuntimeError("Dataset path must be provided in mab_eval.yaml under data.path")

    defaults = cfg.get("defaults", {})
    experiments_cfg = cfg.get("experiments")
    if not experiments_cfg:
        raise RuntimeError("No experiments defined in mab_eval.yaml")

    results: List[Dict[str, Any]] = []
    dataset_bounds: Optional[Tuple[int, int]] = None

    for idx, experiment in enumerate(experiments_cfg, start=1):
        name = experiment.get("name", f"mab_experiment_{idx}")
        merged_cfg = eval_mab._merge_dicts(defaults, experiment)  # type: ignore[attr-defined]
        train_cfg = merged_cfg.get("training", {})
        hyperparameters = merged_cfg.get("hyperparameters", {})
        episodes = int(train_cfg.get("episodes", 0))
        if episodes <= 0:
            raise RuntimeError(f"[MAB] Experiment '{name}' must specify training.episodes > 0")
        seed = int(merged_cfg.get("seed", base_seed))

        dp = DataProcessor()
        dp.load_excel(data_path)

        available_sps = dp.get_available_switch_points()
        if not available_sps:
            raise RuntimeError("[MAB] No switching points available from dataset.")
        if dataset_bounds is None:
            dataset_bounds = (min(available_sps), max(available_sps))

        logger.info("[MAB] Running experiment %s (seed=%d)", name, seed)
        summary = eval_mab.run_experiment(  # type: ignore[attr-defined]
            name=name,
            seed=seed,
            episodes=episodes,
            dp=dp,
            hyperparameters=hyperparameters,
            logger=logger,
        )
        results.append(summary)
        logger.info("[MAB] Completed experiment %s | best_switching_point=%s", name, summary["best_switch_point"])

    return results, dataset_bounds or (0, 1)


def _run_mc(cfg: Dict[str, Any], logger, base_seed: int) -> Tuple[List[Dict[str, Any]], Tuple[int, int]]:
    outputs_dir = cfg.get("outputs_dir", "outputs")
    data_cfg = cfg.get("data", {})
    data_path = data_cfg.get("path", "")
    if not data_path:
        raise RuntimeError("Dataset path must be provided in mc_eval.yaml under data.path")

    defaults = cfg.get("defaults", {})
    experiments_cfg = cfg.get("experiments")
    if not experiments_cfg:
        raise RuntimeError("No experiments defined in mc_eval.yaml")

    results: List[Dict[str, Any]] = []
    dataset_bounds: Optional[Tuple[int, int]] = None

    for idx, experiment in enumerate(experiments_cfg, start=1):
        name = experiment.get("name", f"mc_experiment_{idx}")
        merged_cfg = eval_mc._merge_dicts(defaults, experiment)  # type: ignore[attr-defined]
        train_cfg = merged_cfg.get("training", {})
        hyperparameters = merged_cfg.get("hyperparameters", {})
        episodes = int(train_cfg.get("episodes", 0))
        if episodes <= 0:
            raise RuntimeError(f"[MC] Experiment '{name}' must specify training.episodes > 0")
        seed = int(merged_cfg.get("seed", base_seed))

        dp = DataProcessor()
        dp.load_excel(data_path)

        available_sps = dp.get_available_switch_points()
        if not available_sps:
            raise RuntimeError("[MC] No switching points available from dataset.")
        if dataset_bounds is None:
            dataset_bounds = (min(available_sps), max(available_sps))

        logger.info("[MC] Running experiment %s (seed=%d)", name, seed)
        summary = eval_mc.run_experiment(  # type: ignore[attr-defined]
            name=name,
            seed=seed,
            episodes=episodes,
            dp=dp,
            hyperparameters=hyperparameters,
            logger=logger,
        )
        results.append(summary)
        logger.info("[MC] Completed experiment %s | best_switching_point=%s", name, summary["best_switch_point"])

    return results, dataset_bounds or (0, 1)


def main() -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("outputs", f"compare_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    qvalues_dir = os.path.join(output_dir, "qvalues")
    sp_trajectory_dir = os.path.join(output_dir, "sp_trajectory")
    os.makedirs(qvalues_dir, exist_ok=True)
    os.makedirs(sp_trajectory_dir, exist_ok=True)

    log_path = os.path.join(output_dir, "compare.log")
    logger = logging.getLogger(f"compare_{timestamp}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    handler = logging.FileHandler(log_path)
    handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)
    logger.propagate = False

    mab_cfg = eval_mab.load_config()
    mc_cfg = eval_mc.load_config()

    copy_config_to_output(eval_mab.CONFIG_PATH, output_dir, destination_name="mab_eval_config.yaml")
    copy_config_to_output(eval_mc.CONFIG_PATH, output_dir, destination_name="mc_eval_config.yaml")

    base_seed = int(mab_cfg.get("seed", 42))

    mab_results, mab_bounds = _run_mab(mab_cfg, logger, base_seed)
    mc_results, mc_bounds = _run_mc(mc_cfg, logger, base_seed)

    combined_bounds = (
        min(mab_bounds[0], mc_bounds[0]),
        max(mab_bounds[1], mc_bounds[1]),
    )

    trajectory_data = {
        f"MAB - {res['name']}": (res["episode_numbers"], res["model_selected"])
        for res in mab_results
    }
    trajectory_data.update(
        {
            f"MC - {res['name']}": (res["episode_numbers"], res["model_selected"])
            for res in mc_results
        }
    )

    individual_qvalue_paths: Dict[str, str] = {}
    individual_switching_paths: Dict[str, str] = {}
    config_display: Dict[str, str] = {}
    mab_switch_points: Dict[str, Optional[int]] = {}
    mc_switch_points: Dict[str, Optional[int]] = {}

    comparison_paths = {
        "switching": os.path.join(output_dir, "compare_switching_trajectory.png"),
        "best_switch_points": os.path.join(output_dir, "compare_best_switch_points.png"),
    }

    plot_multi_switching_trajectory(
        trajectory_data,
        comparison_paths["switching"],
        switch_point_bounds=combined_bounds,
    )
    for idx, res in enumerate(mab_results, start=1):
        display_name = res.get("name") or f"MAB Experiment {idx}"
        label = f"MAB - {display_name}"
        base_stub = f"mab_{_slugify(display_name)}"
        file_stub = base_stub
        suffix = 1
        while file_stub in individual_qvalue_paths or file_stub in individual_switching_paths:
            suffix += 1
            file_stub = f"{base_stub}_{suffix}"

        config_slug, config_label = _normalise_config_name(display_name, "mab")
        base_config_slug = config_slug
        suffix = 1
        while config_slug in config_display and config_display[config_slug].lower() != config_label.lower():
            suffix += 1
            config_slug = f"{base_config_slug}_{suffix}"
            if not config_label.lower().endswith("(mab)"):
                config_label = f"{config_label} (MAB)"
        config_display.setdefault(config_slug, config_label)
        mab_switch_points[config_slug] = res.get("best_switch_point")

        trajectory_path = os.path.join(sp_trajectory_dir, f"{file_stub}.png")
        plot_multi_switching_trajectory(
            {label: (res.get("episode_numbers", []), res.get("model_selected", []))},
            trajectory_path,
            switch_point_bounds=combined_bounds,
        )
        individual_switching_paths[file_stub] = trajectory_path

        qvalues_path = os.path.join(qvalues_dir, f"{file_stub}.png")
        plot_multi_qvalue_vs_state(
            {label: res.get("q_table", {})},
            qvalues_path,
            best_switch_points={label: res.get("best_switch_point")},
        )
        individual_qvalue_paths[file_stub] = qvalues_path

    for idx, res in enumerate(mc_results, start=1):
        display_name = res.get("name") or f"MC Experiment {idx}"
        label = f"MC - {display_name}"
        base_stub = f"mc_{_slugify(display_name)}"
        file_stub = base_stub
        suffix = 1
        while file_stub in individual_qvalue_paths or file_stub in individual_switching_paths:
            suffix += 1
            file_stub = f"{base_stub}_{suffix}"

        config_slug, config_label = _normalise_config_name(display_name, "mc")
        base_config_slug = config_slug
        suffix = 1
        while config_slug in config_display and config_display[config_slug].lower() != config_label.lower():
            suffix += 1
            config_slug = f"{base_config_slug}_{suffix}"
            if not config_label.lower().endswith("(mc)"):
                config_label = f"{config_label} (MC)"
        config_display.setdefault(config_slug, config_label)
        mc_switch_points[config_slug] = res.get("best_switch_point")

        trajectory_path = os.path.join(sp_trajectory_dir, f"{file_stub}.png")
        plot_multi_switching_trajectory(
            {label: (res.get("episode_numbers", []), res.get("model_selected", []))},
            trajectory_path,
            switch_point_bounds=combined_bounds,
        )
        individual_switching_paths[file_stub] = trajectory_path

        qvalues_path = os.path.join(qvalues_dir, f"{file_stub}.png")
        plot_multi_qvalue_pair_tables(
            {label: res.get("q_table", {})},
            qvalues_path,
            best_switch_points={label: res.get("best_switch_point")},
        )
        individual_qvalue_paths[file_stub] = qvalues_path

    plot_compare_method_switch_points(
        {
            "MAB": mab_switch_points,
            "MC": mc_switch_points,
        },
        config_display,
        comparison_paths["best_switch_points"],
    )

    def _serialise_mab(res: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "name": res.get("name"),
            "seed": res.get("seed"),
            "episodes": len(res.get("episode_numbers", [])),
            "best_switching_point": res.get("best_switch_point"),
            "q_table": {str(k): float(v) for k, v in res.get("q_table", {}).items()},
            "trajectory": res.get("model_selected", []),
        }

    def _serialise_mc(res: Dict[str, Any]) -> Dict[str, Any]:
        q_table = res.get("q_table", {})
        serialised_q = {
            str(state): {str(action): float(value) for action, value in ((1, q_table.get((state, 1), 0.0)), (-1, q_table.get((state, -1), 0.0)))}
            for state in sorted({s for (s, _) in q_table.keys()})
        }
        q_history_serialised = [
            {f"{state}:{action}": float(value) for (state, action), value in snapshot.items()}
            for snapshot in res.get("q_history", [])
        ]
        update_counts = {
            f"{state}:{action}": int(count)
            for (state, action), count in res.get("update_counts", {}).items()
        }
        return {
            "name": res.get("name"),
            "seed": res.get("seed"),
            "episodes": len(res.get("episode_numbers", [])),
            "best_switching_point": res.get("best_switch_point"),
            "trajectory": res.get("model_selected", []),
            "q_table": serialised_q,
            "q_history": q_history_serialised,
            "update_counts": update_counts,
        }

    metrics_payload = {
        "timestamp": timestamp,
        "mab_experiments": [_serialise_mab(res) for res in mab_results],
        "mc_experiments": [_serialise_mc(res) for res in mc_results],
        "artifacts": comparison_paths,
        "individual_artifacts": {
            "qvalues": individual_qvalue_paths,
            "switching_trajectories": individual_switching_paths,
        },
    }
    metrics_path = os.path.join(output_dir, "compare_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as handle:
        json.dump(metrics_payload, handle, indent=2)

    logger.info("Saved MAB/MC comparison outputs to %s", output_dir)


if __name__ == "__main__":
    main()

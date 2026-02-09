# scripts/eval_compare_train.py
from __future__ import annotations

import json
import math
import os
import re
from datetime import datetime
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from utils.data_processing import DataProcessor
import logging

from utils.logging_utils import copy_config_to_output
from utils.plotting_utils import (
    plot_switching_trajectory_with_exploration,
    plot_multi_qvalue_vs_state,
    plot_multi_qvalue_pair_tables,
    plot_compare_method_switch_points,
    plot_summary_switching_trajectory,
)

from scripts import eval_mab_train as eval_mab, eval_mc_train as eval_mc
import numpy as np


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


def _infer_tick_step(values: Iterable[int], max_labels: int = 15) -> Optional[int]:
    items = sorted({int(val) for val in values})
    if not items:
        return None
    if len(items) <= max_labels:
        return 1
    span = items[-1] - items[0]
    if span <= 0:
        return 1
    return max(1, int(math.ceil(span / max(1, max_labels - 1))))


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


def _normalise_plot_format(value: Any) -> Optional[str]:
    if value is None:
        return None
    fmt = str(value).strip().lower().lstrip(".")
    if fmt not in {"pdf", "png"}:
        return None
    return fmt


def _normalise_bool(value: Any) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "1", "on"}:
            return True
        if lowered in {"false", "no", "0", "off"}:
            return False
    return None


def _resolve_plot_format(*values: Any, default: str = "pdf") -> str:
    for value in values:
        fmt = _normalise_plot_format(value)
        if fmt is not None:
            return fmt
    return default


def _run_mab(
    cfg: Dict[str, Any],
    logger,
    base_seeds: Iterable[int],
) -> Tuple[List[Dict[str, Any]], Tuple[int, int], Dict[str, List[Dict[str, Any]]]]:
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
    results_by_name: Dict[str, List[Dict[str, Any]]] = {}

    for idx, experiment in enumerate(experiments_cfg, start=1):
        name = experiment.get("name", f"mab_experiment_{idx}")
        merged_cfg = eval_mab._merge_dicts(defaults, experiment)  # type: ignore[attr-defined]
        train_cfg = merged_cfg.get("training", {})
        hyperparameters = merged_cfg.get("hyperparameters", {})
        episodes = int(train_cfg.get("episodes", 0))
        if episodes <= 0:
            raise RuntimeError(f"[MAB] Experiment '{name}' must specify training.episodes > 0")
        seed_override = merged_cfg.get("seeds")
        if seed_override is None and "seed" in merged_cfg:
            seed_override = merged_cfg.get("seed")
        seeds = _resolve_seed_list(seed_override, base_seeds)

        for seed in seeds:
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
            results_by_name.setdefault(name, []).append(summary)
            logger.info("[MAB] Completed experiment %s | best_switching_point=%s", name, summary["best_switch_point"])

    return results, dataset_bounds or (0, 1), results_by_name


def _run_mc(
    cfg: Dict[str, Any],
    logger,
    base_seeds: Iterable[int],
) -> Tuple[List[Dict[str, Any]], Tuple[int, int], Dict[str, List[Dict[str, Any]]]]:
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
    results_by_name: Dict[str, List[Dict[str, Any]]] = {}

    for idx, experiment in enumerate(experiments_cfg, start=1):
        name = experiment.get("name", f"mc_experiment_{idx}")
        merged_cfg = eval_mc._merge_dicts(defaults, experiment)  # type: ignore[attr-defined]
        train_cfg = merged_cfg.get("training", {})
        hyperparameters = merged_cfg.get("hyperparameters", {})
        episodes = int(train_cfg.get("episodes", 0))
        if episodes <= 0:
            raise RuntimeError(f"[MC] Experiment '{name}' must specify training.episodes > 0")
        seed_override = merged_cfg.get("seeds")
        if seed_override is None and "seed" in merged_cfg:
            seed_override = merged_cfg.get("seed")
        seeds = _resolve_seed_list(seed_override, base_seeds)

        for seed in seeds:
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
            results_by_name.setdefault(name, []).append(summary)
            logger.info("[MC] Completed experiment %s | best_switching_point=%s", name, summary["best_switch_point"])

    return results, dataset_bounds or (0, 1), results_by_name


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

    mab_base_seeds = _resolve_seed_list(mab_cfg.get("seeds"), [int(mab_cfg.get("seed", 42))])
    mc_base_seeds = _resolve_seed_list(mc_cfg.get("seeds"), [int(mc_cfg.get("seed", 42))])

    mab_format = _normalise_plot_format(mab_cfg.get("plot_format"))
    mc_format = _normalise_plot_format(mc_cfg.get("plot_format"))
    if mab_format and mc_format and mab_format != mc_format:
        logger.warning(
            "MAB plot_format (%s) differs from MC plot_format (%s); using %s",
            mab_format,
            mc_format,
            mab_format,
        )
    plot_format = _resolve_plot_format(mab_format, mc_format, default="pdf")

    mab_exploration = _normalise_bool(mab_cfg.get("plot_show_exploration"))
    mc_exploration = _normalise_bool(mc_cfg.get("plot_show_exploration"))
    if mab_exploration is not None and mc_exploration is not None and mab_exploration != mc_exploration:
        logger.warning(
            "MAB plot_show_exploration (%s) differs from MC plot_show_exploration (%s); using %s",
            mab_exploration,
            mc_exploration,
            mab_exploration,
        )
    show_exploration = mab_exploration if mab_exploration is not None else (mc_exploration if mc_exploration is not None else True)

    mab_results, mab_bounds, mab_by_name = _run_mab(mab_cfg, logger, mab_base_seeds)
    mc_results, mc_bounds, mc_by_name = _run_mc(mc_cfg, logger, mc_base_seeds)

    combined_bounds = (
        min(mab_bounds[0], mc_bounds[0]),
        max(mab_bounds[1], mc_bounds[1]),
    )

    trajectory_data: Dict[str, Tuple[List[int], List[float]]] = {}
    trajectory_bands: Dict[str, Tuple[List[float], List[float]]] = {}
    exploration_data: Dict[str, List[Optional[float]]] = {}
    exploration_colors: Dict[str, str] = {}
    for method_label, grouped in (("MAB", mab_by_name), ("MC", mc_by_name)):
        for name, summaries in grouped.items():
            if not summaries:
                continue
            model_lists = [summary.get("model_selected", []) for summary in summaries]
            if not model_lists:
                continue
            min_len = min(len(items) for items in model_lists)
            if min_len <= 0:
                continue
            episode_nums = list(summaries[0].get("episode_numbers", []))[:min_len]
            stacked = np.array([items[:min_len] for items in model_lists], dtype=float)
            label = f"{method_label} - {name}"
            trajectory_data[label] = (episode_nums, np.median(stacked, axis=0).tolist())
            trajectory_bands[label] = (
                np.min(stacked, axis=0).tolist(),
                np.max(stacked, axis=0).tolist(),
            )
            if method_label == "MAB":
                exploration_colors[label] = "#FF9100"
            else:
                exploration_colors[label] = "#00E5FF"
            explored_lists = [summary.get("explored", []) for summary in summaries]
            explored_vals: List[Optional[float]] = []
            for idx in range(min_len):
                candidates = [
                    float(explored[idx])
                    for explored in explored_lists
                    if idx < len(explored) and explored[idx] is not None
                ]
                if candidates:
                    explored_vals.append(float(np.median(candidates)))
                else:
                    explored_vals.append(None)
            exploration_data[label] = explored_vals

    individual_qvalue_paths: Dict[str, str] = {}
    individual_switching_paths: Dict[str, str] = {}
    config_display: Dict[str, str] = {}
    mab_switch_points: Dict[str, Optional[int]] = {}
    mc_switch_points: Dict[str, Optional[int]] = {}

    comparison_paths = {
        "switching": os.path.join(output_dir, f"compare_switching_trajectory.{plot_format}"),
        "best_switch_points": os.path.join(output_dir, f"compare_best_switch_points.{plot_format}"),
    }

    plot_summary_switching_trajectory(
        trajectory_data,
        comparison_paths["switching"],
        bands=trajectory_bands,
        explored_series=exploration_data,
        exploration_colors=exploration_colors,
        switch_point_bounds=combined_bounds,
        show_legend=False,
        show_titles=False,
        tick_fontsize=32,
        show_exploration=show_exploration,
    )
    for idx, (display_name, summaries) in enumerate(mab_by_name.items(), start=1):
        config_slug, config_label = _normalise_config_name(display_name, "mab")
        base_config_slug = config_slug
        suffix = 1
        while config_slug in config_display and config_display[config_slug].lower() != config_label.lower():
            suffix += 1
            config_slug = f"{base_config_slug}_{suffix}"
            if not config_label.lower().endswith("(mab)"):
                config_label = f"{config_label} (MAB)"
        config_display.setdefault(config_slug, config_label)
        best_values = [
            value for value in (summary.get("best_switch_point") for summary in summaries)
            if value is not None
        ]
        mab_switch_points[config_slug] = float(np.median(best_values)) if best_values else None

    for idx, (display_name, summaries) in enumerate(mc_by_name.items(), start=1):
        config_slug, config_label = _normalise_config_name(display_name, "mc")
        base_config_slug = config_slug
        suffix = 1
        while config_slug in config_display and config_display[config_slug].lower() != config_label.lower():
            suffix += 1
            config_slug = f"{base_config_slug}_{suffix}"
            if not config_label.lower().endswith("(mc)"):
                config_label = f"{config_label} (MC)"
        config_display.setdefault(config_slug, config_label)
        best_values = [
            value for value in (summary.get("best_switch_point") for summary in summaries)
            if value is not None
        ]
        mc_switch_points[config_slug] = float(np.median(best_values)) if best_values else None

    for idx, res in enumerate(mab_results, start=1):
        display_name = res.get("name") or f"MAB Experiment {idx}"
        label = f"MAB - {display_name}"
        base_stub = f"mab_{_slugify(display_name)}"
        file_stub = base_stub
        suffix = 1
        while file_stub in individual_qvalue_paths or file_stub in individual_switching_paths:
            suffix += 1
            file_stub = f"{base_stub}_{suffix}"

        trajectory_path = os.path.join(sp_trajectory_dir, f"{file_stub}.{plot_format}")
        display_label = label.replace("_", " ")
        plot_switching_trajectory_with_exploration(
            res.get("episode_numbers", []),
            res.get("model_selected", []),
            res.get("explored"),
            trajectory_path,
            switch_point_bounds=combined_bounds,
            line_color="#E66100",
            trajectory_label="",
            exploration_color="#FF9100",
            show_exploration=show_exploration,
        )
        individual_switching_paths[file_stub] = trajectory_path

        qvalues_path = os.path.join(qvalues_dir, f"{file_stub}.{plot_format}")
        mab_states = list(res.get("q_table", {}).keys())
        mab_state_min = min(mab_states) if mab_states else None
        mab_state_max = max(mab_states) if mab_states else None
        mab_state_step = _infer_tick_step(mab_states)
        plot_multi_qvalue_vs_state(
            {label: res.get("q_table", {})},
            qvalues_path,
            best_switch_points={label: res.get("best_switch_point")},
            show_titles=False,
            show_legend=False,
            tick_fontsize=28,
            state_min=mab_state_min,
            state_max=mab_state_max,
            state_step=mab_state_step,
            x_tick_rotation=0,
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

        trajectory_path = os.path.join(sp_trajectory_dir, f"{file_stub}.{plot_format}")
        display_label = label.replace("_", " ")
        plot_switching_trajectory_with_exploration(
            res.get("episode_numbers", []),
            res.get("model_selected", []),
            res.get("explored"),
            trajectory_path,
            switch_point_bounds=combined_bounds,
            line_color="#1B4F72",
            trajectory_label="",
            exploration_color="#00E5FF",
            show_exploration=show_exploration,
        )
        individual_switching_paths[file_stub] = trajectory_path

        qvalues_path = os.path.join(qvalues_dir, f"{file_stub}.{plot_format}")
        mc_weights = [state for (state, _) in res.get("q_table", {}).keys()]
        mc_state_min = min(mc_weights) if mc_weights else None
        mc_state_max = max(mc_weights) if mc_weights else None
        mc_state_step = _infer_tick_step(mc_weights)
        plot_multi_qvalue_pair_tables(
            {label: res.get("q_table", {})},
            qvalues_path,
            best_switch_points={label: res.get("best_switch_point")},
            show_titles=False,
            show_legend=True,
            tick_fontsize=28,
            legend_fontsize=24,
            x_tick_rotation=0,
            force_last_xtick=False,
            state_min=mc_state_min,
            state_max=mc_state_max,
            state_step=mc_state_step,
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
            "explored": res.get("explored", []),
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
            "explored": res.get("explored", []),
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

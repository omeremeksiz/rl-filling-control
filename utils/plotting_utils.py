# utils/plotting_utils.py
from __future__ import annotations

from typing import Dict, Iterable, Mapping, Tuple, List, Optional, Sequence

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

FIG_SIZE_STANDARD = (14, 7)
DPI_EXPORT = 600


def plot_qvalue_vs_state_from_pair_table(
    q_table: Mapping[Tuple[int, int], float],
    out_path: Optional[str] = None,
    *,
    ax: Optional[plt.Axes] = None,
    y_limits: Optional[Tuple[float, float]] = None,
    best_switch_point: Optional[int] = None,
    show_legend: bool = True,
) -> None:
    states = sorted({state for state, _ in q_table.keys()})
    q_fast = [q_table.get((state, 1), 0.0) for state in states]
    q_slow = [q_table.get((state, -1), 0.0) for state in states]

    owns_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=FIG_SIZE_STANDARD, dpi=DPI_EXPORT)
        owns_fig = True

    if not states:
        ax.text(0.5, 0.5, "No states available", ha="center", va="center")
        ax.axis("off")
        if owns_fig:
            fig.tight_layout()
            fig.savefig(out_path, dpi=DPI_EXPORT, bbox_inches="tight")
            plt.close(fig)
        return
    if y_limits is None:
        values = q_fast + q_slow
        finite_values = [val for val in values if np.isfinite(val)]
        if finite_values:
            y_min = min(finite_values)
            y_max = max(finite_values)
            if y_min == y_max:
                scale = abs(y_max) if y_max != 0 else 1.0
                padding = max(0.1, 0.02 * scale)
                y_min = y_min - padding
                y_max = y_max + padding
            else:
                y_min = min(y_min, 0.0)
                y_max = max(y_max, 0.0)
        else:
            y_min, y_max = -1.0, 0.0
        y_limits = (min(y_min, 0.0), max(y_max, 0.0))

    y_min, y_max = y_limits
    positions = np.arange(len(states), dtype=float)
    actions = [1, -1]
    action_values = {1: q_fast, -1: q_slow}
    colors = {1: "tab:red", -1: "tab:blue"}
    labels = {1: "Action = 1", -1: "Action = -1"}
    bar_width = 0.85

    ax.set_facecolor("white")
    for action in actions:
        ax.bar(
            positions,
            action_values[action],
            width=bar_width,
            color=colors[action],
            alpha=0.8 if action == 1 else 0.45,
            label=labels[action],
            edgecolor="black",
            linewidth=0.8,
        )

    if best_switch_point is None:
        best_val = float("-inf")
        for state in states:
            for action in actions:
                value = q_table.get((state, action))
                if value is None or not np.isfinite(value):
                    continue
                if value > best_val:
                    best_val = value
                    best_switch_point = state

    if best_switch_point is not None and best_switch_point in states:
        idx = states.index(best_switch_point)
        x_coord = positions[idx]
        ax.axvline(x_coord, color="goldenrod", linestyle="--", linewidth=2.3, alpha=0.9)
        y_span = y_max - y_min if y_max > y_min else max(abs(y_max), 1.0)
        y_offset = 0.06 * y_span
        text_y = y_min + y_offset
        ax.text(
            x_coord,
            text_y,
            f"SP {best_switch_point}",
            ha="center",
            va="bottom",
            fontsize=13,
            color="goldenrod",
            fontweight="bold",
        )

    ax.axhline(0, color="black", linewidth=0.9, alpha=0.6)

    if len(states) > 30:
        step = max(1, len(states) // 15)
        tick_positions = list(positions[::step])
        tick_labels = [states[int(pos)] for pos in tick_positions]
        if tick_positions[-1] != positions[-1]:
            tick_positions.append(positions[-1])
            tick_labels.append(states[-1])
        rotation = 45
    else:
        tick_positions = list(positions)
        tick_labels = states
        rotation = 0

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=rotation, ha="right" if rotation else "center", fontsize=14, fontweight="bold")
    ax.tick_params(axis="x", labelsize=14, width=2, length=6)
    ax.tick_params(axis="y", labelsize=14, width=2, length=6)
    for tick in ax.get_yticklabels():
        tick.set_fontweight("bold")
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=6, prune="both"))

    ax.set_xlabel("Switching Point", fontsize=16, fontweight="bold")
    ax.set_ylabel("Q-Value", fontsize=16, fontweight="bold")
    ax.set_ylim(y_min, y_max)
    ax.set_xlim(positions[0] - bar_width / 2, positions[-1] + bar_width / 2)
    ax.margins(x=0, y=0)
    ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.6)
    ax.set_axisbelow(True)

    if show_legend:
        legend = ax.legend(
            loc="lower right",
            fontsize=13,
            frameon=True,
            handlelength=1.2,
            handletextpad=0.4,
            labelspacing=0.25,
            borderpad=0.3,
            borderaxespad=0.3,
        )
        if legend:
            legend.get_frame().set_alpha(0.3)
            legend.get_frame().set_edgecolor("black")
            legend.get_frame().set_linewidth(0.8)
            for text in legend.get_texts():
                text.set_fontweight("bold")
    else:
        existing = ax.get_legend()
        if existing:
            existing.remove()

    if owns_fig:
        fig.tight_layout()
        if out_path:
            fig.savefig(out_path, dpi=DPI_EXPORT, bbox_inches="tight")
        plt.close(fig)


def plot_qvalue_vs_state_bandit(q_table: Mapping[int, float], out_path: str) -> None:
    switch_points: List[int] = sorted(q_table.keys())
    q_vals: List[float] = [q_table[sp] for sp in switch_points]

    fig, ax = plt.subplots(figsize=FIG_SIZE_STANDARD, dpi=DPI_EXPORT)
    bars = ax.bar(switch_points, q_vals, alpha=0.7, color="skyblue")
    if switch_points:
        best_state = max(q_table, key=q_table.get)
        highlight_idx = switch_points.index(best_state)
        bars[highlight_idx].set_color("red")

    ax.set_title("Q-Value vs Switch Point")
    ax.set_xlabel("Switch Point")
    ax.set_ylabel("Q-Value")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI_EXPORT, bbox_inches="tight")
    plt.close(fig)


def plot_switching_trajectory_with_exploration(
    episode_nums: Iterable[int],
    model_selected: Iterable[Optional[int]],
    explored: Iterable[Optional[int]],
    out_path: str,
    switch_point_bounds: Optional[Tuple[int, int]] = None,
) -> None:
    ep_list = list(episode_nums)
    model_list = list(model_selected)
    explored_list = list(explored)

    fig, ax = plt.subplots(figsize=FIG_SIZE_STANDARD, dpi=DPI_EXPORT)
    ax.set_facecolor("white")
    ax.plot(ep_list, model_list, color="steelblue", linewidth=3.5, alpha=0.95, label="Model Selected")

    for ep, msel, ex in zip(ep_list, model_list, explored_list):
        if ex is not None and ex != msel:
            ax.plot([ep, ep], [msel, ex], color="orange", linewidth=1.2, alpha=0.75)

    if any(ex is not None for ex in explored_list):
        ax.plot([], [], color="orange", linewidth=1.5, label="Explored")

    x_min = min(ep_list) if ep_list else 0
    x_max = max(ep_list) if ep_list else 1
    ax.set_xlim(x_min, x_max)
    ax.margins(x=0)

    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=15))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True, prune="both", nbins=20))

    ax.set_ylim(44, 76)
    ax.set_yticks(np.arange(44, 77, 4))

    ax.set_xlabel("Episode", fontsize=16, fontweight="bold")
    ax.set_ylabel("Switching Point", fontsize=16, fontweight="bold")
    ax.tick_params(axis="both", labelsize=14, width=2, length=6)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")
    ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.6)
    legend = ax.legend(fontsize=14, loc="upper right")
    for text in legend.get_texts():
        text.set_fontweight("bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI_EXPORT, bbox_inches="tight")
    plt.close(fig)


def plot_multi_switching_trajectory(
    trajectories: Mapping[str, Tuple[Iterable[int], Iterable[Optional[int]]]],
    out_path: str,
    switch_point_bounds: Optional[Tuple[int, int]] = None,
) -> None:
    fig, ax = plt.subplots(figsize=FIG_SIZE_STANDARD, dpi=DPI_EXPORT)
    ax.set_facecolor("white")

    x_min, x_max = None, None
    for episodes, _ in trajectories.values():
        ep_list = list(episodes)
        if not ep_list:
            continue
        if x_min is None or min(ep_list) < x_min:
            x_min = min(ep_list)
        if x_max is None or max(ep_list) > x_max:
            x_max = max(ep_list)

    mab_palette = ["#E66100", "#F28E2B", "#FFB55A", "#C7761D"]
    mc_palette = ["#1B4F72", "#2E86C1", "#5DADE2", "#85C1E9"]
    generic_palette = plt.rcParams.get("axes.prop_cycle", plt.cycler(color=["#4E79A7", "#F28E2B", "#59A14F", "#E15759", "#B07AA1", "#FF9DA7", "#76B7B2", "#EDC948"])).by_key().get("color", ["#4E79A7", "#F28E2B", "#59A14F", "#E15759", "#B07AA1", "#FF9DA7", "#76B7B2", "#EDC948"])
    palette_iter = iter(generic_palette)
    method_counts = {"mab": 0, "mc": 0}

    plotted_labels: List[str] = []
    for label, (episodes, model_selected) in trajectories.items():
        ep_list = list(episodes)
        model_list = list(model_selected)
        if not ep_list or not model_list:
            continue
        display_label = label.replace('_', ' ')
        lower_label = label.lower()
        if "mab" in lower_label:
            idx = method_counts["mab"] % len(mab_palette)
            color = mab_palette[idx]
            method_counts["mab"] += 1
        elif "mc" in lower_label:
            idx = method_counts["mc"] % len(mc_palette)
            color = mc_palette[idx]
            method_counts["mc"] += 1
        else:
            try:
                color = next(palette_iter)
            except StopIteration:
                palette_iter = iter(generic_palette)
                color = next(palette_iter)
        ax.plot(ep_list, model_list, linewidth=3.5, alpha=0.95, label=display_label, color=color)
        plotted_labels.append(display_label)

    if x_min is None:
        x_min, x_max = 0, 1
    ax.set_xlim(x_min, x_max)
    ax.margins(x=0)

    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=15))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True, prune="both", nbins=20))

    ax.set_ylim(44, 76)
    ax.set_yticks(np.arange(44, 77, 4))

    ax.set_xlabel("Episode", fontsize=16, fontweight="bold")
    ax.set_ylabel("Switching Point", fontsize=16, fontweight="bold")
    ax.tick_params(axis="both", labelsize=14, width=2, length=6)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")
    ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.6)
    if len(plotted_labels) > 1:
        legend = ax.legend(
            fontsize=13,
            loc="upper right",
            handlelength=1.2,
            handletextpad=0.4,
            labelspacing=0.25,
            borderpad=0.2,
            borderaxespad=0.2,
            frameon=True,
        )
        legend.get_frame().set_alpha(0.3)     # 0.0 = fully transparent, 1.0 = opaque
        legend.get_frame().set_edgecolor("black")  # optional: subtle border
        legend.get_frame().set_linewidth(0.8)      # thinner frame border
        for text in legend.get_texts():
            text.set_fontweight("bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI_EXPORT, bbox_inches="tight")
    plt.close(fig)


def plot_multi_qvalue_vs_state(
    tables: Mapping[str, Mapping[int, float]],
    out_path: str,
    *,
    best_switch_points: Optional[Mapping[str, Optional[int]]] = None,
) -> None:
    if not tables:
        return

    fig, ax = plt.subplots(figsize=FIG_SIZE_STANDARD, dpi=DPI_EXPORT)
    ax.set_facecolor("white")

    all_states: List[int] = sorted({sp for table in tables.values() for sp in table.keys()})
    if not all_states:
        plt.close(fig)
        return

    num_series = max(len(tables), 1)
    total_width = min(0.9, 0.8 + 0.1 * num_series)
    bar_width = total_width / num_series
    base_positions = np.arange(len(all_states), dtype=float)

    min_offset = float("inf")
    max_offset = float("-inf")
    finite_values: List[float] = []
    best_annotations: Dict[int, float] = {}

    for idx, (label, q_table) in enumerate(tables.items()):
        offsets = base_positions + (idx - (num_series - 1) / 2) * bar_width
        if offsets.size:
            min_offset = min(min_offset, float(offsets.min()))
            max_offset = max(max_offset, float(offsets.max()))

        values = [q_table.get(state, 0.0) for state in all_states]
        finite_values.extend([val for val in values if np.isfinite(val)])

        lower_label = label.lower()
        if "mab" in lower_label:
            series_color = "tab:orange"
        elif "mc" in lower_label:
            series_color = "tab:blue"
        else:
            series_color = None

        bars = ax.bar(
            offsets,
            values,
            width=bar_width * 0.95,
            alpha=0.75,
            edgecolor="black",
            linewidth=0.6,
            color=series_color,
        )

        best_state: Optional[int] = None
        if best_switch_points:
            best_state = best_switch_points.get(label)
        if best_state is None and q_table:
            finite_items = [(state, val) for state, val in q_table.items() if np.isfinite(val)]
            if finite_items:
                best_state = max(finite_items, key=lambda item: item[1])[0]

        if best_state is not None and best_state in all_states:
            best_idx = all_states.index(best_state)
            if 0 <= best_idx < len(bars):
                bars[best_idx].set_linewidth(1.4)
                bars[best_idx].set_edgecolor("goldenrod")
            best_annotations.setdefault(best_state, base_positions[best_idx])

    if finite_values:
        y_min = min(finite_values)
        y_max = max(finite_values)
        if y_min == y_max:
            scale = abs(y_max) if y_max != 0 else 1.0
            padding = max(1.0, 0.05 * scale)
            y_min -= padding
            y_max += padding
        else:
            y_min = min(y_min, 0.0)
            y_max = max(y_max, 0.0)
    else:
        y_min, y_max = -1.0, 0.0

    y_min = min(y_min, 0.0)
    y_max = max(y_max, 0.0)

    if min_offset == float("inf") or max_offset == float("-inf"):
        min_offset = base_positions[0]
        max_offset = base_positions[-1]

    ax.set_ylim(y_min, y_max)
    ax.set_xlim(min_offset - bar_width * 0.5, max_offset + bar_width * 0.5)
    ax.margins(x=0, y=0)
    ax.axhline(0, color="black", linewidth=0.9, alpha=0.6)
    ax.set_axisbelow(True)

    ax.set_xticks(base_positions)
    rotation = 45 if len(all_states) > 20 else 0
    ax.set_xticklabels(
        all_states,
        rotation=rotation,
        ha="right" if rotation else "center",
        fontsize=14,
        fontweight="bold",
    )
    ax.tick_params(axis="x", labelsize=14, width=2, length=6)
    ax.tick_params(axis="y", labelsize=14, width=2, length=6)
    for tick in ax.get_yticklabels():
        tick.set_fontweight("bold")

    ax.set_xlabel("Switching Point", fontsize=16, fontweight="bold")
    ax.set_ylabel("Q-Value", fontsize=16, fontweight="bold")
    ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.6)

    for state, x_coord in best_annotations.items():
        ax.axvline(x_coord, color="goldenrod", linestyle="--", linewidth=2.3, alpha=0.9)
        y_span = y_max - y_min if y_max > y_min else max(abs(y_max), 1.0)
        y_offset = 0.06 * y_span
        text_y = y_min + y_offset
        ax.text(
            x_coord,
            text_y,
            f"SP {state}",
            ha="center",
            va="bottom",
            fontsize=13,
            color="goldenrod",
            fontweight="bold",
        )

    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI_EXPORT, bbox_inches="tight")
    plt.close(fig)


def plot_multi_qvalue_pair_tables(
    tables: Mapping[str, Mapping[Tuple[int, int], float]],
    out_path: str,
    *,
    best_switch_points: Optional[Mapping[str, Optional[int]]] = None,
) -> None:
    if not tables:
        return

    def infer_switch_point(q_table: Mapping[Tuple[int, int], float], ordered_states: List[int]) -> Optional[int]:
        best_state: Optional[int] = None
        best_value = float("-inf")
        for state in ordered_states:
            for action in (1, -1):
                value = q_table.get((state, action))
                if value is None or not np.isfinite(value):
                    continue
                if value > best_value:
                    best_value = value
                    best_state = state
        return best_state

    states = sorted({state for table in tables.values() for (state, _) in table.keys()})
    if not states:
        return

    items = list(tables.items())
    num_runs = len(items)
    all_values = [
        value
        for _, table in items
        for value in table.values()
        if value is not None and np.isfinite(value)
    ]
    if all_values:
        ymin = min(all_values)
        ymax = max(all_values)
        if ymin == ymax:
            scale = abs(ymax) if ymax != 0 else 1.0
            padding = max(0.1, 0.02 * scale)
            lower = ymin - padding
            upper = ymax + padding
        else:
            lower = min(ymin, 0.0)
            upper = max(ymax, 0.0)
    else:
        lower, upper = -1.0, 0.0

    lower = min(lower, 0.0)
    upper = max(upper, 0.0)
    shared_limits = (lower, upper)

    if num_runs == 1:
        fig, axes = plt.subplots(1, 1, figsize=FIG_SIZE_STANDARD, dpi=DPI_EXPORT)
        axes = np.array([[axes]])
    else:
        ncols = min(2, num_runs)
        nrows = int(np.ceil(num_runs / ncols))
        fig_width = FIG_SIZE_STANDARD[0] * ncols
        fig_height = FIG_SIZE_STANDARD[1] * nrows
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(fig_width, fig_height),
            dpi=DPI_EXPORT,
            sharey=True,
        )
        if nrows == 1:
            axes = np.array([axes])
        if ncols == 1:
            axes = axes.reshape(nrows, 1)

    axes_flat = axes.flatten()

    for idx, (label, q_table) in enumerate(items):
        ax = axes_flat[idx]
        best_sp = None
        if best_switch_points:
            best_sp = best_switch_points.get(label)
        if best_sp is None:
            best_sp = infer_switch_point(q_table, states)

        plot_qvalue_vs_state_from_pair_table(
            q_table,
            out_path=None,
            ax=ax,
            y_limits=shared_limits,
            best_switch_point=best_sp,
            show_legend=True,
        )

    for idx in range(num_runs, axes_flat.size):
        fig.delaxes(axes_flat[idx])

    for idx, ax in enumerate(axes_flat[:num_runs]):
        row = idx // axes.shape[1]
        col = idx % axes.shape[1]
        if row < axes.shape[0] - 1:
            ax.set_xlabel("")
            ax.tick_params(axis="x", labelbottom=False)
        if col > 0:
            ax.set_ylabel("")
            ax.tick_params(axis="y", labelleft=False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI_EXPORT, bbox_inches="tight")
    plt.close(fig)


def plot_compare_method_switch_points(
    method_points: Mapping[str, Mapping[str, Optional[float]]],
    config_labels: Mapping[str, str],
    out_path: str,
) -> None:
    if not method_points or not config_labels:
        return

    ordered_keys: List[str] = list(config_labels.keys())
    if not ordered_keys:
        ordered_keys = sorted({key for points in method_points.values() for key in points.keys()})
        if not ordered_keys:
            return
        config_labels = {key: key for key in ordered_keys}

    key_positions: Dict[str, float] = {key: idx + 1 for idx, key in enumerate(ordered_keys)}

    fig, ax = plt.subplots(figsize=FIG_SIZE_STANDARD, dpi=DPI_EXPORT)
    ax.set_facecolor("white")

    predefined_colors = {
        "MAB": "tab:orange",
        "MC": "tab:blue",
    }
    prop_cycle = plt.rcParams.get("axes.prop_cycle")
    if prop_cycle is not None:
        default_colors: Sequence[str] = prop_cycle.by_key().get("color", ["tab:blue", "tab:orange", "tab:green", "tab:red"])
    else:
        default_colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    color_iter = iter(default_colors)

    plotted_methods = 0
    y_values_all: List[float] = []
    annotated_series: List[Tuple[str, List[float], List[float], str]] = []

    for method, points in method_points.items():
        if not points:
            continue
        color = predefined_colors.get(method)
        if color is None:
            try:
                color = next(color_iter)
            except StopIteration:
                color_iter = iter(default_colors)
                color = next(color_iter)

        xs: List[float] = []
        ys: List[float] = []
        for key in ordered_keys:
            if key not in points:
                continue
            value = points[key]
            if value is None or not np.isfinite(value):
                continue
            xs.append(key_positions[key])
            ys.append(float(value))
            y_values_all.append(float(value))

        if not xs:
            continue

        ax.plot(
            xs,
            ys,
            marker="o",
            markersize=14,
            linewidth=3.2,
            linestyle="-",
            label=method,
            color=color,
            markerfacecolor="white",
            markeredgecolor=color,
            markeredgewidth=2.4,
        )
        annotated_series.append((method, xs, ys, color))
        plotted_methods += 1

    if not y_values_all:
        plt.close(fig)
        return

    y_min = min(y_values_all)
    y_max = max(y_values_all)
    span = y_max - y_min
    padding = max(1.0, span * 0.08) if span > 0 else max(1.0, abs(y_max) * 0.05 + 1.0)
    ax.set_ylim(y_min - padding, y_max + padding)
    offset = max(span * 0.015, 0.35) if span > 0 else 0.35

    for method, xs, ys, color in annotated_series:
        for x, y in zip(xs, ys):
            ax.text(
                x,
                y + offset,
                f"{int(round(y))}",
                ha="center",
                va="bottom",
                fontsize=12,
                color=color,
                fontweight="bold",
            )

    x_positions = [key_positions[key] for key in ordered_keys]
    display_labels = [config_labels[key].replace("_", " ").replace("-", " ").strip() or key for key in ordered_keys]
    rotation = 45 if len(display_labels) > 6 else 0

    ax.set_xticks(x_positions)
    ax.set_xticklabels(display_labels, rotation=rotation, ha="right" if rotation else "center", fontsize=14, fontweight="bold")
    ax.tick_params(axis="x", labelsize=14, width=2, length=6)
    ax.tick_params(axis="y", labelsize=14, width=2, length=6)
    for tick in ax.get_yticklabels():
        tick.set_fontweight("bold")

    ax.set_xlabel("Hyperparameter Configuration", fontsize=16, fontweight="bold")
    ax.set_ylabel("Best Switching Point", fontsize=16, fontweight="bold")
    ax.set_xlim(min(x_positions) - 0.5, max(x_positions) + 0.5)
    ax.margins(x=0.02)
    ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.6)
    ax.set_axisbelow(True)

    if plotted_methods > 1:
        legend = ax.legend(
            fontsize=13,
            loc="upper left",
            frameon=True,
            handlelength=1.4,
            handletextpad=0.5,
            borderpad=0.3,
            borderaxespad=0.3,
        )
        legend.get_frame().set_alpha(0.3)
        legend.get_frame().set_edgecolor("black")
        legend.get_frame().set_linewidth(0.8)
        for text in legend.get_texts():
            text.set_fontweight("bold")

    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI_EXPORT, bbox_inches="tight")
    plt.close(fig)

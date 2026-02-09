# utils/plotting_utils.py
from __future__ import annotations

from typing import Dict, Iterable, Mapping, Tuple, List, Optional, Sequence

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
from matplotlib.figure import Figure
import numpy as np

FIG_SIZE_STANDARD = (14, 7)
DPI_EXPORT = 400
DPI_PNG_EXPORT = 300
FONT_LABEL = 20
FONT_TICK = 16
FONT_LEGEND = 15
FONT_ANNOT = 22
FONT_TITLE = 20
TRAJECTORY_LINEWIDTH = 4.5


def _save_figure(fig: Figure, out_path: Optional[str], *, dpi: int = DPI_EXPORT) -> None:
    if not out_path:
        return
    dpi_to_use = dpi
    if out_path.lower().endswith(".png"):
        dpi_to_use = DPI_PNG_EXPORT
    fig.savefig(out_path, dpi=dpi_to_use, bbox_inches="tight")






def plot_qvalue_vs_state_from_pair_table(
    q_table: Mapping[Tuple[int, int], float],
    out_path: Optional[str] = None,
    *,
    ax: Optional[plt.Axes] = None,
    y_limits: Optional[Tuple[float, float]] = None,
    best_switch_point: Optional[int] = None,
    show_legend: bool = True,
    show_titles: bool = True,
    tick_fontsize: Optional[int] = None,
    state_min: Optional[int] = None,
    state_max: Optional[int] = None,
    state_step: Optional[int] = None,
    x_tick_rotation: Optional[int] = None,
    force_last_xtick: bool = True,
    legend_fontsize: Optional[int] = None,
) -> None:
    states = sorted({state for state, _ in q_table.keys()})
    if state_min is not None or state_max is not None:
        min_bound = state_min if state_min is not None else (min(states) if states else 0)
        max_bound = state_max if state_max is not None else (max(states) if states else 0)
        states = [state for state in states if min_bound <= state <= max_bound]
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
            _save_figure(fig, out_path)
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

    sp_color = "#0B6E4F"
    sp_linewidth = 4.2

    if best_switch_point is not None and best_switch_point in states:
        idx = states.index(best_switch_point)
        x_coord = positions[idx]
        ax.axvline(x_coord, color=sp_color, linestyle="--", linewidth=sp_linewidth, alpha=0.95)
        y_span = y_max - y_min if y_max > y_min else max(abs(y_max), 1.0)
        y_offset = 0.06 * y_span
        text_y = y_min + y_offset
        ax.text(
            x_coord - 0.35,
            text_y,
            f"SP {best_switch_point}",
            ha="right",
            va="bottom",
            fontsize=FONT_ANNOT,
            color=sp_color,
            fontweight="bold",
        )

    ax.axhline(0, color="black", linewidth=0.9, alpha=0.6)

    if state_min is not None and state_max is not None and state_step:
        step = max(1, int(state_step))
        tick_values = list(range(state_min, state_max + 1, step))
        tick_positions = [positions[states.index(val)] for val in tick_values if val in states]
        tick_labels = [val for val in tick_values if val in states]
        rotation = x_tick_rotation if x_tick_rotation is not None else 0
    elif len(states) > 30:
        step = max(1, len(states) // 15)
        tick_positions = list(positions[::step])
        tick_labels = [states[int(pos)] for pos in tick_positions]
        if force_last_xtick and tick_positions[-1] != positions[-1]:
            tick_positions.append(positions[-1])
            tick_labels.append(states[-1])
        rotation = x_tick_rotation if x_tick_rotation is not None else 45
    else:
        tick_positions = list(positions)
        tick_labels = states
        rotation = x_tick_rotation if x_tick_rotation is not None else 0

    tick_size = tick_fontsize or FONT_TICK
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=rotation, ha="right" if rotation else "center", fontsize=tick_size, fontweight="bold")
    ax.tick_params(axis="x", labelsize=tick_size, width=2, length=6)
    ax.tick_params(axis="y", labelsize=tick_size, width=2, length=6)
    for tick in ax.get_yticklabels():
        tick.set_fontweight("bold")
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=6, prune="both"))

    if show_titles:
        ax.set_xlabel("Switching Point", fontsize=FONT_LABEL, fontweight="bold")
        ax.set_ylabel("Q-Value", fontsize=FONT_LABEL, fontweight="bold")
    ax.set_ylim(y_min, y_max)
    ax.set_xlim(positions[0] - bar_width / 2, positions[-1] + bar_width / 2)
    ax.margins(x=0, y=0)
    ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.6)
    ax.set_axisbelow(True)

    if show_legend:
        legend = ax.legend(
            loc="lower right",
            fontsize=legend_fontsize or FONT_LEGEND,
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
        _save_figure(fig, out_path)
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

    ax.set_title("Q-Value vs Switch Point", fontsize=FONT_TITLE, fontweight="bold")
    ax.set_xlabel("Switch Point", fontsize=FONT_LABEL, fontweight="bold")
    ax.set_ylabel("Q-Value", fontsize=FONT_LABEL, fontweight="bold")
    ax.tick_params(axis="both", labelsize=FONT_TICK, width=2, length=6)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save_figure(fig, out_path)
    plt.close(fig)


def plot_switching_trajectory_with_exploration(
    episode_nums: Iterable[int],
    model_selected: Iterable[Optional[int]],
    explored: Optional[Iterable[Optional[int]]] = None,
    out_path: str = "",
    switch_point_bounds: Optional[Tuple[int, int]] = None,
    *,
    line_color: str = "#1B4F72",
    trajectory_label: str = "Model Selected",
    exploration_color: str = "#39FF14",
    show_exploration: bool = True,
) -> None:
    ep_list = list(episode_nums)
    model_list = list(model_selected)
    explored_list = list(explored) if explored is not None else [None] * len(model_list)

    fig, ax = plt.subplots(figsize=FIG_SIZE_STANDARD, dpi=DPI_EXPORT)
    ax.set_facecolor("white")
    ax.plot(
        ep_list,
        model_list,
        color=line_color,
        linewidth=TRAJECTORY_LINEWIDTH,
        alpha=0.95,
        label=trajectory_label,
        zorder=3,
    )

    if show_exploration:
        for ep, msel, ex in zip(ep_list, model_list, explored_list):
            if ex is not None and ex != msel:
                ax.plot([ep, ep], [msel, ex], color=exploration_color, linewidth=2.4, alpha=0.9, zorder=2)

    x_min = min(ep_list) if ep_list else 0
    x_max = max(ep_list) if ep_list else 1
    ax.set_xlim(x_min, x_max)
    ax.margins(x=0)

    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=15))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True, prune="both", nbins=20))

    ax.set_ylim(44, 76)
    ax.set_yticks(np.arange(44, 77, 2))

    ax.set_xlabel("Episode", fontsize=FONT_LABEL, fontweight="bold")
    ax.set_ylabel("Switching Point", fontsize=FONT_LABEL, fontweight="bold")
    ax.tick_params(axis="both", labelsize=FONT_TICK, width=2, length=6)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")
    ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.6)

    # Legends are omitted for single trajectories per request

    fig.tight_layout()
    if out_path:
        _save_figure(fig, out_path)
    plt.close(fig)


def plot_multi_switching_trajectory(
    trajectories: Mapping[str, Tuple[Iterable[int], Iterable[Optional[int]]]],
    out_path: str,
    switch_point_bounds: Optional[Tuple[int, int]] = None,
    *,
    show_legend: bool = True,
    show_titles: bool = True,
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
        ax.plot(ep_list, model_list, linewidth=TRAJECTORY_LINEWIDTH, alpha=0.95, label=display_label, color=color)
        plotted_labels.append(display_label)

    if x_min is None:
        x_min, x_max = 0, 1
    ax.set_xlim(x_min, x_max)
    ax.margins(x=0)

    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=15))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True, prune="both", nbins=20))

    ax.set_ylim(44, 76)
    ax.set_yticks(np.arange(44, 77, 4))

    if show_titles:
        ax.set_xlabel("Episode", fontsize=FONT_LABEL, fontweight="bold")
        ax.set_ylabel("Switching Point", fontsize=FONT_LABEL, fontweight="bold")
    ax.tick_params(axis="both", labelsize=FONT_TICK, width=2, length=6)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")
    ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.6)
    if show_legend and len(plotted_labels) > 1:
        legend = ax.legend(
            fontsize=FONT_LEGEND,
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
    _save_figure(fig, out_path)
    plt.close(fig)


def plot_multi_qvalue_vs_state(
    tables: Mapping[str, Mapping[int, float]],
    out_path: str,
    *,
    best_switch_points: Optional[Mapping[str, Optional[int]]] = None,
    show_titles: bool = True,
    show_legend: bool = False,
    tick_fontsize: Optional[int] = None,
    state_min: Optional[int] = None,
    state_max: Optional[int] = None,
    state_step: Optional[int] = None,
    x_tick_rotation: Optional[int] = None,
) -> None:
    if not tables:
        return

    fig, ax = plt.subplots(figsize=FIG_SIZE_STANDARD, dpi=DPI_EXPORT)
    ax.set_facecolor("white")

    all_states = sorted({sp for table in tables.values() for sp in table.keys()})
    if state_min is not None or state_max is not None:
        min_bound = state_min if state_min is not None else (min(all_states) if all_states else 0)
        max_bound = state_max if state_max is not None else (max(all_states) if all_states else 0)
        all_states = [state for state in all_states if min_bound <= state <= max_bound]
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
                bars[best_idx].set_edgecolor("#0B6E4F")
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

    tick_size = tick_fontsize or FONT_TICK
    if state_min is not None and state_max is not None and state_step:
        tick_values = list(range(state_min, state_max + 1, state_step))
        tick_positions = [base_positions[all_states.index(val)] for val in tick_values if val in all_states]
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(
            [val for val in tick_values if val in all_states],
            rotation=x_tick_rotation if x_tick_rotation is not None else 0,
            ha="right" if (x_tick_rotation or 0) else "center",
            fontsize=tick_size,
            fontweight="bold",
        )
    else:
        ax.set_xticks(base_positions)
        rotation = x_tick_rotation if x_tick_rotation is not None else (45 if len(all_states) > 20 else 0)
        ax.set_xticklabels(
            all_states,
            rotation=rotation,
            ha="right" if rotation else "center",
            fontsize=tick_size,
            fontweight="bold",
        )
    ax.tick_params(axis="x", labelsize=tick_size, width=2, length=6)
    ax.tick_params(axis="y", labelsize=tick_size, width=2, length=6)
    for tick in ax.get_yticklabels():
        tick.set_fontweight("bold")

    if show_titles:
        ax.set_xlabel("Switching Point", fontsize=FONT_LABEL, fontweight="bold")
        ax.set_ylabel("Q-Value", fontsize=FONT_LABEL, fontweight="bold")
    ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.6)

    sp_color = "#0B6E4F"
    sp_linewidth = 4.2
    for state, x_coord in best_annotations.items():
        ax.axvline(x_coord, color=sp_color, linestyle="--", linewidth=sp_linewidth, alpha=0.95)
        y_span = y_max - y_min if y_max > y_min else max(abs(y_max), 1.0)
        y_offset = 0.06 * y_span
        text_y = y_min + y_offset
        ax.text(
            x_coord - (bar_width * 0.35),
            text_y,
            f"SP {state}",
            ha="right",
            va="bottom",
            fontsize=FONT_ANNOT,
            color=sp_color,
            fontweight="bold",
        )

    if show_legend:
        ax.legend(fontsize=FONT_LEGEND)

    fig.tight_layout()
    _save_figure(fig, out_path)
    plt.close(fig)


def plot_multi_qvalue_pair_tables(
    tables: Mapping[str, Mapping[Tuple[int, int], float]],
    out_path: str,
    *,
    best_switch_points: Optional[Mapping[str, Optional[int]]] = None,
    show_titles: bool = True,
    show_legend: bool = True,
    tick_fontsize: Optional[int] = None,
    state_min: Optional[int] = None,
    state_max: Optional[int] = None,
    state_step: Optional[int] = None,
    x_tick_rotation: Optional[int] = None,
    force_last_xtick: bool = True,
    legend_fontsize: Optional[int] = None,
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
    if state_min is not None or state_max is not None:
        min_bound = state_min if state_min is not None else (min(states) if states else 0)
        max_bound = state_max if state_max is not None else (max(states) if states else 0)
        states = [state for state in states if min_bound <= state <= max_bound]
    if not states:
        return
    allowed_states = set(states)

    items = list(tables.items())
    num_runs = len(items)
    all_values = [
        value
        for _, table in items
        for (state, _), value in table.items()
        if state in allowed_states and value is not None and np.isfinite(value)
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
            show_legend=show_legend,
            show_titles=show_titles,
            tick_fontsize=tick_fontsize,
            state_min=state_min,
            state_max=state_max,
            state_step=state_step,
            x_tick_rotation=x_tick_rotation,
            force_last_xtick=force_last_xtick,
            legend_fontsize=legend_fontsize,
        )

    for idx in range(num_runs, axes_flat.size):
        fig.delaxes(axes_flat[idx])

    for idx, ax in enumerate(axes_flat[:num_runs]):
        row = idx // axes.shape[1]
        col = idx % axes.shape[1]
        if show_titles and row < axes.shape[0] - 1:
            ax.set_xlabel("")
            ax.tick_params(axis="x", labelbottom=False)
        if show_titles and col > 0:
            ax.set_ylabel("")
            ax.tick_params(axis="y", labelleft=False)

    fig.tight_layout()
    _save_figure(fig, out_path)
    plt.close(fig)


def plot_summary_switching_trajectory(
    trajectories: Mapping[str, Tuple[Iterable[int], Iterable[float]]],
    out_path: str,
    *,
    bands: Optional[Mapping[str, Tuple[Iterable[float], Iterable[float]]]] = None,
    explored_series: Optional[Mapping[str, Iterable[Optional[float]]]] = None,
    exploration_colors: Optional[Mapping[str, str]] = None,
    switch_point_bounds: Optional[Tuple[int, int]] = None,
    show_legend: bool = True,
    show_titles: bool = True,
    tick_fontsize: Optional[int] = None,
    show_exploration: bool = True,
    exploration_color: str = "#39FF14",
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
    for label, (episodes, medians) in trajectories.items():
        ep_list = list(episodes)
        median_list = list(medians)
        if not ep_list or not median_list:
            continue
        display_label = label.replace("_", " ")
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

        plot_len = min(len(ep_list), len(median_list))
        if plot_len <= 0:
            continue
        ep_list = ep_list[:plot_len]
        median_list = median_list[:plot_len]

        ax.plot(ep_list, median_list, linewidth=TRAJECTORY_LINEWIDTH, alpha=0.95, label=display_label, color=color, zorder=3)
        plotted_labels.append(display_label)

        if show_exploration and explored_series and label in explored_series:
            explored_list = list(explored_series[label])
            exp_len = min(len(ep_list), len(explored_list), len(median_list))
            if exp_len > 0:
                color_override = exploration_colors.get(label) if exploration_colors else None
                plot_color = color_override or exploration_color
                for ep, msel, ex in zip(ep_list[:exp_len], median_list[:exp_len], explored_list[:exp_len]):
                    if ex is None:
                        continue
                    if np.isclose(float(ex), float(msel)):
                        continue
                    ax.plot([ep, ep], [msel, ex], color=plot_color, linewidth=2.4, alpha=0.9, zorder=2)

        if bands and label in bands:
            lower, upper = bands[label]
            lower_list = list(lower)
            upper_list = list(upper)
            band_len = min(len(ep_list), len(lower_list), len(upper_list))
            if band_len > 0:
                ax.fill_between(
                    ep_list[:band_len],
                    lower_list[:band_len],
                    upper_list[:band_len],
                    color=color,
                    alpha=0.18,
                    linewidth=0,
                    zorder=1,
                )

    if x_min is None:
        x_min, x_max = 0, 1
    ax.set_xlim(x_min, x_max)
    ax.margins(x=0)

    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=15))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True, prune="both", nbins=20))

    ax.set_ylim(44, 76)
    ax.set_yticks(np.arange(44, 77, 4))

    if show_titles:
        ax.set_xlabel("Episode", fontsize=FONT_LABEL, fontweight="bold")
        ax.set_ylabel("Switching Point", fontsize=FONT_LABEL, fontweight="bold")
    tick_size = tick_fontsize or FONT_TICK
    ax.tick_params(axis="both", labelsize=tick_size, width=2, length=6)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")
    ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.6)
    if show_legend and len(plotted_labels) > 1:
        legend = ax.legend(
            fontsize=FONT_LEGEND,
            loc="upper right",
            handlelength=1.2,
            handletextpad=0.4,
            labelspacing=0.25,
            borderpad=0.2,
            borderaxespad=0.2,
            frameon=True,
        )
        legend.get_frame().set_alpha(0.3)
        legend.get_frame().set_edgecolor("black")
        legend.get_frame().set_linewidth(0.8)
        for text in legend.get_texts():
            text.set_fontweight("bold")
    fig.tight_layout()
    _save_figure(fig, out_path)
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

    y_min = 44
    y_max = 76
    span = y_max - y_min
    ax.set_ylim(y_min, y_max)
    ax.set_yticks(np.arange(44, 77, 4))
    offset = max(span * 0.015, 0.35)

    for method, xs, ys, color in annotated_series:
        for x, y in zip(xs, ys):
            ax.text(
                x,
                y + offset,
                f"{int(round(y))}",
                ha="center",
                va="bottom",
                fontsize=FONT_ANNOT,
                color=color,
                fontweight="bold",
            )

    x_positions = [key_positions[key] for key in ordered_keys]
    display_labels = [config_labels[key].replace("_", " ").replace("-", " ").strip() or key for key in ordered_keys]
    rotation = 45 if len(display_labels) > 6 else 0

    ax.set_xticks(x_positions)
    ax.set_xticklabels(display_labels, rotation=rotation, ha="right" if rotation else "center", fontsize=FONT_TICK, fontweight="bold")
    ax.tick_params(axis="x", labelsize=FONT_TICK, width=2, length=6)
    ax.tick_params(axis="y", labelsize=FONT_TICK, width=2, length=6)
    for tick in ax.get_yticklabels():
        tick.set_fontweight("bold")

    ax.set_xlabel("Hyperparameter Configuration", fontsize=FONT_LABEL, fontweight="bold")
    ax.set_ylabel("Best Switching Point", fontsize=FONT_LABEL, fontweight="bold")
    ax.set_xlim(min(x_positions) - 0.5, max(x_positions) + 0.5)
    ax.margins(x=0.02)
    ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.6)
    ax.set_axisbelow(True)

    if plotted_methods > 1:
        legend = ax.legend(
            fontsize=FONT_LEGEND,
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
    _save_figure(fig, out_path)
    plt.close(fig)


def plot_penalty_sweep_best_switch_points(
    method_points: Mapping[str, Mapping[int, Sequence[float]]],
    out_path: str,
    *,
    penalty_order: Optional[Sequence[int]] = None,
    x_tick_step: int = 10,
    x_tick_schedule: Optional[Sequence[Mapping[str, int]]] = None,
    mark_changes_only: bool = True,
    show_legend: bool = True,
) -> None:
    if not method_points:
        return

    if penalty_order is None:
        penalty_order = sorted({penalty for points in method_points.values() for penalty in points.keys()})
    if not penalty_order:
        return

    fig, ax = plt.subplots(figsize=FIG_SIZE_STANDARD, dpi=DPI_EXPORT)
    ax.set_facecolor("white")

    colors = {
        "MAB": "tab:orange",
        "MC": "tab:blue",
    }
    prop_cycle = plt.rcParams.get("axes.prop_cycle")
    if prop_cycle is not None:
        default_colors: Sequence[str] = prop_cycle.by_key().get("color", ["tab:blue", "tab:orange", "tab:green", "tab:red"])
    else:
        default_colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    color_iter = iter(default_colors)

    for method, points in method_points.items():
        if not points:
            continue
        color = colors.get(method)
        if color is None:
            try:
                color = next(color_iter)
            except StopIteration:
                color_iter = iter(default_colors)
                color = next(color_iter)

        xs: List[int] = []
        medians: List[float] = []
        mins: List[float] = []
        maxs: List[float] = []
        for penalty in penalty_order:
            values = points.get(penalty, [])
            finite_vals = [float(val) for val in values if val is not None and np.isfinite(val)]
            xs.append(int(penalty))
            if finite_vals:
                medians.append(float(np.median(finite_vals)))
                mins.append(float(min(finite_vals)))
                maxs.append(float(max(finite_vals)))
            else:
                medians.append(float("nan"))
                mins.append(float("nan"))
                maxs.append(float("nan"))

        xs_arr = np.array(xs, dtype=float)
        med_arr = np.array(medians, dtype=float)
        min_arr = np.array(mins, dtype=float)
        max_arr = np.array(maxs, dtype=float)
        mask = np.isfinite(med_arr)
        if not mask.any():
            continue

        ax.plot(
            xs_arr[mask],
            med_arr[mask],
            linewidth=TRAJECTORY_LINEWIDTH,
            label=method,
            color=color,
        )
        if mark_changes_only:
            change_indices: List[int] = []
            prev_val: Optional[float] = None
            for idx in np.where(mask)[0]:
                val = med_arr[idx]
                if prev_val is None or not np.isclose(val, prev_val):
                    change_indices.append(int(idx))
                prev_val = val
            if change_indices:
                ax.plot(
                    xs_arr[change_indices],
                    med_arr[change_indices],
                    linestyle="None",
                    marker="o",
                    markersize=18,
                    color=color,
                    markerfacecolor="white",
                    markeredgecolor=color,
                    markeredgewidth=2.6,
                )
        else:
            ax.plot(
                xs_arr[mask],
                med_arr[mask],
                linestyle="None",
                marker="o",
                markersize=18,
                color=color,
                markerfacecolor="white",
                markeredgecolor=color,
                markeredgewidth=2.6,
            )
        ax.fill_between(
            xs_arr[mask],
            min_arr[mask],
            max_arr[mask],
            color=color,
            alpha=0.18,
            linewidth=0,
            zorder=1,
        )

    start = penalty_order[0]
    end = penalty_order[-1]
    ax.set_xlim(start, end)

    if x_tick_schedule:
        ticks: List[int] = []
        for seg in x_tick_schedule:
            seg_start = int(seg.get("start", start))
            seg_end = int(seg.get("end", end))
            seg_step = int(seg.get("step", x_tick_step))
            if seg_step == 0:
                continue
            if seg_start <= seg_end and seg_step < 0:
                seg_step = -seg_step
            if seg_start >= seg_end and seg_step > 0:
                seg_step = -seg_step
            current = seg_start
            if seg_step > 0:
                while current <= seg_end:
                    ticks.append(current)
                    current += seg_step
            else:
                while current >= seg_end:
                    ticks.append(current)
                    current += seg_step
        ticks.append(start)
        ticks.append(end)
        ticks = sorted(set(ticks), reverse=(start > end))
        ax.set_xticks(ticks)
    elif x_tick_step > 0:
        step = x_tick_step if start <= end else -abs(x_tick_step)
        ticks = list(range(start, end + step, step))
        if ticks and ticks[0] != start:
            ticks.insert(0, start)
        if ticks and ticks[-1] != end:
            ticks.append(end)
        ax.set_xticks(ticks)
    else:
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=10))

    ax.set_ylim(44, 76)
    ax.set_yticks(np.arange(44, 77, 4))

    label_size = max(FONT_LABEL, 26)
    ax.set_xlabel("Penalty Constant", fontsize=label_size, fontweight="bold")
    ax.set_ylabel("Best Switching Point", fontsize=label_size, fontweight="bold")
    ax.tick_params(axis="both", labelsize=FONT_TICK, width=2, length=6)
    for tick in ax.get_yticklabels() + ax.get_xticklabels():
        tick.set_fontweight("bold")
    ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.6)
    ax.set_axisbelow(True)

    if show_legend:
        legend = ax.legend(
            fontsize=max(FONT_LEGEND, 24),
            loc="upper right",
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
    _save_figure(fig, out_path)
    plt.close(fig)

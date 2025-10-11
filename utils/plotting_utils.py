# utils/plotting_utils.py
from __future__ import annotations

from typing import Dict, Iterable, Mapping, Tuple, List, Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


def plot_qvalue_vs_state_from_pair_table(
    q_table: Mapping[Tuple[int, int], float],
    out_path: Optional[str] = None,
    *,
    ax: Optional[plt.Axes] = None,
    y_limits: Optional[Tuple[float, float]] = None,
    best_switch_point: Optional[int] = None,
) -> None:
    states = sorted({state for state, _ in q_table.keys()})
    q_fast = [q_table.get((state, 1), 0.0) for state in states]
    q_slow = [q_table.get((state, -1), 0.0) for state in states]

    owns_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 6), dpi=300)
        owns_fig = True

    if not states:
        ax.text(0.5, 0.5, "No states available", ha="center", va="center")
        ax.axis("off")
        if owns_fig:
            fig.tight_layout()
            fig.savefig(out_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
        return

    if y_limits is None:
        values = q_fast + q_slow
        finite_values = [val for val in values if np.isfinite(val)]
        if finite_values:
            y_min = min(finite_values)
            y_max = max(finite_values)
            padding = max(1.0, 0.05 * (y_max - y_min))
            y_limits = (y_min - padding, y_max + padding)
        else:
            y_limits = (-1.0, 1.0)

    positions = np.arange(len(states), dtype=float)

    ax.bar(
        positions,
        q_fast,
        width=0.9,
        color="tab:red",
        alpha=0.8,
        label="Action = 1",
        align="center",
        edgecolor="black",
        linewidth=0.5,
    )
    ax.bar(
        positions,
        q_slow,
        width=0.9,
        color="tab:blue",
        alpha=0.55,
        label="Action = -1",
        align="center",
        edgecolor="black",
        linewidth=0.5,
    )

    if best_switch_point is not None and best_switch_point in states:
        idx = states.index(best_switch_point)
        ax.axvline(idx, color="goldenrod", linestyle="--", linewidth=1.2, alpha=0.85)
        y_span = y_limits[1] - y_limits[0]
        y_offset = 0.06 * y_span
        ax.text(
            idx,
            y_limits[1] - y_offset,
            f"SP {best_switch_point}",
            ha="center",
            va="top",
            fontsize=9,
            color="goldenrod",
            fontweight="bold",
        )

    ax.axhline(0, color="black", linewidth=0.8, alpha=0.5)
    if len(states) > 30:
        step = max(1, len(states) // 15)
        tick_positions = list(positions[::step])
        tick_labels = [states[int(pos)] for pos in tick_positions]
        if tick_positions[-1] != positions[-1]:
            tick_positions.append(positions[-1])
            tick_labels.append(states[-1])
    else:
        tick_positions = list(positions)
        tick_labels = states
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=8)
    ax.tick_params(axis="y", labelsize=8)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=6, prune="both"))
    ax.set_xlabel("State (Weight)", fontsize=10)
    ax.set_ylabel("Q-Value", fontsize=10)
    ax.set_ylim(y_limits)
    ax.grid(axis="y", alpha=0.25)
    ax.set_xlim(-0.6, len(states) - 0.4)

    if owns_fig:
        ax.set_title("Q-Values by State and Action")
        legend_handles = [
            plt.Line2D([0], [0], color="tab:red", linewidth=10, alpha=0.8, label="Action = 1"),
            plt.Line2D([0], [0], color="tab:blue", linewidth=10, alpha=0.55, label="Action = -1"),
        ]
        if best_switch_point is not None and best_switch_point in states:
            legend_handles.append(
                plt.Line2D(
                    [0],
                    [0],
                    color="goldenrod",
                    linestyle="--",
                    linewidth=2,
                    label="Switch Point",
                )
            )
        ax.legend(handles=legend_handles, fontsize=9, loc="lower left")
        fig.tight_layout()
        if out_path:
            fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)


def plot_qvalue_vs_state_bandit(q_table: Mapping[int, float], out_path: str) -> None:
    switch_points: List[int] = sorted(q_table.keys())
    q_vals: List[float] = [q_table[sp] for sp in switch_points]

    plt.figure(figsize=(12, 6))
    bars = plt.bar(switch_points, q_vals, alpha=0.7, color="skyblue")
    if switch_points:
        best_state = max(q_table, key=q_table.get)
        highlight_idx = switch_points.index(best_state)
        bars[highlight_idx].set_color("red")

    plt.title("Q-Value vs Switch Point")
    plt.xlabel("Switch Point")
    plt.ylabel("Q-Value")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


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

    plt.style.use("seaborn-v0_8")
    plt.figure(figsize=(14, 7), dpi=300)
    plt.plot(ep_list, model_list, color="blue", linewidth=3.5, alpha=0.9, label="Model Selected")

    for ep, msel, ex in zip(ep_list, model_list, explored_list):
        if ex is not None and ex != msel:
            plt.plot([ep, ep], [msel, ex], color="orange", linewidth=1.0, alpha=0.6)

    if any(ex is not None for ex in explored_list):
        plt.plot([], [], color="orange", linewidth=1.2, label="Explored")

    ax = plt.gca()
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=15))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True, prune="both", nbins=20))

    if switch_point_bounds is not None:
        y_min, y_max = switch_point_bounds
    else:
        observed: List[int] = [
            sp for sp in model_list if sp is not None
        ] + [sp for sp in explored_list if sp is not None]
        if observed:
            y_min = min(observed)
            y_max = max(observed)
        else:
            y_min, y_max = 0, 1

    if y_min == y_max:
        y_min -= 1
        y_max += 1

    ax.set_ylim(y_min, y_max)

    plt.xlabel("Episode", fontsize=14)
    plt.ylabel("Switching Point", fontsize=14)
    plt.title("Switching Point Trajectory", fontsize=16, weight="bold")
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    plt.legend(fontsize=10, loc="upper right")
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def plot_multi_switching_trajectory(
    trajectories: Mapping[str, Tuple[Iterable[int], Iterable[Optional[int]]]],
    out_path: str,
    switch_point_bounds: Optional[Tuple[int, int]] = None,
) -> None:
    plt.style.use("seaborn-v0_8")
    plt.figure(figsize=(14, 7), dpi=300)
    ax = plt.gca()

    for label, (episodes, model_selected) in trajectories.items():
        ep_list = list(episodes)
        model_list = list(model_selected)
        if not ep_list or not model_list:
            continue
        plt.plot(ep_list, model_list, linewidth=3.5, alpha=0.9, label=label)

    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=15))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True, prune="both", nbins=20))

    if switch_point_bounds is not None:
        y_min, y_max = switch_point_bounds
    else:
        all_switch_points: List[int] = []
        for _, model_selected in trajectories.items():
            all_switch_points.extend(sp for sp in model_selected if sp is not None)
        if all_switch_points:
            y_min = min(all_switch_points)
            y_max = max(all_switch_points)
        else:
            y_min, y_max = 0, 1

    if y_min == y_max:
        y_min -= 1
        y_max += 1

    ax.set_ylim(y_min, y_max)

    plt.xlabel("Episode", fontsize=14)
    plt.ylabel("Switching Point", fontsize=14)
    plt.title("Switching Point Trajectory", fontsize=16, weight="bold")
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    plt.legend(fontsize=10, loc="upper right")
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def plot_multi_qvalue_vs_state(
    tables: Mapping[str, Mapping[int, float]],
    out_path: str,
) -> None:
    if not tables:
        return

    plt.figure(figsize=(12, 6))
    all_states: List[int] = sorted({sp for table in tables.values() for sp in table.keys()})
    if not all_states:
        plt.close()
        return

    num_series = max(len(tables), 1)
    total_width = min(0.9, 0.8 + 0.1 * num_series)
    bar_width = total_width / num_series
    base_positions = np.arange(len(all_states))

    for idx, (label, q_table) in enumerate(tables.items()):
        offsets = base_positions + (idx - (num_series - 1) / 2) * bar_width
        values = [q_table.get(state, 0.0) for state in all_states]
        bars = plt.bar(
            offsets,
            values,
            width=bar_width * 0.95,
            alpha=0.7,
            label=label,
        )
        if values:
            best_state = all_states[int(np.argmax(values))]
            best_value = max(values)
            for bar_state, bar in zip(all_states, bars):
                if bar_state == best_state and best_value == bar.get_height():
                    bar.set_edgecolor("black")
                    bar.set_linewidth(1.0)

    plt.title("Q-Value vs Switch Point", fontsize=16, weight="bold")
    plt.xlabel("Switch Point", fontsize=14)
    plt.ylabel("Q-Value", fontsize=14)
    plt.xticks(base_positions, all_states)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10, loc="best")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_multi_qvalue_pair_tables(
    tables: Mapping[str, Mapping[Tuple[int, int], float]],
    out_path: str,
) -> None:
    if not tables:
        return

    def infer_switch_point(q_table: Mapping[Tuple[int, int], float], ordered_states: List[int]) -> Optional[int]:
        for state in ordered_states:
            fast = q_table.get((state, 1))
            slow = q_table.get((state, -1))
            if fast is None or slow is None:
                continue
            if slow > fast:
                return state
        # fallback: best overall action=-1 value
        slow_values = [(state, q_table.get((state, -1), float("-inf"))) for state in ordered_states]
        slow_values = [item for item in slow_values if np.isfinite(item[1])]
        if slow_values:
            return max(slow_values, key=lambda item: item[1])[0]
        return None

    actions = [1, -1]
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
        padding = max(1.0, 0.05 * (ymax - ymin))
        shared_limits = (ymin - padding, ymax + padding)
    else:
        shared_limits = (-1.0, 1.0)

    if num_runs == 1:
        fig, axes = plt.subplots(1, 1, figsize=(16, 6), dpi=300)
        axes = np.array([[axes]])
    else:
        ncols = min(2, num_runs)
        nrows = int(np.ceil(num_runs / ncols))
        fig_width = max(16.0, ncols * 8.0)
        fig_height = max(6.0, nrows * 4.0)
        fig, axes = plt.subplots(nrows, ncols, figsize=(fig_width, fig_height), dpi=300, sharey=True)
        if nrows == 1:
            axes = np.array([axes])
        if ncols == 1:
            axes = axes.reshape(nrows, 1)

    axes_flat = axes.flatten()

    for idx, (label, q_table) in enumerate(items):
        ax = axes_flat[idx]
        plot_qvalue_vs_state_from_pair_table(
            q_table,
            out_path=None,
            ax=ax,
            y_limits=shared_limits,
            best_switch_point=infer_switch_point(q_table, states),
        )
        ax.set_title(label, fontsize=13, weight="bold", loc="left")

    for idx in range(num_runs, axes_flat.size):
        fig.delaxes(axes_flat[idx])

    for idx, ax in enumerate(axes_flat[:num_runs]):
        row = idx // axes.shape[1]
        col = idx % axes.shape[1]
        if row < axes.shape[0] - 1:
            ax.set_xlabel("")
            ax.set_xticklabels([])
        if col > 0:
            ax.set_ylabel("")

    legend_handles = [
        plt.Line2D([0], [0], color="tab:red", linewidth=10, alpha=0.8, label="Action = 1"),
        plt.Line2D([0], [0], color="tab:blue", linewidth=10, alpha=0.55, label="Action = -1"),
        plt.Line2D([0], [0], color="goldenrod", linestyle="--", linewidth=2, label="Switch Point"),
    ]
    fig.legend(
        legend_handles,
        [h.get_label() for h in legend_handles],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.91),
        ncol=len(legend_handles),
        fontsize=11,
    )
    fig.suptitle("Q-Values by State and Action", fontsize=16, weight="bold", y=0.96)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

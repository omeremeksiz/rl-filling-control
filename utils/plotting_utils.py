# utils/plotting_utils.py
from __future__ import annotations

from typing import Dict, Iterable, Mapping, Tuple, List, Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def plot_qvalue_vs_state_from_pair_table(q_table: Mapping[Tuple[int, int], float], out_path: str) -> None:
    states = sorted({state for state, _ in q_table.keys()})
    q_fast = [q_table.get((state, 1), 0.0) for state in states]
    q_slow = [q_table.get((state, -1), 0.0) for state in states]

    plt.figure(figsize=(14, 6), dpi=300)
    plt.bar(
        states,
        q_fast,
        width=1.0,
        color="tab:red",
        alpha=0.8,
        label="Action = 1",
        align="center",
        edgecolor="black",
        linewidth=0.5,
    )
    plt.bar(
        states,
        q_slow,
        width=1.0,
        color="tab:blue",
        alpha=0.55,
        label="Action = -1",
        align="center",
        edgecolor="black",
        linewidth=0.5,
    )

    plt.axhline(0, color="black", linewidth=0.8, alpha=0.5)
    plt.xticks(states, states, rotation=45, ha="right", fontsize=8)
    plt.yticks(fontsize=8)
    plt.title("Q-Values by State and Action")
    plt.xlabel("State (Weight)", fontsize=10)
    plt.ylabel("Q-Value", fontsize=10)
    plt.grid(axis="y", alpha=0.25)
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_qvalue_vs_state_bandit(q_table: Mapping[int, float], out_path: str) -> None:
    switch_points: List[int] = sorted(q_table.keys())
    q_vals: List[float] = [q_table[sp] for sp in switch_points]

    plt.figure(figsize=(12, 6))
    bars = plt.bar(switch_points, q_vals, alpha=0.7, color="skyblue")
    if switch_points:
        best_state = max(q_table, key=q_table.get)
        highlight_idx = switch_points.index(best_state)
        bars[highlight_idx].set_color("red")

    plt.title("Q-Value vs State (Switch Points)")
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

    plt.xlabel("Episode", fontsize=14)
    plt.ylabel("Switching Point", fontsize=14)
    plt.title("Switching Point Trajectory with Exploration", fontsize=16, weight="bold")
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    plt.legend(fontsize=10, loc="upper right")
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()

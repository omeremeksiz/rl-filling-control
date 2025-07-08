import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

class PlotHandler:
    def __init__(self, output_folder="output",
                 cluster_histogram_path="cluster_histogram.png",
                 qvalue_sate_plot_path="qvalue_vs_state.png",
                 switching_point_trajectory_path="switching_point_trajectory.png"):
        self.output_folder = output_folder
        self.cluster_histogram_path = cluster_histogram_path
        self.qvalue_sate_plot_path = qvalue_sate_plot_path
        self.switching_point_trajectory_path = switching_point_trajectory_path
        os.makedirs(self.output_folder, exist_ok=True)

    def plot_cluster_histogram(self, clusters: dict):
        """
        Plots a histogram showing the size of each cluster (number of fillings per switching point).
        """
        switching_points = sorted(clusters.keys())
        sizes = [len(clusters[sp]) for sp in switching_points]

        plt.figure(figsize=(12, 6), dpi=300)
        plt.bar(switching_points, sizes)
        plt.xlabel("Switching Point")
        plt.ylabel("Number of Fillings")
        plt.title("Histogram of Cluster Sizes")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_folder, self.cluster_histogram_path))
        plt.close()

    def plot_qvalue_vs_state(self, q_value_dict: dict):
        """
        Plots Q-value vs weight for the best action (between +1 and -1) at each weight.
        """
        from state import State  # Make sure to import State class

        weight_to_best_q = {}

        for state_obj, q_value in q_value_dict.items():
            weight = state_obj.weight
            action = state_obj.action

            # Track best Q-value per weight
            if weight not in weight_to_best_q:
                weight_to_best_q[weight] = (action, q_value)
            else:
                _, existing_q = weight_to_best_q[weight]
                if q_value > existing_q:
                    weight_to_best_q[weight] = (action, q_value)

        weights = sorted(weight_to_best_q.keys())
        best_q_values = [weight_to_best_q[w][1] for w in weights]

        plt.figure(figsize=(12, 6), dpi=300)
        plt.plot(weights, best_q_values, marker='o')
        plt.xlabel("Weight (State)")
        plt.ylabel("Q-Value (Best Action)")
        plt.title("Q-Value vs Weight for Best Actions")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_folder, self.qvalue_sate_plot_path))
        plt.close()

    def plot_switching_point_trajectory(self, all_episode_info: list, line_width=3.5):
        """
        Plots:
        - Blue line: model-selected switching point trajectory
        - Orange vertical lines: offset between model and explored switching points
        Clean integer axis ticks, no clutter.
        """
        episode_nums = [ep["episode_num"] for ep in all_episode_info]
        model_selected = [ep["model_selected_switching_point"] for ep in all_episode_info]

        # Create figure
        plt.figure(figsize=(14, 7), dpi=300)

        # Plot main model trajectory (blue line)
        plt.plot(episode_nums, model_selected, color="blue", linewidth=line_width, alpha=0.9, label="Model Selected")

        # Draw vertical lines for exploration offsets (orange)
        for ep in all_episode_info:
            if ep["explored_switching_point"] is not None:
                plt.plot(
                    [ep["episode_num"], ep["episode_num"]],
                    [ep["model_selected_switching_point"], ep["explored_switching_point"]],
                    color='orange',
                    linewidth=1.0,
                    alpha=0.6
                )

        # Add dummy line for legend entry
        plt.plot([], [], color='orange', linewidth=1.2, label='Explored')

        # Axis formatting
        ax = plt.gca()
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=15))
        ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True, prune='both', nbins=20))

        # Labels, grid, title
        plt.xlabel("Episode", fontsize=14)
        plt.ylabel("Switching Point", fontsize=14)
        plt.title("Switching Point Trajectory with Exploration", fontsize=16, weight='bold')
        plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
        plt.legend(fontsize=10, loc="upper right")
        plt.tight_layout()

        # Save
        save_path = os.path.join(self.output_folder, self.switching_point_trajectory_path)
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
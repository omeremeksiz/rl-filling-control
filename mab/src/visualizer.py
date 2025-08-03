"""
Visualization module for the container filling control system.
Provides plotting and analysis tools for training results.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, List
from q_learning_agent import QLearningAgent
from data_processor import DataProcessor
from config import DEFAULT_PENALTY_MULTIPLIER


class TrainingVisualizer:
    """Visualizes training results and Q-table analysis."""
    
    def __init__(self, agent: QLearningAgent, data_processor: DataProcessor):
        self.agent = agent
        self.data_processor = data_processor
        self.setup_plotting_style()
    
    def setup_plotting_style(self) -> None:
        """Set up consistent plotting style."""
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def plot_switching_point_trajectory(self, training_history: List[Dict], 
                                      save_path: str = None, line_width: float = None) -> None:
        """
        Plots:
        - Blue line: model-selected switching point trajectory
        - Orange vertical lines: offset between model and explored switching points
        Clean integer axis ticks, no clutter.
        
        Args:
            training_history: List of training episode statistics
            save_path: Optional path to save the plot
            line_width: Width of the main trajectory line (uses config default if None)
        """
        if not training_history:
            print("No training history to plot.")
            return
        
        import matplotlib.ticker as ticker
        
        # Extract data
        episode_nums = [ep['episode'] for ep in training_history]
        next_switch_points = [ep['next_switch_point'] for ep in training_history]
        
        # Create figure
        plt.figure(figsize=(14, 7), dpi=300)
        
        # Plot next switch point trajectory (blue line)
        plt.plot(episode_nums, next_switch_points, color="blue", linewidth=line_width, 
                alpha=0.9, label="Next Switch Point")
        

        
        # Axis formatting
        ax = plt.gca()
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=15))
        ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True, prune='both', nbins=20))
        
        # Labels, grid, title
        plt.xlabel("Episode", fontsize=14)
        plt.ylabel("Switching Point", fontsize=14)
        plt.title(f"Next Switch Point Trajectory (Penalty Multiplier: {DEFAULT_PENALTY_MULTIPLIER})", fontsize=16, weight='bold')
        plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
        plt.legend(fontsize=10, loc="upper right")
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    
    def plot_q_value_vs_state(self, save_path: str = None) -> None:
        """
        Plot Q-values vs state (switch points).
        
        Args:
            save_path: Optional path to save the plot
        """
        q_table = self.agent.get_q_table()
        
        if not q_table:
            print("No Q-table to plot.")
            return
        
        switch_points = list(q_table.keys())
        q_values = list(q_table.values())
        
        plt.figure(figsize=(12, 6), dpi=300)
        bars = plt.bar(switch_points, q_values, alpha=0.7, color='skyblue')
        
        # Highlight the best switch point
        best_switch_point = self.agent.get_optimal_switch_point()
        best_index = switch_points.index(best_switch_point)
        bars[best_index].set_color('red')
        bars[best_index].set_alpha(0.8)
        
        plt.title('Q-Value vs State (Switch Points)')
        plt.xlabel('Switch Point')
        plt.ylabel('Q-Value')
        plt.grid(True, alpha=0.3)
        
        # Add value labels on the best bar
        best_q_value = q_table[best_switch_point]
        plt.text(best_switch_point, best_q_value + 1, f'{best_q_value:.2f}', 
                ha='center', va='bottom', fontweight='bold')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_cluster_histogram(self, save_path: str = None) -> None:
        """
        Plot histogram of cluster sizes (switching point distribution).
        
        Args:
            save_path: Optional path to save the plot
        """
        sessions = self.data_processor.sessions
        
        if not sessions:
            print("No sessions to analyze.")
            return
        
        switch_points = [s.switch_point for s in sessions if s.switch_point is not None]
        
        plt.figure(figsize=(10, 6), dpi=300)
        plt.hist(switch_points, bins=20, alpha=0.7, color='blue', edgecolor='black')
        plt.title('Histogram of Cluster Sizes')
        plt.xlabel('Switching Point')
        plt.ylabel('Number of Fillings')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def print_training_summary(self, training_history: List[Dict]) -> None:
        """Print a summary of training results."""
        # This method is kept for backward compatibility but does nothing
        # The actual logging is now handled by the logger
        pass 
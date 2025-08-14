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
from config import DEFAULT_OVERFLOW_PENALTY_CONSTANT, DEFAULT_UNDERFLOW_PENALTY_CONSTANT


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
                                      save_path: str = None, line_width: float = 3.5) -> None:
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
        
        # Extract data - handle both legacy and new formats
        if 'episode_num' in training_history[0] and 'model_selected_switching_point' in training_history[0]:
            # New format with exploration data
            episode_nums = [ep['episode_num'] for ep in training_history]
            # Show what the model selected (before exploration)
            model_selected = [ep['model_selected_switching_point'] for ep in training_history]
        
        # Create figure
        plt.figure(figsize=(14, 7), dpi=300)
        
        # Plot main model trajectory (blue line)
        plt.plot(episode_nums, model_selected, color="blue", linewidth=line_width, 
                alpha=0.9, label="Model Selected")
        
        # Draw vertical lines for exploration offsets (orange) if data available
        if 'explored_switching_point' in training_history[0]:
            for ep in training_history:
                explored_choice = ep.get('explored_switching_point')
                if explored_choice is not None:
                    # Show exploration as a vertical line from model choice to explored choice
                    model_choice = ep.get('model_selected_switching_point')
                    if model_choice != explored_choice:
                        plt.plot(
                            [ep['episode_num'], ep['episode_num']],
                            [model_choice, explored_choice],
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
        
        # Handle different Q-table formats
        if not q_table:
            print("Empty Q-table.")
            return
            
        if isinstance(next(iter(q_table.keys())), tuple):
            # Monte Carlo format: (state, action) -> q_value
            # Separate Q-values by action for grouped bar chart
            fast_q_values = {}  # action = 1
            slow_q_values = {}  # action = -1
            
            for (state, action), q_value in q_table.items():
                # Filter out indicators (-1 and 300) - only show actual weights
                if state == -1 or state == 300:
                    continue
                if action == 1:
                    fast_q_values[state] = q_value
                elif action == -1:
                    slow_q_values[state] = q_value
            
            # Get all states that have either action
            all_states = sorted(set(fast_q_values.keys()) | set(slow_q_values.keys()))
            
            # Prepare data for grouped bar chart
            fast_values = [fast_q_values.get(state, 0) for state in all_states]
            slow_values = [slow_q_values.get(state, 0) for state in all_states]
            
            # Create stacked bar chart
            import numpy as np
            
            plt.figure(figsize=(16, 6), dpi=300)
            
            # Create overlapping bars starting from 0 for direct comparison
            bars1 = plt.bar(all_states, fast_values, label='Fast (action=1)', alpha=0.9, color='blue')
            bars2 = plt.bar(all_states, slow_values, label='Slow (action=-1)', alpha=0.7, color='orange')
            
            # Mark the switching point with a simple red dashed line
            best_switch_point = self.agent.get_optimal_switch_point()
            if best_switch_point in all_states:
                plt.axvline(x=best_switch_point, color='red', linestyle='--', linewidth=2, alpha=0.8, 
                           label=f'Switching Point ({best_switch_point})')
            
            plt.xlabel('State (Weight)', fontsize=12)
            plt.ylabel('Q-Value', fontsize=12)
            plt.title('Q-Values for Both Actions by State (Monte Carlo)', fontsize=14, fontweight='bold')
            
            # Better x-axis formatting - show every 5th state for readability
            if len(all_states) > 20:
                step = max(1, len(all_states) // 15)  # Show about 15 labels max
                plt.xticks(all_states[::step], rotation=45)
            else:
                plt.xticks(all_states, rotation=45)
            
            plt.legend(fontsize=11)
            plt.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            
        else:
            # MAB format: switch_point -> q_value (single action per state)
            switch_point_q_values = q_table
            switch_points = list(switch_point_q_values.keys())
            q_values = list(switch_point_q_values.values())
            
            plt.figure(figsize=(12, 6), dpi=300)
            bars = plt.bar(switch_points, q_values, alpha=0.7, color='skyblue')
            
            # Highlight the best switch point
            best_switch_point = self.agent.get_optimal_switch_point()
            
            # Check if best switch point exists in the plot data
            if best_switch_point in switch_point_q_values:
                best_index = switch_points.index(best_switch_point)
                bars[best_index].set_color('red')
                bars[best_index].set_alpha(0.8)
                
            else:
                # If best switch point not in plot, add a vertical line
                plt.axvline(x=best_switch_point, color='red', linestyle='--', alpha=0.8, 
                           label=f'Optimal: {best_switch_point}')
            
            plt.title('Q-Value vs State (Switch Points)')
            plt.xlabel('Switch Point')
            plt.ylabel('Q-Value')
            plt.grid(True, alpha=0.3)
        
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
"""
Logging system for the container filling control system.
Handles training ID generation, output folder management, and detailed logging.
"""

import os
import uuid
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List
from config import DATA_FILE_PATH


class TrainingLogger:
    """Manages logging and output organization for training sessions."""
    
    def __init__(self, output_base_dir: str = "output"):
        self.output_base_dir = output_base_dir
        self.training_id = self._generate_training_id()
        self.output_dir = os.path.join(output_base_dir, self.training_id)
        self.log_file_path = os.path.join(self.output_dir, "training_process.log")
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Setup logging
        self._setup_logging()
        
        # Log initial information
        self._log_initial_info()
    
    def _generate_training_id(self) -> str:
        """Generate a unique training ID with timestamp."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:6]
        return f"{timestamp}_{unique_id}"
    
    def _setup_logging(self) -> None:
        """Setup logging configuration."""
        # Create a unique logger for this training session
        logger_name = f"training_{self.training_id}"
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(logging.INFO)
        
        # Clear any existing handlers
        self.logger.handlers.clear()
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        
        # File handler
        file_handler = logging.FileHandler(self.log_file_path)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
        # Console handler (optional - we already have console output)
        # console_handler = logging.StreamHandler()
        # console_handler.setFormatter(formatter)
        # self.logger.addHandler(console_handler)
        
        # Prevent propagation to root logger
        self.logger.propagate = False
    
    def _log_initial_info(self) -> None:
        """Log initial training information."""
        self.logger.info(f"Job ID: {self.training_id}")
        self.logger.info(f"All experiment outputs will be saved in: {self.output_dir}")
        self.logger.info("")
        self.logger.info(f"=== Training started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")
        self.logger.info("")
    
    def log_experiment_config(self, config: Dict[str, Any]) -> None:
        """Log experiment configuration."""
        self.logger.info("=== Experiment Configuration ===")
        
        # Data path
        self.logger.info("[DATA_PATH]")
        self.logger.info(f"{DATA_FILE_PATH}")
        self.logger.info("")
        
        # Output paths
        self.logger.info("[OUTPUT_PATHS]")
        output_paths = {
            'qvalue_vs_state_path': os.path.join(self.output_dir, "qvalue_vs_state.png"),
            'switching_point_trajectory_path': os.path.join(self.output_dir, "switching_point_trajectory.png"),
            'log_file_path': self.log_file_path
        }
        
        for key, path in output_paths.items():
            self.logger.info(f"{key}: {path}")
        self.logger.info("")
        
        # Training parameters
        self.logger.info("[TRAINING]")
        training_params = {
            'rl_method': config.get('rl_method', 'mab'),
            'random_seed': config.get('random_seed', 42),
            'learning_rate': config.get('learning_rate', 0.1),
            'exploration_probability': config.get('exploration_rate', 0.1),
            'min_weight': config.get('safe_weight_min', 80),
            'max_weight': config.get('safe_weight_max', 120),
            'overflow_penalty_constant': config.get('overflow_penalty_constant', -10.0),
            'underflow_penalty_constant': config.get('underflow_penalty_constant', -10.0),
            'starting_switch_point': config.get('starting_switch_point', 'Random')
        }
        
        for key, value in training_params.items():
            self.logger.info(f"{key}: {value}")
        self.logger.info("")
    
    def log_cluster_information(self, data_processor) -> None:
        """Log cluster information from data processor."""
        self.logger.info("=== Cluster Information ===")
        
        # Get cluster statistics
        switch_point_clusters = data_processor.switch_point_clusters
        total_sessions = len(data_processor.sessions)
        
        # Create cluster summary
        cluster_summary = []
        for switch_point in sorted(switch_point_clusters.keys()):
            count = len(switch_point_clusters[switch_point])
            cluster_summary.append((switch_point, count))
        
        # Log cluster information
        self.logger.info("Switching Point     Number of Fillings")
        self.logger.info("-------------------- --------------------")
        
        for switch_point, count in cluster_summary:
            self.logger.info(f"{switch_point:<20} {count}")
        
        self.logger.info("")
        self.logger.info(f"Total Switching Points (Clusters): {len(cluster_summary)}")
        self.logger.info(f"Total Episodes: {total_sessions}")
        self.logger.info("")
    
    def log_episode(self, episode_num: int, total_episodes: int, 
                   experienced_switch_point: int, termination_type: str,
                   model_selected_next: int, explored_switch_point: Optional[int] = None) -> None:
        """Log individual episode information."""
        self.logger.info(f"--- Episode {episode_num}/{total_episodes} ---")
        self.logger.info(f"Experienced Switching Point: {experienced_switch_point}")
        self.logger.info(f"Termination Type: {termination_type}")
        self.logger.info(f"Model-Selected Next Switching Point: {model_selected_next}")
        
        if explored_switch_point is not None:
            self.logger.info(f"Explored Switching Point: {explored_switch_point}")
        else:
            self.logger.info("Explored Switching Point: None")
        self.logger.info("")
    
    def log_training_completion(self, training_history: list, q_table: Dict[int, float]) -> None:
        """Log training completion and save results."""
        self.logger.info("=== Training finished ===")
        self.logger.info("")
        
        # Save plots
        self._save_plots(training_history, q_table)
        
        self.logger.info(f"=== Training finished at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")
    
    def log_training_summary(self, training_history: List[Dict]) -> None:
        """Log training summary to the log file."""
        if not training_history:
            self.logger.info("No training history to summarize.")
            return
        
        # Calculate statistics
        total_episodes = len(training_history)
        
        self.logger.info("")
        self.logger.info("="*50)
        self.logger.info("TRAINING SUMMARY")
        self.logger.info("="*50)
        self.logger.info(f"Total Episodes: {total_episodes}")
        self.logger.info("="*50)
    
    def log_optimal_policy(self, optimal_switch_point: int, safe_weight_range: tuple) -> None:
        """Log optimal policy to the log file."""
        self.logger.info("")
        self.logger.info("="*50)
        self.logger.info("LEARNED OPTIMAL POLICY")
        self.logger.info("="*50)
        self.logger.info(f"Optimal Switch Point: {optimal_switch_point}")
        self.logger.info(f"Safe Weight Range: [{safe_weight_range[0]}, {safe_weight_range[1]}]")
        self.logger.info("="*50)
    
    
    def _save_plots(self, training_history: list, q_table: Dict[int, float]) -> None:
        """Save visualization plots."""
        # Q-value vs state plot
        qvalue_plot_path = os.path.join(self.output_dir, "qvalue_vs_state.png")
        self._create_qvalue_vs_state_plot(q_table, qvalue_plot_path)
        self.logger.info(f"Q-value vs state plot is saved to: {qvalue_plot_path}")
        
        # Switching point trajectory plot
        trajectory_plot_path = os.path.join(self.output_dir, "switching_point_trajectory.png")
        self._create_trajectory_plot(training_history, trajectory_plot_path)
        self.logger.info(f"Switching point trajectory plot is saved to: {trajectory_plot_path}")
    
    def _create_qvalue_vs_state_plot(self, q_table: Dict, file_path: str) -> None:
        """Create Q-value vs state plot."""
        import matplotlib.pyplot as plt
        
        # Handle different Q-table formats
        if isinstance(next(iter(q_table.keys())), tuple):
            # Monte Carlo Q-table: (state, action) -> value
            # Convert to switch point -> max Q-value format
            switch_point_q_values = {}
            for (state, action), q_value in q_table.items():
                if state not in switch_point_q_values:
                    switch_point_q_values[state] = {}
                switch_point_q_values[state][action] = q_value
            
            # Get max Q-value for each state
            switch_points = []
            q_values = []
            for state in sorted(switch_point_q_values.keys()):
                max_q_value = max(switch_point_q_values[state].values())
                switch_points.append(state)
                q_values.append(max_q_value)
        else:
            # Standard Q-table: switch_point -> value
            switch_points = list(q_table.keys())
            q_values = list(q_table.values())
        
        plt.figure(figsize=(12, 6))
        bars = plt.bar(switch_points, q_values, alpha=0.7, color='skyblue')
        
        # Highlight the best switch point
        best_switch_point = max(q_table, key=q_table.get) if not isinstance(next(iter(q_table.keys())), tuple) else max(switch_point_q_values, key=lambda x: max(switch_point_q_values[x].values()))
        best_index = switch_points.index(best_switch_point)
        bars[best_index].set_color('red')
        bars[best_index].set_alpha(0.8)
        
        plt.title('Q-Value vs State (Switch Points)')
        plt.xlabel('Switch Point')
        plt.ylabel('Q-Value')
        plt.grid(True, alpha=0.3)
        
        # Add value labels on the best bar
        best_q_value = q_values[best_index]
        plt.text(best_switch_point, best_q_value + 1, f'{best_q_value:.2f}', 
                ha='center', va='bottom', fontweight='bold')
        
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_trajectory_plot(self, training_history: list, file_path: str, line_width=1.5) -> None:
        """Create switching point trajectory plot."""
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
        
        # Extract data - handle different training history formats
        episode_nums = [ep['episode'] for ep in training_history]
        
        # Check if it's MAB format (has 'next_switch_point') or MC format (has 'switch_point')
        if 'next_switch_point' in training_history[0]:
            switch_points = [ep['next_switch_point'] for ep in training_history]
        else:
            switch_points = [ep['switch_point'] for ep in training_history]
        
        # Create figure
        plt.figure(figsize=(14, 7), dpi=300)
        
        # Plot switch point trajectory (blue line)
        plt.plot(episode_nums, switch_points, color="blue", linewidth=line_width, 
                alpha=0.9, label="Switch Point")
        
        # Axis formatting
        ax = plt.gca()
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=15))
        ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True, prune='both', nbins=20))
        
        # Labels, grid, title
        plt.xlabel("Episode", fontsize=14)
        plt.ylabel("Switching Point", fontsize=14)
        plt.title("Switch Point Trajectory", fontsize=16, weight='bold')
        plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
        plt.legend(fontsize=10, loc="upper right")
        plt.tight_layout()
        
        plt.savefig(file_path, bbox_inches='tight')
        plt.close()
    
    def get_output_paths(self) -> Dict[str, str]:
        """Get all output file paths."""
        return {
            'qvalue_vs_state_path': os.path.join(self.output_dir, "qvalue_vs_state.png"),
            'switching_point_trajectory_path': os.path.join(self.output_dir, "switching_point_trajectory.png"),
            'log_file_path': self.log_file_path
        } 
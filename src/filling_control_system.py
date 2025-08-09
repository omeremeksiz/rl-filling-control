"""
Main application for optimal switching point learning.
Orchestrates the complete training pipeline with clean separation of concerns.
"""

import sys
from typing import Optional
from config import (
    DATA_FILE_PATH, DEFAULT_TRAINING_EPISODES, DEFAULT_LEARNING_RATE, 
    DEFAULT_EXPLORATION_RATE, DEFAULT_SAFE_WEIGHT_MIN, DEFAULT_SAFE_WEIGHT_MAX,
    DEFAULT_RANDOM_SEED, DEFAULT_STARTING_SWITCH_POINT, DEFAULT_RL_METHOD,
    DEFAULT_OVERFLOW_PENALTY_CONSTANT, DEFAULT_UNDERFLOW_PENALTY_CONSTANT
)
from data_processor import DataProcessor
from reward_calculator import RewardCalculator
from agent_factory import AgentFactory
from visualizer import TrainingVisualizer
from logger import TrainingLogger


class FillingControlSystem:
    """Main system orchestrating the optimal switching point learning."""
    
    def __init__(self, 
                 data_file_path: str = DATA_FILE_PATH,
                 safe_weight_min: int = DEFAULT_SAFE_WEIGHT_MIN,
                 safe_weight_max: int = DEFAULT_SAFE_WEIGHT_MAX,
                 learning_rate: float = DEFAULT_LEARNING_RATE,
                 exploration_rate: float = DEFAULT_EXPLORATION_RATE,
                 random_seed: int = DEFAULT_RANDOM_SEED,
                 starting_switch_point: Optional[int] = DEFAULT_STARTING_SWITCH_POINT,
                 overflow_penalty_constant: float = DEFAULT_OVERFLOW_PENALTY_CONSTANT,
                 underflow_penalty_constant: float = DEFAULT_UNDERFLOW_PENALTY_CONSTANT,
                 rl_method: str = DEFAULT_RL_METHOD):
        """
        Initialize the filling control system.
        
        Args:
            data_file_path: Path to the Excel data file
            safe_weight_min: Minimum safe weight
            safe_weight_max: Maximum safe weight
            learning_rate: Q-learning learning rate
            exploration_rate: Exploration rate for epsilon-greedy policy
        """
        self.data_file_path = data_file_path
        self.safe_weight_min = safe_weight_min
        self.safe_weight_max = safe_weight_max
        self.learning_rate = learning_rate
        self.exploration_rate = exploration_rate
        self.random_seed = random_seed
        self.starting_switch_point = starting_switch_point
        self.overflow_penalty_constant = overflow_penalty_constant
        self.underflow_penalty_constant = underflow_penalty_constant
        self.rl_method = rl_method
        
        # Initialize components
        self.data_processor = None
        self.reward_calculator = None
        self.agent = None
        self.visualizer = None
        self.logger = None
        
        self._initialize_components()
    
    def _initialize_components(self) -> None:
        """Initialize all system components."""
        try:
            # Load and process data
            self.data_processor = DataProcessor(self.data_file_path)
            self.data_processor.load_data()
            
            # Initialize reward calculator
            self.reward_calculator = RewardCalculator(
                safe_weight_min=self.safe_weight_min,
                safe_weight_max=self.safe_weight_max,
                overflow_penalty_constant=self.overflow_penalty_constant,
                underflow_penalty_constant=self.underflow_penalty_constant
            )
            
            # Initialize RL agent using factory
            self.agent = AgentFactory.create_agent(
                method=self.rl_method,
                data_processor=self.data_processor,
                reward_calculator=self.reward_calculator,
                learning_rate=self.learning_rate,
                exploration_rate=self.exploration_rate,
                random_seed=self.random_seed
            )
            
            # Initialize visualizer
            self.visualizer = TrainingVisualizer(self.agent, self.data_processor)
            
            # Initialize logger
            self.logger = TrainingLogger()
            
        except Exception as e:
            print(f"Error initializing system: {e}")
            sys.exit(1)
    
    def train(self, num_episodes: int = DEFAULT_TRAINING_EPISODES, 
              initial_switch_point: Optional[int] = None) -> list:
        """
        Train the agent for the specified number of episodes.
        
        Args:
            num_episodes: Number of training episodes
            initial_switch_point: Starting switch point (random if None)
            
        Returns:
            Training history
        """
        # Use the system's starting switch point if not provided
        if initial_switch_point is None:
            initial_switch_point = self.starting_switch_point
            
        # Log experiment configuration
        config = {
            'rl_method': self.rl_method,
            'learning_rate': self.learning_rate,
            'exploration_rate': self.exploration_rate,
            'safe_weight_min': self.safe_weight_min,
            'safe_weight_max': self.safe_weight_max,
            'starting_switch_point': initial_switch_point,
            'overflow_penalty_constant': self.overflow_penalty_constant,
            'underflow_penalty_constant': self.underflow_penalty_constant,
            'random_seed': self.random_seed
        }
        self.logger.log_experiment_config(config)
        
        # Log cluster information
        self.logger.log_cluster_information(self.data_processor)
        
        training_history = self.agent.train(num_episodes, initial_switch_point, self.logger)
        
        # Log training completion
        self.logger.log_training_completion(training_history, self.agent.get_q_table())
        
        return training_history
    
    def analyze_results(self, training_history: list, save_plots: bool = True) -> None:
        """
        Analyze and visualize training results.
        
        Args:
            training_history: Training episode history
            save_plots: Whether to save plots to files
        """
        # Log training summary
        self.logger.log_training_summary(training_history)
        
        # Get output paths from logger
        output_paths = self.logger.get_output_paths()
        
        # Plot 1: Switching point trajectory
        if save_plots:
            self.visualizer.plot_switching_point_trajectory(training_history, output_paths['switching_point_trajectory_path'])
        else:
            self.visualizer.plot_switching_point_trajectory(training_history)
        
        # Plot 2: Cluster histogram
        if save_plots:
            self.visualizer.plot_cluster_histogram(output_paths['cluster_histogram_path'])
        else:
            self.visualizer.plot_cluster_histogram()
        
        # Plot 3: Q-value vs state
        if save_plots:
            self.visualizer.plot_q_value_vs_state(output_paths['qvalue_vs_state_path'])
        else:
            self.visualizer.plot_q_value_vs_state()
    
    def get_optimal_policy(self) -> dict:
        """
        Get the learned optimal policy.
        
        Returns:
            Dictionary containing optimal switch point and Q-table
        """
        optimal_switch_point = self.agent.get_optimal_switch_point()
        q_table = self.agent.get_q_table()
        
        return {
            'optimal_switch_point': optimal_switch_point,
            'q_table': q_table,
            'safe_weight_range': (self.safe_weight_min, self.safe_weight_max)
        }
    
    def print_optimal_policy(self) -> None:
        """Log the learned optimal policy to the log file."""
        policy = self.get_optimal_policy()
        
        self.logger.log_optimal_policy(
            policy['optimal_switch_point'], 
            policy['safe_weight_range']
        )


def main():
    """Main function to run the filling control system with all parameters from config."""
    
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='RL Filling Control System')
    parser.add_argument('--method', type=str, default=DEFAULT_RL_METHOD, 
                       choices=['mab', 'mc', 'td', 'qlearning', 'q'], 
                       help='RL method to use (mab, mc, td, qlearning, or q)')
    parser.add_argument('--episodes', type=int, default=DEFAULT_TRAINING_EPISODES,
                       help='Number of training episodes')
    
    args = parser.parse_args()
    
    # Initialize the system with specified method
    system = FillingControlSystem(rl_method=args.method)
    
    # Train the agent
    training_history = system.train(
        num_episodes=args.episodes,
        initial_switch_point=DEFAULT_STARTING_SWITCH_POINT
    )
    
    # Analyze results
    system.analyze_results(training_history)
    
    # Print optimal policy
    system.print_optimal_policy()


if __name__ == "__main__":
    main() 
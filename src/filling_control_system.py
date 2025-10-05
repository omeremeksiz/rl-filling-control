"""
Main application for optimal switching point learning.
Orchestrates the complete training pipeline with clean separation of concerns.
"""

import sys
from typing import Optional
from config import (
    DATA_FILE_PATH, DEFAULT_TRAINING_EPISODES, DEFAULT_TESTING_EPISODES, DEFAULT_LEARNING_RATE, 
    DEFAULT_EXPLORATION_RATE, DEFAULT_SAFE_WEIGHT_MIN, DEFAULT_SAFE_WEIGHT_MAX,
    DEFAULT_RANDOM_SEED, DEFAULT_STARTING_SWITCH_POINT, DEFAULT_RL_METHOD,
    DEFAULT_OVERFLOW_PENALTY_CONSTANT, DEFAULT_UNDERFLOW_PENALTY_CONSTANT,
    DEFAULT_EXPLORATION_DECAY, DEFAULT_EXPLORATION_MIN_RATE,
    DEFAULT_EXPLORATION_DECAY_RATE, DEFAULT_EXPLORATION_DECAY_INTERVAL,
    DEFAULT_OPERATION_MODE, DEFAULT_WEIGHT_QUANTIZATION_STEP
)
from data_processor import DataProcessor
from reward_calculator import RewardCalculator
from agent_factory import AgentFactory
from visualizer import TrainingVisualizer
from logger import TrainingLogger
from real_world_tester import RealWorldTester


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
                 rl_method: str = DEFAULT_RL_METHOD,
                 exploration_decay: bool = DEFAULT_EXPLORATION_DECAY,
                 exploration_min_rate: float = DEFAULT_EXPLORATION_MIN_RATE,
                 exploration_decay_rate: float = DEFAULT_EXPLORATION_DECAY_RATE,
                 exploration_decay_interval: int = DEFAULT_EXPLORATION_DECAY_INTERVAL,
                 operation_mode: str = DEFAULT_OPERATION_MODE):
        """
        Initialize the filling control system.
        
        Args:
            data_file_path: Path to the Excel data file
            safe_weight_min: Minimum safe weight
            safe_weight_max: Maximum safe weight
            learning_rate: Q-learning learning rate
            exploration_rate: Initial exploration rate for epsilon-greedy policy
            exploration_decay: Whether to use exploration decay
            exploration_min_rate: Minimum exploration rate
            exploration_decay_rate: Decay factor when decay occurs
            exploration_decay_interval: Decay every N episodes
            operation_mode: "train" for simulation or "test" for real-world testing
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
        self.exploration_decay = exploration_decay
        self.exploration_min_rate = exploration_min_rate
        self.exploration_decay_rate = exploration_decay_rate
        self.exploration_decay_interval = exploration_decay_interval
        self.operation_mode = operation_mode
        
        # Initialize components
        self.data_processor = None
        self.reward_calculator = None
        self.agent = None
        self.visualizer = None
        self.logger = None
        self.real_world_tester = None
        
        self._initialize_components()
    
    def _initialize_components(self) -> None:
        """Initialize all system components."""
        try:
            # Initialize reward calculator (needed for both modes)
            self.reward_calculator = RewardCalculator(
                safe_weight_min=self.safe_weight_min,
                safe_weight_max=self.safe_weight_max,
                overflow_penalty_constant=self.overflow_penalty_constant,
                underflow_penalty_constant=self.underflow_penalty_constant
            )
            
            if self.operation_mode.lower() == "train":
                # Training mode: use simulation data
                self.data_processor = DataProcessor(self.data_file_path)
                self.data_processor.load_excel(self.data_file_path)
                
                # Initialize RL agent using factory
                self.agent = AgentFactory.create_agent(
                    method=self.rl_method,
                    data_processor=self.data_processor,
                    reward_calculator=self.reward_calculator,
                    learning_rate=self.learning_rate,
                    exploration_rate=self.exploration_rate,
                    random_seed=self.random_seed,
                    exploration_decay=self.exploration_decay,
                    exploration_min_rate=self.exploration_min_rate,
                    exploration_decay_rate=self.exploration_decay_rate,
                    exploration_decay_interval=self.exploration_decay_interval
                )
                
                # Initialize visualizer and logger for training
                self.visualizer = TrainingVisualizer(self.agent, self.data_processor)
                self.logger = TrainingLogger()
                
            elif self.operation_mode.lower() == "test":
                # Real-world testing mode: initialize RL agent + real-world tester
                # Create a minimal data processor for agent initialization (needed for Q-table structure)
                self.data_processor = DataProcessor()
                
                # Initialize RL agent using factory (same as training)
                self.agent = AgentFactory.create_agent(
                    method=self.rl_method,
                    data_processor=self.data_processor,
                    reward_calculator=self.reward_calculator,
                    learning_rate=self.learning_rate,
                    exploration_rate=self.exploration_rate,
                    random_seed=self.random_seed,
                    exploration_decay=self.exploration_decay,
                    exploration_min_rate=self.exploration_min_rate,
                    exploration_decay_rate=self.exploration_decay_rate,
                    exploration_decay_interval=self.exploration_decay_interval
                )
                
                # Initialize logger for testing sessions
                self.logger = TrainingLogger(output_base_dir="output")
                
                # Initialize real-world tester
                self.real_world_tester = RealWorldTester(
                    reward_calculator=self.reward_calculator
                )
                print(f"Real-world testing mode initialized with {self.rl_method} agent")
            else:
                raise ValueError(f"Invalid operation mode: {self.operation_mode}. Use 'train' or 'test'")
            
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
            
        # Print training configuration to console (same format as log)
        print(f"\n=== Experiment Configuration ===")
        print(f"[TRAINING]")
        print(f"rl_method: {self.rl_method}")
        print(f"random_seed: {self.random_seed}")
        print(f"learning_rate: {self.learning_rate}")
        print(f"exploration_probability: {self.exploration_rate}")
        print(f"min_weight: {self.safe_weight_min}")
        print(f"max_weight: {self.safe_weight_max}")
        print(f"starting_switch_point: {initial_switch_point}")
        print(f"\n[OUTPUT_PATHS]")
        print(f"Output Directory: {self.logger.output_dir}")
        print()

        # Log experiment configuration to file
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
        
        # Print completion message
        print(f"\n=== Training Completed ===")
        print(f"Total Episodes: {num_episodes}")
        print(f"Results saved to: {self.logger.output_dir}")
        print("=" * 40)
        
        # Log training completion to file
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
        
        # Plot 2: Q-value vs state
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
    
    def test_with_rl_agent(self, num_episodes: int = 10) -> list:
        """
        Test using RL agent with real-world device (agent learns from real episodes).
        
        Args:
            num_episodes: Number of real-world episodes to run
            
        Returns:
            List of episode results
        """
        if self.operation_mode.lower() != "test":
            raise ValueError("test_with_rl_agent() can only be called in 'test' operation mode")
        
        if not self.real_world_tester:
            raise ValueError("Real-world tester not initialized")
            
        if not self.agent:
            raise ValueError("RL agent not initialized")
        
        # Print testing configuration (similar to training)
        print(f"\n=== Real-World Testing Configuration ===")
        print(f"[TESTING]")
        print(f"rl_method: {self.rl_method}")
        print(f"num_episodes: {num_episodes}")
        print(f"safe_weight_min: {self.safe_weight_min}")
        print(f"safe_weight_max: {self.safe_weight_max}")
        print(f"starting_switch_point: {self.starting_switch_point}")
        print(f"\n[OUTPUT_PATHS]")
        print(f"Output Directory: {self.logger.output_dir}")
        print()
        
        print(f"Starting real-world testing with {self.rl_method} agent for {num_episodes} episodes...")
        return self.real_world_tester.run_agent_testing(self.agent, num_episodes, self.starting_switch_point)
    
    def test_with_switching_points(self, switching_points: list) -> list:
        """
        Test with specified switching points using real-world device (manual mode).
        
        Args:
            switching_points: List of switching points to test
            
        Returns:
            List of episode results
        """
        if self.operation_mode.lower() != "test":
            raise ValueError("test_with_switching_points() can only be called in 'test' operation mode")
        
        if not self.real_world_tester:
            raise ValueError("Real-world tester not initialized")
        
        print(f"Starting manual testing with {len(switching_points)} switching points...")
        return self.real_world_tester.run_manual_testing(switching_points)

def main():
    """Main function to run the filling control system using config.py settings."""
    
    import argparse
    
    # Simple argument parser for optional overrides
    parser = argparse.ArgumentParser(description='RL Filling Control System')
    parser.add_argument('--method', type=str, default=DEFAULT_RL_METHOD, 
                       choices=['mab', 'mc', 'td', 'qlearning'], 
                       help='RL method to use (overrides config.py)')
    parser.add_argument('--episodes', type=int, default=None,
                       help='Number of episodes (overrides config.py)')
    
    args = parser.parse_args()
    
    # Initialize the system with config.py settings (can be overridden by CLI)
    system = FillingControlSystem(
        rl_method=args.method,
        # All other parameters come from config.py defaults
    )
    
    # Determine number of episodes based on mode and arguments
    if DEFAULT_OPERATION_MODE.lower() == "train":
        num_episodes = args.episodes if args.episodes else DEFAULT_TRAINING_EPISODES
        
        # Training mode
        print(f"Starting training with {args.method} method for {num_episodes} episodes...")
        training_history = system.train(
            num_episodes=num_episodes,
            initial_switch_point=DEFAULT_STARTING_SWITCH_POINT
        )
        
        # Analyze results
        system.analyze_results(training_history)
        
        # Print optimal policy
        system.print_optimal_policy()
        
    elif DEFAULT_OPERATION_MODE.lower() == "test":
        num_episodes = args.episodes if args.episodes else DEFAULT_TESTING_EPISODES
        
        # Real-world testing mode using RL agent
        results = system.test_with_rl_agent(num_episodes=num_episodes)
        
        print(f"\n=== Testing Completed ===")
        print(f"Total Episodes: {len(results)}")
        print(f"Results saved to: {system.logger.output_dir}")
        print(f"========================================")
        
        # Show final policy learned from real-world data
        if system.agent:
            policy = system.get_optimal_policy()
            print(f"Final optimal switching point learned: {policy['optimal_switch_point']}")
    
    else:
        print(f"Error: Invalid operation mode '{DEFAULT_OPERATION_MODE}' in config.py")
        print("Set DEFAULT_OPERATION_MODE to 'train' or 'test' in src/config.py")


if __name__ == "__main__":
    main() 
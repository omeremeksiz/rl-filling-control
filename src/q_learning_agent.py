"""
Q-learning agent for optimal switching point learning.
Implements stateless reinforcement learning for container filling control.
"""

import numpy as np
import random
from typing import Dict, List, Optional, Tuple
from config import DEFAULT_LEARNING_RATE, DEFAULT_EXPLORATION_RATE, DEFAULT_RANDOM_SEED, EXPLORATION_STEPS, EXPLORATION_PROBABILITIES
from data_processor import DataProcessor, FillingSession
from reward_calculator import RewardCalculator
from base_agent import BaseRLAgent


class QLearningAgent(BaseRLAgent):
    """Q-learning agent for learning optimal switching points."""
    
    def __init__(self, 
                 data_processor: DataProcessor,
                 reward_calculator: RewardCalculator,
                 learning_rate: float = DEFAULT_LEARNING_RATE,
                 exploration_rate: float = DEFAULT_EXPLORATION_RATE,
                 random_seed: int = DEFAULT_RANDOM_SEED):
        super().__init__(data_processor, reward_calculator, exploration_rate, random_seed)
        self.learning_rate = learning_rate
        
        # Set random seed for reproducibility
        random.seed(random_seed)
        
        # Initialize Q-table
        self.q_table = self._initialize_q_table()
    
    def _initialize_q_table(self) -> Dict[int, float]:
        """Initialize Q-table with zero values for all available switch points."""
        return {switch_point: 0.0 for switch_point in self.available_switch_points}
    
    def _get_best_switch_point(self) -> int:
        """Get the switch point with the highest Q-value."""
        best_switch_point = max(self.q_table, key=self.q_table.get)
        return best_switch_point
    
    def train_episode(self, current_switch_point: int) -> Tuple[float, int, int]:
        """
        Train on a single episode using the given switch point.
        
        Args:
            current_switch_point: Switch point to use for this episode
            
        Returns:
            Tuple of (reward, episode_length, final_weight)
        """
        # Get unused sessions from the cluster of current switch point
        unused_sessions = self.data_processor.get_unused_sessions_for_switch_point(current_switch_point)
        
        if not unused_sessions:
            # If no sessions for this switch point, use a random session
            all_sessions = self.data_processor.sessions
            selected_session = random.choice(all_sessions)
        else:
            selected_session = random.choice(unused_sessions)
            # Mark this session as used
            self.data_processor.mark_session_as_used(current_switch_point, selected_session)
        
        # Simulate the episode
        episode_length = selected_session.episode_length
        final_weight = selected_session.final_weight
        
        # Calculate reward
        reward = self.reward_calculator.calculate_reward(episode_length, final_weight)
        
        # Update Q-value
        self._update_q_value(current_switch_point, reward)
        
        return reward, episode_length, final_weight
    
    def _update_q_value(self, switch_point: int, reward: float) -> None:
        """
        Update Q-value using stateless Q-learning update rule.
        
        Args:
            switch_point: The switch point being updated
            reward: The reward received
        """
        current_q_value = self.q_table[switch_point]
        new_q_value = current_q_value + self.learning_rate * (reward - current_q_value)
        self.q_table[switch_point] = new_q_value
    
    def train(self, num_episodes: int, initial_switch_point: Optional[int] = None, logger=None) -> List[Dict]:
        """
        Train the agent for multiple episodes.
        
        Args:
            num_episodes: Number of training episodes
            initial_switch_point: Starting switch point (random if None)
            logger: Optional logger for detailed logging
            
        Returns:
            List of training statistics for each episode
        """
        if initial_switch_point is None:
            current_switch_point = random.choice(self.available_switch_points)
        else:
            if initial_switch_point not in self.available_switch_points:
                # If the given starting switch point is not available, 
                # use the smallest available switching point
                current_switch_point = min(self.available_switch_points)
                if logger:
                    logger.logger.info(f"Starting switch point {initial_switch_point} is not available.")
                    logger.logger.info(f"Using smallest available switching point: {current_switch_point}")
            else:
                current_switch_point = initial_switch_point
        
        self.training_history = []
        
        for episode in range(num_episodes):
            # Train on current episode
            reward, episode_length, final_weight = self.train_episode(current_switch_point)
            
            # Determine termination type
            termination_type = self._determine_termination_type(final_weight)
            
            # Get the best action (what model would select)
            best_switch_point = self._get_best_switch_point()
            
            # Select next action for next episode (may include exploration)
            next_switch_point = self.select_action(current_switch_point)
            
            # Determine if exploration occurred
            explored_switch_point = None
            if next_switch_point != best_switch_point:
                explored_switch_point = next_switch_point
            
            # Record training statistics
            episode_stats = {
                'episode': episode + 1,
                'episode_num': episode + 1,  # For compatibility with desired plot format
                'switch_point': current_switch_point,
                'model_selected_switching_point': best_switch_point,
                'explored_switching_point': explored_switch_point,
                'reward': reward,
                'episode_length': episode_length,
                'final_weight': final_weight,
                'q_value': self.q_table[current_switch_point],
                'termination_type': termination_type,
                'next_switch_point': next_switch_point
            }
            self.training_history.append(episode_stats)
            
            # Log episode if logger is provided
            if logger:
                logger.log_episode(
                    episode_num=episode + 1,
                    total_episodes=num_episodes,
                    experienced_switch_point=current_switch_point,
                    termination_type=termination_type,
                    model_selected_next=next_switch_point,
                    explored_switch_point=explored_switch_point
                )
            
            # Update current switch point
            current_switch_point = next_switch_point
        
        return self.training_history
    
    def _determine_termination_type(self, final_weight: int) -> str:
        """Determine the termination type based on final weight."""
        if self.reward_calculator._is_weight_safe(final_weight):
            return "Normal"
        elif final_weight < self.reward_calculator.safe_weight_min:
            return "Underflow"
        else:
            return "Overflow"
    
    def get_optimal_switch_point(self) -> int:
        """Get the switch point with the highest Q-value (optimal policy)."""
        return self._get_best_switch_point()
    
    def get_q_table(self) -> Dict[int, float]:
        """Get the current Q-table."""
        return self.q_table.copy()
    
    def get_training_statistics(self) -> Dict:
        """Get statistics about the training process."""
        if not self.training_history:
            return {}
        
        return {
            'total_episodes': len(self.training_history),
            'best_switch_point': self.get_optimal_switch_point(),
            'best_q_value': max(self.q_table.values())
        } 
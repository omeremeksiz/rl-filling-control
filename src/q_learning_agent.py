"""
Q-learning agent for optimal switching point learning.
Implements stateless reinforcement learning for container filling control.
"""

import numpy as np
import random
from typing import Dict, List, Optional, Tuple
from config import (
    DEFAULT_LEARNING_RATE, DEFAULT_EXPLORATION_RATE, DEFAULT_RANDOM_SEED, 
    EXPLORATION_STEPS, EXPLORATION_PROBABILITIES,
    DEFAULT_EXPLORATION_DECAY, DEFAULT_EXPLORATION_MIN_RATE,
    DEFAULT_EXPLORATION_DECAY_RATE, DEFAULT_EXPLORATION_DECAY_INTERVAL
)
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
                 random_seed: int = DEFAULT_RANDOM_SEED,
                 exploration_decay: bool = DEFAULT_EXPLORATION_DECAY,
                 exploration_min_rate: float = DEFAULT_EXPLORATION_MIN_RATE,
                 exploration_decay_rate: float = DEFAULT_EXPLORATION_DECAY_RATE,
                 exploration_decay_interval: int = DEFAULT_EXPLORATION_DECAY_INTERVAL):
        super().__init__(data_processor, reward_calculator, exploration_rate, random_seed, 
                         learning_rate, exploration_decay=exploration_decay,
                         exploration_min_rate=exploration_min_rate,
                         exploration_decay_rate=exploration_decay_rate,
                         exploration_decay_interval=exploration_decay_interval)
        self.learning_rate = learning_rate
        
        # Set random seed for reproducibility
        random.seed(random_seed)
        
        # Initialize Q-table
        self.q_table = self._initialize_q_table()
    
    def _initialize_q_table(self) -> Dict[int, float]:
        """Initialize Q-table with zero values for all available switch points."""
        return {switch_point: 0.0 for switch_point in self.available_switch_points}
    
    def _get_best_switch_point(self, current_switch_point: int = None) -> int:
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
        
        selected_session = random.choice(unused_sessions)
        # Mark this session as used
        self.data_processor.mark_session_as_used(current_switch_point, selected_session)
    
        # Simulate the episode
        episode_length = selected_session.episode_length
        final_weight = selected_session.final_weight
        
        # Calculate reward
        reward = self.reward_calculator.calculate_reward(episode_length, final_weight, method="mab")
        
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
        current_switch_point = initial_switch_point
        
        self.training_history = []
        
        for episode in range(num_episodes):
            # Update exploration rate using decay
            self.update_exploration_rate(episode)
            
            # Train on current episode
            reward, episode_length, final_weight = self.train_episode(current_switch_point)
            
            # Determine termination type
            termination_type = self._determine_termination_type(final_weight)
            
            # Get the best action (what model would select)
            best_switch_point = self._get_best_switch_point()
            
            # Select next action for next episode (may include exploration)
            next_switch_point, exploration_flag = self.select_action(current_switch_point)
            
            # Determine if exploration occurred by checking if the selected action
            # differs from the best action according to Q-values
            explored_switch_point = None
            if exploration_flag:
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
                'exploration_rate': self.exploration_rate
            }
            self.training_history.append(episode_stats)
            
            # Show progress to console (same format as log) - every 100 episodes + first episode
            if (episode + 1) % 100 == 0 or episode == 0:
                print(f"--- Episode {episode + 1}/{num_episodes} ---")
                print(f"Experienced Switching Point: {current_switch_point}")
                print(f"Termination Type: {termination_type}")
                print(f"Model-Selected Next Switching Point: {best_switch_point}")
                print(f"Explored Switching Point: {explored_switch_point}")
                print()
            
            # Log episode if logger is provided
            if logger:
                logger.log_episode(
                    episode_num=episode + 1,
                    total_episodes=num_episodes,
                    experienced_switch_point=current_switch_point,
                    termination_type=termination_type,
                    model_selected_next=best_switch_point,
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
    
    def _get_step_reward(self) -> float:
        """Q-learning doesn't use step rewards, only final episode rewards."""
        return 0.0
    
    def _update_q_values_from_episode(self, episode: List[Tuple[int, int, float]]) -> None:
        """Q-learning doesn't use episode trajectories, uses direct Q-value updates."""
        pass
    
    def _update_q_values(self, *args, **kwargs) -> None:
        """Q-learning uses _update_q_value method instead."""
        pass
    
    def get_training_statistics(self) -> Dict:
        """Get statistics about the training process."""
        if not self.training_history:
            return {}
        
        return {
            'total_episodes': len(self.training_history),
            'best_switch_point': self.get_optimal_switch_point(),
            'best_q_value': max(self.q_table.values())
        } 

    def select_action(self, current_switch_point: Optional[int] = None) -> int:
        """
        Q-learning specific action selection using only available switch points.
        
        Args:
            current_switch_point: Current switch point (for exploration)
            
        Returns:
            Selected switch point
        """
        exploration_flag = False
        if random.random() < self.exploration_rate:
            # Exploration: step-based exploration from current switch point
            exploration_flag = True
            return self._explore_with_steps(current_switch_point), exploration_flag
        else:
            # Exploitation: choose best action based on Q-values
            exploration_flag = False
            return self._get_best_switch_point(), exploration_flag
    
    def _explore_with_steps(self, current_switch_point: int) -> int:
        """
        Q-learning step-based exploration from current switching point.
        
        Args:
            current_switch_point: The current switching point to explore from
            
        Returns:
            Selected switch point for exploration
        """        
        # Get available switch points in ascending order
        available_points = sorted(self.available_switch_points)
        
        # Find the current switch point index in sorted available points
        current_index = available_points.index(current_switch_point)
                
        # Probabilistically select which exploration step to take
        # Probabilities must sum to 1.0 to guarantee selection
        random_value = random.random()
        cumulative_prob = 0.0
        
        for step, prob in zip(EXPLORATION_STEPS, EXPLORATION_PROBABILITIES):
            cumulative_prob += prob
            if random_value <= cumulative_prob:
                # Try to move +step from current switch point
                target_index = current_index + step
                if target_index < len(available_points):
                    result = available_points[target_index]
                    return result
                else:
                    # If step goes beyond available points, return the last available point
                    result = available_points[-1]
                    return result
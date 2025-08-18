"""
Temporal Difference (TD) Learning Agent for Filling Control System.

This agent implements true TD learning (SARSA-style) where Q-values are updated
after each step using the immediate reward and the Q-value of the actual next 
action taken (not the maximum). This makes it an on-policy learning algorithm.
"""

import random
from typing import Dict, Tuple, Any, Optional, List
from src.base_agent import BaseRLAgent
from src.data_processor import DataProcessor
from src.logger import TrainingLogger
from src.reward_calculator import RewardCalculator
from src.config import (
    DEFAULT_LEARNING_RATE,
    DEFAULT_EXPLORATION_RATE,
    DEFAULT_DISCOUNT_FACTOR,
    DEFAULT_INITIAL_Q_VALUE,
    DEFAULT_RANDOM_SEED,
    DEFAULT_EXPLORATION_DECAY,
    DEFAULT_EXPLORATION_MIN_RATE,
    DEFAULT_EXPLORATION_DECAY_RATE,
    DEFAULT_EXPLORATION_DECAY_INTERVAL
)


class TDAgent(BaseRLAgent):
    """Temporal Difference learning agent for filling control."""

    def __init__(self, data_processor: DataProcessor, reward_calculator: RewardCalculator,
                 learning_rate: float = DEFAULT_LEARNING_RATE,
                 exploration_rate: float = DEFAULT_EXPLORATION_RATE,
                 discount_factor: float = DEFAULT_DISCOUNT_FACTOR,
                 initial_q_value: float = DEFAULT_INITIAL_Q_VALUE,
                 random_seed: int = DEFAULT_RANDOM_SEED,
                 exploration_decay: bool = DEFAULT_EXPLORATION_DECAY,
                 exploration_min_rate: float = DEFAULT_EXPLORATION_MIN_RATE,
                 exploration_decay_rate: float = DEFAULT_EXPLORATION_DECAY_RATE,
                 exploration_decay_interval: int = DEFAULT_EXPLORATION_DECAY_INTERVAL):
        """
        Initialize TD agent.
        
        Args:
            data_processor: For accessing training data
            reward_calculator: For calculating rewards
            learning_rate: Learning rate (α)
            exploration_rate: Epsilon for epsilon-greedy exploration
            discount_factor: Discount factor (γ)
            initial_q_value: Initial Q-value for all state-action pairs
            random_seed: Random seed for reproducibility
            exploration_decay: Whether to use exploration decay
            exploration_min_rate: Minimum exploration rate
            exploration_decay_rate: Decay factor when decay occurs
            exploration_decay_interval: Decay every N episodes
        """
        super().__init__(data_processor, reward_calculator, exploration_rate, random_seed,
                         learning_rate, discount_factor, initial_q_value,
                         exploration_decay, exploration_min_rate, exploration_decay_rate, exploration_decay_interval)
        
        # Set random seed for reproducibility
        random.seed(random_seed)
        
        self.q_table = self._initialize_q_table()



    def _get_best_switch_point(self, current_switch_point: int = None) -> int:
        """Get the optimal switch point based on learned Q-values."""
        # Create policy from Q-values with tie-breaking rule: if equal, choose -1
        policy = self._create_policy_from_q_values_with_tie_breaking()
        
        # Use all available weights
        available_weights = sorted(self.available_weights)
        
        # Sort by weight value and find the first 1 to -1 transition (flipping)
        for weight in sorted(policy.keys()):
            if weight in available_weights and policy[weight] == -1:
                return weight
        
        # If there is no flipping (no -1 action found), continue with current switch point
        return current_switch_point

    def train_episode(self, current_switch_point: int) -> Tuple[float, int, int]:
        """
        Train on a single episode using the given switch point.
        
        Args:
            current_switch_point: Switch point to use for this episode
            
        Returns:
            Tuple of (reward, episode_length, final_weight)
        """
        # Use the common base class implementation
        episode_length, final_weight = super().train_episode(current_switch_point)
        
        # Return with placeholder reward (not needed for TD)
        return episode_length, final_weight
    
    def _get_step_reward(self) -> float:
        """TD learning uses time penalty per step."""
        return 0.0
    
    def _update_q_values_from_episode(self, episode: List[Tuple[int, int, float]]) -> None:
        """Update Q-values using TD learning method."""
        self._update_q_values(episode)

    def _update_q_values(self, trajectory: List[Tuple[int, int, float]]) -> None:
        """
        Update Q-values using TD learning update rule.
        
        Args:
            trajectory: List of (state, action, reward) tuples
        """
        # TD learning: update Q-values after each step
        for i in range(len(trajectory) - 1):
            current_state, current_action, current_reward = trajectory[i]
            next_state, next_action, next_reward = trajectory[i + 1]
            
            # Get current Q-value
            current_q = self.q_table.get((current_state, current_action), self.initial_q_value)
            
            # Get Q-value for the actual next action taken (true TD learning)
            next_q = self.q_table.get((next_state, next_action), self.initial_q_value)
            
            # True TD update: Q(s,a) = Q(s,a) + α[r + γ*Q(s',a') - Q(s,a)]
            # Uses actual next action taken, not max (SARSA-style)
            td_target = current_reward + self.discount_factor * next_q
            self.q_table[(current_state, current_action)] = current_q + self.learning_rate * (td_target - current_q)
        
        # Handle the last step (terminal state)
        if trajectory:
            last_state, last_action, last_reward = trajectory[-1]
            
            current_q = self.q_table.get((last_state, last_action), self.initial_q_value)
            
            # For terminal state, next Q-value is 0
            td_target = last_reward  # No next state
            self.q_table[(last_state, last_action)] = current_q + self.learning_rate * (td_target - current_q)

    def train(self, num_episodes: int, initial_switch_point: Optional[int] = None, logger: Optional[TrainingLogger] = None) -> List[Dict[str, Any]]:
        """
        Train the agent for the specified number of episodes.
        
        Args:
            num_episodes: Number of training episodes
            initial_switch_point: Starting switch point (random if None)
            logger: Logger for tracking training progress
            
        Returns:
            Training history
        """
        return super().train(num_episodes, initial_switch_point, logger) 
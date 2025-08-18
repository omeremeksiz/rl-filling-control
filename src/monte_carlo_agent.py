"""
Monte Carlo agent for optimal switching point learning.
Implements Monte Carlo method for container filling control.
"""

import numpy as np
import random
from typing import Dict, List, Optional, Tuple
from config import (
    DEFAULT_LEARNING_RATE, DEFAULT_EXPLORATION_RATE, DEFAULT_RANDOM_SEED, 
    DEFAULT_DISCOUNT_FACTOR, DEFAULT_INITIAL_Q_VALUE,
    DEFAULT_EXPLORATION_DECAY, DEFAULT_EXPLORATION_MIN_RATE,
    DEFAULT_EXPLORATION_DECAY_RATE, DEFAULT_EXPLORATION_DECAY_INTERVAL
)
from data_processor import DataProcessor, FillingSession
from reward_calculator import RewardCalculator
from base_agent import BaseRLAgent


class MonteCarloAgent(BaseRLAgent):
    """Monte Carlo agent for learning optimal switching points."""
    
    def __init__(self, 
                 data_processor: DataProcessor,
                 reward_calculator: RewardCalculator,
                 learning_rate: float = DEFAULT_LEARNING_RATE,
                 exploration_rate: float = DEFAULT_EXPLORATION_RATE,
                 discount_factor: float = DEFAULT_DISCOUNT_FACTOR,
                 initial_q_value: float = DEFAULT_INITIAL_Q_VALUE,
                 random_seed: int = DEFAULT_RANDOM_SEED,
                 exploration_decay: bool = DEFAULT_EXPLORATION_DECAY,
                 exploration_min_rate: float = DEFAULT_EXPLORATION_MIN_RATE,
                 exploration_decay_rate: float = DEFAULT_EXPLORATION_DECAY_RATE,
                 exploration_decay_interval: int = DEFAULT_EXPLORATION_DECAY_INTERVAL):
        super().__init__(data_processor, reward_calculator, exploration_rate, random_seed, 
                         learning_rate, discount_factor, initial_q_value,
                         exploration_decay, exploration_min_rate, exploration_decay_rate, exploration_decay_interval)
        
        # Set random seed for reproducibility
        random.seed(random_seed)
        
        # Initialize Q-table for state-action pairs
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
        
        # Return with placeholder reward (not needed for Monte Carlo)
        return episode_length, final_weight
    
    def _get_step_reward(self) -> float:
        """Monte Carlo uses no step penalty, only final outcome matters."""
        return -1.0
    
    def _update_q_values_from_episode(self, episode: List[Tuple[int, int, float]]) -> None:
        """Update Q-values using Monte Carlo method."""
        # Calculate G-values for the episode (now includes final reward)
        g_values = self._calculate_g_values(episode)
        
        # Update Q-values using Monte Carlo update
        self._update_q_values(episode, g_values)
    
    def _calculate_g_values(self, episode: List[Tuple[int, int, float]]) -> List[float]:
        """
        Calculate G-values for each step in the episode.
        
        Args:
            episode: List of (state, action, reward) tuples
            
        Returns:
            List of G-values for each step
        """
        g_values = []
        
        for t in range(len(episode)):
            # Calculate discounted return from time t to end
            g_value = 0
            for i in range(t, len(episode)):
                reward = episode[i][2]  # reward
                discount = self.discount_factor ** (i - t)
                g_value += discount * reward
            
            g_values.append(g_value)
        
        return g_values
    
    def _update_q_values(self, episode: List[Tuple[int, int, float]], g_values: List[float]) -> None:
        """
        Update Q-values using Monte Carlo update rule.
        
        Args:
            episode: List of (state, action, reward) tuples
            g_values: List of G-values for each step
        """
        for t, (state, action, _) in enumerate(episode):
            g_value = g_values[t]
            current_q = self.q_table.get((state, action), 0.0)
            
            # Check if we can update this state-action pair
            can_update = True
            
            # For slow actions (-1), only update if the state has been updated with fast action (1) at least once
            if action == -1 and state not in self.states_with_fast_action_updated:
                can_update = False
            
            if can_update:
                # Monte Carlo update: Q(s,a) = Q(s,a) + α(G - Q(s,a))
                new_q = current_q + self.learning_rate * (g_value - current_q)
                self.q_table[(state, action)] = new_q
                
                # Track that this state has been updated with action 1
                if action == 1:
                    self.states_with_fast_action_updated.add(state)
    
    def train(self, num_episodes: int, initial_switch_point: Optional[int] = None, logger=None) -> List[Dict]:
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
    
 
"""
Abstract base class for all reinforcement learning agents.
Defines the common interface that all RL methods must implement.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
import random
from data_processor import DataProcessor
from reward_calculator import RewardCalculator
from config import EXPLORATION_STEPS, EXPLORATION_PROBABILITIES


class BaseRLAgent(ABC):
    """Abstract base class for all RL agents."""
    
    def __init__(self, 
                 data_processor: DataProcessor,
                 reward_calculator: RewardCalculator,
                 exploration_rate: float = 0.5,
                 random_seed: int = 42):
        """
        Initialize the base agent.
        
        Args:
            data_processor: Data processor for loading and managing data
            reward_calculator: Reward calculator for computing rewards
            exploration_rate: Rate of exploration vs exploitation
            random_seed: Random seed for reproducibility
        """
        self.data_processor = data_processor
        self.reward_calculator = reward_calculator
        self.exploration_rate = exploration_rate
        self.random_seed = random_seed
        
        # Get available switch points for action selection
        self.available_switch_points = data_processor.get_available_switch_points()
        
        # Training statistics
        self.training_history = []
    

    
    @abstractmethod
    def train_episode(self, current_switch_point: int) -> Tuple[float, int, int]:
        """
        Train on a single episode using the given switch point.
        
        Args:
            current_switch_point: Switch point to use for this episode
            
        Returns:
            Tuple of (reward, episode_length, final_weight)
        """
        pass
    
    @abstractmethod
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
        pass
    
    @abstractmethod
    def get_optimal_switch_point(self) -> int:
        """
        Get the learned optimal switch point.
        
        Returns:
            Optimal switch point
        """
        pass
    
    @abstractmethod
    def get_q_table(self) -> Dict:
        """
        Get the agent's value function (Q-table or equivalent).
        
        Returns:
            Dictionary containing the value function
        """
        pass
    
    def get_training_statistics(self) -> Dict:
        """
        Get training statistics.
        
        Returns:
            Dictionary containing training statistics
        """
        return {
            'training_history': self.training_history,
            'method': self.__class__.__name__
        }
    
    def select_action(self, current_switch_point: Optional[int] = None) -> int:
        """
        Common action selection logic using epsilon-greedy policy with step-based exploration.
        
        Args:
            current_switch_point: Current switch point (for exploration)
            
        Returns:
            Selected switch point
        """
        if random.random() < self.exploration_rate:
            # Exploration: step-based exploration
            return self._explore_with_steps()
        else:
            # Exploitation: choose best action with bounds checking
            best_switch_point = self._get_best_switch_point()
            
            # Bounds checking: ensure best_switch_point is within available range
            if best_switch_point > max(self.available_switch_points):
                return max(self.available_switch_points)
            elif best_switch_point < min(self.available_switch_points):
                return min(self.available_switch_points)
            elif best_switch_point not in self.available_switch_points:
                # If not in list but within bounds, find nearest available point
                return min(self.available_switch_points, key=lambda x: abs(x - best_switch_point))
            
            return best_switch_point
    
    def _explore_with_steps(self) -> int:
        """
        Explore using step-based exploration from best action.
        Only explores in positive direction (higher switch points).
        Guarantees exploration occurs by probabilistically selecting step size.
        
        Returns:
            Selected switch point for exploration
        """
        best_switch_point = self._get_best_switch_point()
        
        # Get available switch points in ascending order
        available_points = sorted(self.available_switch_points)
        
        # Bounds checking: ensure best_switch_point is within available range
        if best_switch_point > max(available_points):
            best_switch_point = max(available_points)
        elif best_switch_point < min(available_points):
            best_switch_point = min(available_points)
        elif best_switch_point not in available_points:
            # If not in list but within bounds, find nearest available point
            best_switch_point = min(available_points, key=lambda x: abs(x - best_switch_point))
        
        # Find the best action index in sorted available points
        best_index = available_points.index(best_switch_point)
        
        # Probabilistically select which exploration step to take
        # Probabilities must sum to 1.0 to guarantee selection
        random_value = random.random()
        cumulative_prob = 0.0
        
        for step, prob in zip(EXPLORATION_STEPS, EXPLORATION_PROBABILITIES):
            cumulative_prob += prob
            if random_value <= cumulative_prob:
                # Try to move +step from best action (only positive direction)
                target_index = best_index + step
                if target_index < len(available_points):
                    return available_points[target_index]
                else:
                    # If step goes beyond available points, return the last available point
                    return available_points[-1]
        
        # This should never happen if probabilities sum to 1.0
        # But as a safety fallback, take the smallest step
        target_index = best_index + EXPLORATION_STEPS[0]
        if target_index < len(available_points):
            return available_points[target_index]
        else:
            return available_points[-1]
    
    @abstractmethod
    def _get_best_switch_point(self) -> int:
        """
        Get the best switch point based on the agent's learned policy.
        This method should be implemented by each agent.
        
        Returns:
            Best switch point
        """
        pass 
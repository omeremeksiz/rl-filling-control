"""
Reward calculation module for the container filling control system.
Handles computation of rewards based on episode length and safety constraints.
"""

from typing import Tuple
from config import DEFAULT_SAFE_WEIGHT_MIN, DEFAULT_SAFE_WEIGHT_MAX, DEFAULT_PENALTY_MULTIPLIER


class RewardCalculator:
    """Calculates rewards for filling episodes based on length and safety constraints."""
    
    def __init__(self, 
                 safe_weight_min: int = DEFAULT_SAFE_WEIGHT_MIN,
                 safe_weight_max: int = DEFAULT_SAFE_WEIGHT_MAX,
                 penalty_multiplier: float = DEFAULT_PENALTY_MULTIPLIER):
        self.safe_weight_min = safe_weight_min
        self.safe_weight_max = safe_weight_max
        self.penalty_multiplier = penalty_multiplier
    
    def calculate_reward(self, episode_length: int, final_weight: int) -> float:
        """
        Calculate reward for a filling episode.
        
        Args:
            episode_length: Number of steps until termination
            final_weight: Final weight achieved
            
        Returns:
            Reward value: -length + β × penalty
        """
        penalty = self._calculate_penalty(final_weight)
        return -episode_length + self.penalty_multiplier * penalty
    
    def _calculate_penalty(self, final_weight: int) -> float:
        """
        Calculate penalty based on whether final weight is in safe range.
        
        Args:
            final_weight: The final weight achieved
            
        Returns:
            Penalty value: 0 if safe, otherwise overflow/underflow amount
        """
        if self._is_weight_safe(final_weight):
            return 0.0
        
        if final_weight < self.safe_weight_min:
            return self.safe_weight_min - final_weight  # Underflow penalty
        else:
            return final_weight - self.safe_weight_max  # Overflow penalty
    
    def _is_weight_safe(self, final_weight: int) -> bool:
        """Check if the final weight is within the safe range."""
        return self.safe_weight_min <= final_weight <= self.safe_weight_max
    
    def get_safe_range(self) -> Tuple[int, int]:
        """Get the safe weight range."""
        return (self.safe_weight_min, self.safe_weight_max)
    
    def set_safe_range(self, min_weight: int, max_weight: int) -> None:
        """Update the safe weight range."""
        if min_weight >= max_weight:
            raise ValueError("Safe weight minimum must be less than maximum")
        self.safe_weight_min = min_weight
        self.safe_weight_max = max_weight 
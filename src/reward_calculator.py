"""
Reward calculation module for the container filling control system.
Handles computation of rewards based on episode length and safety constraints.
"""

from typing import Tuple
from config import (
    DEFAULT_SAFE_WEIGHT_MIN, 
    DEFAULT_SAFE_WEIGHT_MAX, 
    DEFAULT_OVERFLOW_PENALTY_CONSTANT,
    DEFAULT_UNDERFLOW_PENALTY_CONSTANT
)


class RewardCalculator:
    """Calculates rewards for filling episodes based on final weight and penalty constants."""
    
    def __init__(self, 
                 safe_weight_min: int = DEFAULT_SAFE_WEIGHT_MIN,
                 safe_weight_max: int = DEFAULT_SAFE_WEIGHT_MAX,
                 overflow_penalty_constant: float = DEFAULT_OVERFLOW_PENALTY_CONSTANT,
                 underflow_penalty_constant: float = DEFAULT_UNDERFLOW_PENALTY_CONSTANT):
        self.safe_weight_min = safe_weight_min
        self.safe_weight_max = safe_weight_max
        self.overflow_penalty_constant = overflow_penalty_constant
        self.underflow_penalty_constant = underflow_penalty_constant
    
    def calculate_reward(self, episode_length: int, final_weight: int, method: str = "standard") -> float:
        """
        Calculate reward for a filling episode.
        
        Args:
            episode_length: Number of steps until termination
            final_weight: Final weight achieved
            method: RL method ("mab" for Multi-Armed Bandit, "standard" for others)
            
        Returns:
            Reward value:
            For MAB:
              - Safe: -length
              - Overflow: -length + overflow_penalty * (final_weight - safe_min)
              - Underflow: -length + (safe_min - final_weight) * underflow_penalty
            For other methods:
              - Safe: -1 + 0 = -1
              - Overflow: -1 + (final_weight - safe_min) * overflow_penalty_constant  
              - Underflow: -1 + (safe_min - final_weight) * underflow_penalty_constant
        """
        if method.lower() == "mab":
            # MAB-specific reward calculation
            base_reward = -episode_length
            
            if self._is_weight_safe(final_weight):
                penalty = 0.0
            elif final_weight > self.safe_weight_max:  # Overflow
                overflow_amount = final_weight - self.safe_weight_max
                penalty = overflow_amount * self.overflow_penalty_constant
            else:  # Underflow
                underflow_amount = self.safe_weight_min - final_weight
                penalty = underflow_amount * self.underflow_penalty_constant
            
            return base_reward + penalty
        else:
            # Standard reward calculation for other methods
            base_reward = 0.0
            
            if self._is_weight_safe(final_weight):
                penalty = 0.0
            elif final_weight > self.safe_weight_max:  # Overflow
                overflow_amount = final_weight - self.safe_weight_max
                penalty = overflow_amount * self.overflow_penalty_constant
            else:  # Underflow
                underflow_amount = self.safe_weight_min - final_weight
                penalty = underflow_amount * self.underflow_penalty_constant
            
            return base_reward + penalty
    
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
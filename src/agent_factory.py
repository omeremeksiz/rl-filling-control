"""
Factory class for creating different RL agents.
"""

from typing import Dict, Any
from base_agent import BaseRLAgent
from q_learning_agent import QLearningAgent
from monte_carlo_agent import MonteCarloAgent
from td_agent import TDAgent
from qlearning_standard_agent import StandardQLearningAgent


class AgentFactory:
    """Factory for creating RL agents."""
    
    @staticmethod
    def create_agent(method: str, **kwargs) -> BaseRLAgent:
        """
        Create an RL agent based on the specified method.
        
        Args:
            method: RL method name ("mab", "mc", etc.)
            **kwargs: Agent-specific parameters
            
        Returns:
            Configured RL agent
        """
        if method.lower() == "mab":
            return QLearningAgent(**kwargs)
        elif method.lower() == "mc":
            return MonteCarloAgent(**kwargs)
        elif method.lower() == "td":
            return TDAgent(**kwargs)
        elif method.lower() == "qlearning" or method.lower() == "q":
            return StandardQLearningAgent(**kwargs)
        else:
            raise ValueError(f"Unknown RL method: {method}")
    
    @staticmethod
    def get_available_methods() -> list:
        """Get list of available RL methods."""
        return ["mab", "mc", "td", "qlearning", "q"] 
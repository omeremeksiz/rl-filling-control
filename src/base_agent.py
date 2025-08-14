"""
Abstract base class for all reinforcement learning agents.
Defines the common interface that all RL methods must implement.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
import random
from data_processor import DataProcessor, FillingSession
from reward_calculator import RewardCalculator
from config import EXPLORATION_STEPS, EXPLORATION_PROBABILITIES


class BaseRLAgent(ABC):
    """Abstract base class for all RL agents."""
    
    def __init__(self, 
                 data_processor: DataProcessor,
                 reward_calculator: RewardCalculator,
                 exploration_rate: float = 0.5,
                 random_seed: int = 42,
                 learning_rate: float = 0.1,
                 discount_factor: float = 0.99,
                 initial_q_value: float = -125.0):
        """
        Initialize the base agent.
        
        Args:
            data_processor: Data processor for loading and managing data
            reward_calculator: Reward calculator for computing rewards
            exploration_rate: Rate of exploration vs exploitation
            random_seed: Random seed for reproducibility
            learning_rate: Learning rate for Q-value updates
            discount_factor: Discount factor for future rewards
            initial_q_value: Initial Q-value for state-action pairs
        """
        self.data_processor = data_processor
        self.reward_calculator = reward_calculator
        self.exploration_rate = exploration_rate
        self.random_seed = random_seed
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.initial_q_value = initial_q_value
        
        # Get available switch points for action selection
        self.available_switch_points = data_processor.get_available_switch_points()
        self.available_weights = data_processor.get_all_available_weights()
        
        # Training statistics
        self.training_history = []
        
        # Q-table will be initialized by subclasses
        self.q_table = {}
        self.states_with_fast_action_updated = set()
    

    
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
        exploration_flag = False
        if random.random() < self.exploration_rate:
            # Exploration: step-based exploration
            exploration_flag = True
            return self._explore_with_steps(), exploration_flag
        else:
            # Exploitation: choose best action with bounds checking
            exploration_flag = False
            best_switch_point = self._get_best_switch_point()
            
            # Check if best switch point is in available points
            if best_switch_point not in self.available_switch_points:
                # If not in list find nearest available point
                best_switch_point = min(self.available_switch_points, key=lambda x: abs(x - best_switch_point))
            
            return best_switch_point, exploration_flag
    
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
        
        # Check if best switch point is in available points
        if best_switch_point not in available_points:
            # If not in list find nearest available point
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
    
    @abstractmethod
    def _get_best_switch_point(self) -> int:
        """
        Get the best switch point based on the agent's learned policy.
        This method should be implemented by each agent.
        
        Returns:
            Best switch point
        """
        pass
    
    @abstractmethod 
    def _update_q_values(self, *args, **kwargs) -> None:
        """
        Update Q-values using the specific learning algorithm.
        This method should be implemented by each agent.
        """
        pass
    
    def _initialize_q_table(self) -> Dict[Tuple[int, int], float]:
        """Initialize Q-table for state-action pairs."""
        q_table = {}
        # Get only filling process weights (exclude final weights and indicators)
        filling_weights = set()
        for session in self.data_processor.sessions:
            # Find termination index
            try:
                termination_index = session.weight_sequence.index(300)
            except ValueError:
                termination_index = len(session.weight_sequence)
            
            # Process only weights during filling process (before final weight)
            for i in range(termination_index):
                weight = session.weight_sequence[i]
                
                # Skip indicators (-1 and 300)
                if weight == -1 or weight == 300:
                    continue
                
                # Skip final weight - only include filling process weights
                if i == termination_index - 1:  # This is the final weight position
                    continue
                    
                filling_weights.add(weight)
        
        # Initialize Q-values for filling process weight state-action pairs only
        for weight in filling_weights:
            q_table[(weight, 1)] = self.initial_q_value   # Fast action
            q_table[(weight, -1)] = self.initial_q_value  # Slow action
        
        # Track which states have been updated with action 1
        self.states_with_fast_action_updated = set()
        
        return q_table
    
    def _create_policy_from_q_values(self) -> Dict[int, int]:
        """Create policy from Q-values by selecting best action for each state."""
        policy = {}
        
        # Group Q-values by state (weight)
        state_actions = {}
        for (weight, action), q_value in self.q_table.items():
            if weight not in state_actions:
                state_actions[weight] = {}
            state_actions[weight][action] = q_value
        
        # Select best action for each state
        for weight, actions in state_actions.items():
            best_action = max(actions, key=actions.get)
            policy[weight] = best_action
        
        return policy
    
    def _create_policy_from_q_values_with_tie_breaking(self) -> Dict[int, int]:
        """
        Create policy from Q-values with tie-breaking rule:
        - Compare Q-values of each state for both actions (1 and -1)
        - If Q-values are equal, select action -1 (slow)
        - Otherwise, select the action with higher Q-value
        """
        policy = {}
        
        # Group Q-values by state (weight)
        state_actions = {}
        for (weight, action), q_value in self.q_table.items():
            if weight not in state_actions:
                state_actions[weight] = {}
            state_actions[weight][action] = q_value
        
        # Select best action for each state with tie-breaking
        for weight, actions in state_actions.items():
            if len(actions) == 2:  # Both actions available
                q_fast = actions.get(1, float('-inf'))  # Fast action
                q_slow = actions.get(-1, float('-inf'))  # Slow action
                
                # If equal Q-values or slow is better, choose slow (-1)
                # If fast is strictly better, choose fast (1)
                if q_slow >= q_fast:
                    policy[weight] = -1
                else:
                    policy[weight] = 1
            else:
                # Only one action available, choose it
                best_action = max(actions, key=actions.get)
                policy[weight] = best_action
        
        return policy
    
    def train_episode(self, current_switch_point: int) -> Tuple[int, int]:
        """
        Common episode training logic shared across Monte Carlo, TD, and Standard Q-Learning.
        This is the base implementation that can be called by subclasses.
        
        Args:
            current_switch_point: Switch point to use for this episode
            
        Returns:
            Tuple of (episode_length, final_weight)
        """
        # Get unused sessions from the cluster of current switch point
        unused_sessions = self.data_processor.get_unused_sessions_for_switch_point(current_switch_point)
        
        # Select a random session from the unused sessions
        selected_session = random.choice(unused_sessions)
        
        # Create episode trajectory
        episode = self._create_episode_trajectory(selected_session, current_switch_point)
        
        # Get final weight and calculate final episode reward using reward calculator
        final_weight = selected_session.final_weight if selected_session.final_weight is not None else 0
        episode_length = selected_session.episode_length
        final_episode_reward = self.reward_calculator.calculate_reward(episode_length, final_weight)
        
        # Add final episode reward to the last step in trajectory
        if episode:
            # Update the last step with the final episode reward
            last_state, last_action, last_step_reward = episode[-1]
            episode[-1] = (last_state, last_action, last_step_reward + final_episode_reward)
        
        # Delegate to subclass-specific Q-value update method
        self._update_q_values_from_episode(episode)
        
        return episode_length, final_weight
    
    @abstractmethod
    def _update_q_values_from_episode(self, episode: List[Tuple[int, int, float]]) -> None:
        """
        Update Q-values from episode trajectory using the specific learning algorithm.
        This method should be implemented by each agent.
        
        Args:
            episode: List of (state, action, reward) tuples
        """
        pass
    
    def _create_episode_trajectory(self, session: FillingSession, switch_point: int) -> List[Tuple[int, int, float]]:
        """
        Create episode trajectory with state-action-reward tuples.
        Only includes weights during the filling process, NOT the final weight.
        
        Args:
            session: Filling session data
            switch_point: Switch point for this episode
            
        Returns:
            List of (state, action, reward) tuples
        """
        trajectory = []
        
        # Find the termination index to exclude final weight
        try:
            termination_index = session.weight_sequence.index(300)
        except ValueError:
            termination_index = len(session.weight_sequence)
        
        # Process only weights during filling process (before final weight)
        for i in range(termination_index):
            weight = session.weight_sequence[i]
            
            # Skip indicators (-1 and 300) - only process actual weights
            if weight == -1 or weight == 300:
                continue
            
            # Skip final weight - only process filling process weights
            if i == termination_index - 1:  # This is the final weight position
                continue
                
            # Determine action based on switch point
            if weight < switch_point:
                action = 1  # Fast action before switch point
            else:
                action = -1  # Slow action after switch point
            
            # Step reward varies by agent - this is the base implementation
            step_reward = self._get_step_reward()
            
            trajectory.append((weight, action, step_reward))
        
        return trajectory
    
    @abstractmethod
    def _get_step_reward(self) -> float:
        """
        Get the step reward for the specific agent type.
        This method should be implemented by each agent.
        
        Returns:
            Step reward value
        """
        pass
    
    def train(self, num_episodes: int, initial_switch_point: Optional[int] = None, logger=None) -> List[Dict]:
        """
        Common training logic shared across Monte Carlo, TD, and Standard Q-Learning.
        
        Args:
            num_episodes: Number of training episodes
            initial_switch_point: Starting switch point (random if None)
            logger: Logger for tracking training progress
            
        Returns:
            Training history
        """
        current_switch_point = initial_switch_point
        
        for episode in range(num_episodes):
            # Train on one episode using common logic
            episode_length, final_weight = self.train_episode(current_switch_point)
            
            # Determine termination type
            termination_type = self._determine_termination_type(final_weight)
            
            # Get what model would select for next episode (without exploration)
            model_selected_next_switch_point = self._get_best_switch_point()
            
            # Select next action for next episode (may include exploration)
            next_switch_point, exploration_flag = self.select_action(current_switch_point)
            
            # Determine if exploration occurred
            explored_switch_point = None
            if exploration_flag:
                explored_switch_point = next_switch_point
            
            # Log episode results
            episode_data = {
                'episode': episode + 1,
                'episode_num': episode + 1,  # For compatibility with desired plot format
                'switch_point': current_switch_point,
                'model_selected_switching_point': model_selected_next_switch_point,
                'explored_switching_point': explored_switch_point,
                'episode_length': episode_length,
                'final_weight': final_weight,
                'termination_type': termination_type
            }
            
            self.training_history.append(episode_data)
            
            # Log progress
            if logger:
                logger.log_episode(episode + 1, num_episodes, current_switch_point, termination_type, model_selected_next_switch_point, explored_switch_point)
            
            # Update current switch point
            current_switch_point = next_switch_point
        
        return self.training_history
    
    def _determine_termination_type(self, final_weight: int) -> str:
        """Determine the type of episode termination."""
        if final_weight < self.reward_calculator.safe_weight_min:
            return "underweight"
        elif final_weight > self.reward_calculator.safe_weight_max:
            return "overweight"
        else:
            return "safe"
    
    def get_optimal_switch_point(self) -> int:
        """Get the learned optimal switch point."""
        return self._get_best_switch_point()
    
    def get_q_table(self) -> Dict:
        """Get the Q-table."""
        return self.q_table 
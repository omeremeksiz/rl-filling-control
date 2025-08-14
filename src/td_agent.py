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


class TDAgent(BaseRLAgent):
    """Temporal Difference learning agent for filling control."""

    def __init__(self, data_processor: DataProcessor, reward_calculator: RewardCalculator,
                 learning_rate: float = 0.1, exploration_rate: float = 0.5,
                 discount_factor: float = 0.99, initial_q_value: float = -125.0,
                 random_seed: int = 42):
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
        """
        super().__init__(data_processor, reward_calculator, exploration_rate, random_seed)
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.initial_q_value = initial_q_value
        self.q_table: Dict[Tuple[int, int], float] = {}
        self._initialize_q_table()

    def _initialize_q_table(self):
        """Initialize Q-table with initial values for all state-action pairs."""
        filling_weights = set()
        
        # Collect all unique weights from training data (excluding indicators and final weights)
        for session in self.data_processor.sessions:
            # Find termination index
            try:
                termination_index = session.weight_sequence.index(300)
            except ValueError:
                termination_index = len(session.weight_sequence)
            
            # Process only weights during filling process (before final weight)
            for i in range(termination_index):
                weight = session.weight_sequence[i]
                if weight == -1 or weight == 300:
                    continue
                if i == termination_index - 1:  # Skip final weight position
                    continue
                filling_weights.add(weight)

        # Initialize Q-values for all state-action pairs
        for weight in filling_weights:
            for action in [1, -1]:  # Fast (1) and slow (-1) actions
                self.q_table[(weight, action)] = self.initial_q_value

    def _get_best_switch_point(self) -> int:
        """Get the optimal switch point based on learned Q-values with improved logic."""
        # Create policy from Q-values with tie-breaking rule: if equal, choose -1
        policy = self._create_policy_from_q_values()
        
        # Sort states by weight value and find first 1 to -1 transition
        sorted_states = sorted(policy.keys())
        
        for i in range(len(sorted_states) - 1):
            current_state = sorted_states[i]
            next_state = sorted_states[i + 1]
            
            # Check if transition from fast (1) to slow (-1) occurs
            if policy[current_state] == 1 and policy[next_state] == -1:
                return next_state  # Return the first -1 state (switching point)
        
        # If no transition found, check if the last state is -1
        if sorted_states and policy[sorted_states[-1]] == -1:
            return sorted_states[-1]
        
        # If no -1 action found at all, return maximum available switching point
        return max(self.available_switch_points)

    def _create_policy_from_q_values(self) -> Dict[int, int]:
        """Create policy from Q-values by selecting best action for each state."""
        policy = {}
        
        # Group Q-values by state (weight)
        state_actions = {}
        for (weight, action), q_value in self.q_table.items():
            if weight not in state_actions:
                state_actions[weight] = {}
            state_actions[weight][action] = q_value
        
        # Select best action for each state (tie-breaking: prefer -1 if equal)
        for weight, actions in state_actions.items():
            if len(actions) == 2:  # Both actions available
                if abs(actions[1] - actions[-1]) < 1e-6:  # Q-values are equal
                    policy[weight] = -1  # Choose slow action (-1) in case of tie
                else:
                    policy[weight] = max(actions, key=actions.get)
            else:
                # Only one action available
                policy[weight] = max(actions, key=actions.get)
        
        return policy

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
        
        # Create episode trajectory
        trajectory = self._create_episode_trajectory(selected_session, current_switch_point)
        
        # TD learning: update Q-values after each step
        self._update_q_values(trajectory)
        
        # Return episode statistics
        final_weight = selected_session.final_weight if selected_session.final_weight is not None else 0
        episode_length = selected_session.episode_length
        final_episode_reward = self.reward_calculator.calculate_reward(episode_length, final_weight)
        
        return final_episode_reward, episode_length, final_weight

    def _create_episode_trajectory(self, session, switch_point: int) -> List[Tuple[int, int, float]]:
        """Create episode trajectory from session data."""
        trajectory = []
        
        # Find termination index
        try:
            termination_index = session.weight_sequence.index(300)
        except ValueError:
            termination_index = len(session.weight_sequence)
        
        # Create trajectory (excluding indicators and final weight)
        for i in range(termination_index):
            weight = session.weight_sequence[i]
            if weight == -1 or weight == 300:
                continue
            if i == termination_index - 1:  # Skip final weight position
                continue
            
            # Determine action based on switch point
            action = 1 if weight < switch_point else -1
            step_reward = -1.0  # Time penalty per step
            trajectory.append((weight, action, step_reward))
        
        # Add final episode reward to the last step
        if trajectory:
            final_weight = session.final_weight if session.final_weight is not None else 0
            episode_length = session.episode_length
            final_episode_reward = self.reward_calculator.calculate_reward(episode_length, final_weight)
            
            last_state, last_action, last_step_reward = trajectory[-1]
            trajectory[-1] = (last_state, last_action, last_step_reward + final_episode_reward)
        
        return trajectory

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
        """Train the TD agent for specified number of episodes."""
        training_history = []
        current_switch_point = initial_switch_point if initial_switch_point is not None else self.available_switch_points[0]
        
        for episode in range(num_episodes):
            # Get model's next selection (without exploration) for logging
            model_selected_next_switch_point = self._get_best_switch_point()
            
            # Train on one episode
            reward, episode_length, final_weight = self.train_episode(current_switch_point)
            
            # Determine termination type
            termination_type = self._determine_termination_type(final_weight)
            
            # Select next action for next episode (may include exploration)
            next_switch_point = self.select_action(current_switch_point)
            
            # Determine if exploration occurred
            explored_switch_point = None
            if next_switch_point != model_selected_next_switch_point:
                explored_switch_point = next_switch_point

            # Log episode results
            episode_data = {
                'episode': episode + 1,
                'episode_num': episode + 1,  # For compatibility with desired plot format
                'switch_point': current_switch_point,
                'model_selected_switching_point': model_selected_next_switch_point,
                'explored_switching_point': explored_switch_point,
                'reward': reward,
                'episode_length': episode_length,
                'final_weight': final_weight,
                'termination_type': termination_type
            }
            training_history.append(episode_data)

            # Log progress
            if logger:
                logger.log_episode(episode + 1, num_episodes, current_switch_point, termination_type, model_selected_next_switch_point, explored_switch_point)

            # Update current switch point for next episode
            current_switch_point = next_switch_point

        return training_history

    def get_optimal_switch_point(self) -> int:
        """Get the current optimal switch point."""
        return self._get_best_switch_point()

    def get_q_table(self) -> Dict[Tuple[int, int], float]:
        """Get the agent's Q-table."""
        return self.q_table

    def _determine_termination_type(self, final_weight: int) -> str:
        """Determine if episode ended in safe, overflow, or underflow."""
        if self.reward_calculator._is_weight_safe(final_weight):
            return "safe"
        elif final_weight > self.reward_calculator.safe_weight_max:
            return "overflow"
        else:
            return "underflow"
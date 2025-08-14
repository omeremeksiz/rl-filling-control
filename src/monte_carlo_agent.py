"""
Monte Carlo agent for optimal switching point learning.
Implements Monte Carlo method for container filling control.
"""

import numpy as np
import random
from typing import Dict, List, Optional, Tuple
from config import DEFAULT_LEARNING_RATE, DEFAULT_EXPLORATION_RATE, DEFAULT_RANDOM_SEED, DEFAULT_DISCOUNT_FACTOR, DEFAULT_MC_INITIAL_Q_VALUE
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
                 initial_q_value: float = DEFAULT_MC_INITIAL_Q_VALUE,
                 random_seed: int = DEFAULT_RANDOM_SEED):
        super().__init__(data_processor, reward_calculator, exploration_rate, random_seed)
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.initial_q_value = initial_q_value
        
        # Set random seed for reproducibility
        random.seed(random_seed)
        
        # Initialize Q-table for state-action pairs
        self.available_weights = data_processor.get_all_available_weights()
        self.q_table = self._initialize_q_table()
    
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
    
    def _get_best_switch_point(self) -> int:
        """Get the optimal switch point based on learned Q-values."""
        # Create policy from Q-values with tie-breaking rule: if equal, choose -1
        policy = self._create_policy_from_q_values_with_tie_breaking()
        
        # Use all available weights
        available_weights = sorted(self.available_weights)
        
        # Sort by weight value and find the first 1 to -1 transition (flipping)
        for weight in sorted(policy.keys()):
            if weight in available_weights and policy[weight] == -1:
                return weight
        
        # If there is no flipping (no -1 action found), continue with maximum available weight
        return max(available_weights)
    
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
        
        # Calculate G-values for the episode (now includes final reward)
        g_values = self._calculate_g_values(episode)
        
        # Update Q-values using Monte Carlo update
        self._update_q_values(episode, g_values)
        
        return episode_length, final_weight
    
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
            
            # For Monte Carlo, use small step penalty during episode
            # Final episode reward will be calculated separately
            step_reward = 0  # No step penalty, only final outcome matters
            
            trajectory.append((weight, action, step_reward))
        
        return trajectory
    
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
        current_switch_point = initial_switch_point
        
        for episode in range(num_episodes):
            # Train on one episode
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
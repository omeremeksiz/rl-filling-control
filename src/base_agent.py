"""
Abstract base class for all reinforcement learning agents.
Defines the common interface that all RL methods must implement.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
import random
from data_processor import DataProcessor, FillingSession, SWITCH_TOKEN, TERMINATION_TOKEN
from reward_calculator import RewardCalculator
from config import (
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
from config import EXPLORATION_STEPS, EXPLORATION_PROBABILITIES


class BaseRLAgent(ABC):
    """Abstract base class for all RL agents."""
    
    def __init__(self, 
                 data_processor: DataProcessor,
                 reward_calculator: RewardCalculator,
                 exploration_rate: float = DEFAULT_EXPLORATION_RATE,
                 random_seed: int = DEFAULT_RANDOM_SEED,
                 learning_rate: float = DEFAULT_LEARNING_RATE,
                 discount_factor: float = DEFAULT_DISCOUNT_FACTOR,
                 initial_q_value: float = DEFAULT_INITIAL_Q_VALUE,
                 exploration_decay: bool = DEFAULT_EXPLORATION_DECAY,
                 exploration_min_rate: float = DEFAULT_EXPLORATION_MIN_RATE,
                 exploration_decay_rate: float = DEFAULT_EXPLORATION_DECAY_RATE,
                 exploration_decay_interval: int = DEFAULT_EXPLORATION_DECAY_INTERVAL):
        """
        Initialize the base agent.
        """
        self.data_processor = data_processor
        self.reward_calculator = reward_calculator
        self.initial_exploration_rate = exploration_rate
        self.exploration_rate = exploration_rate
        self.random_seed = random_seed
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.initial_q_value = initial_q_value
        
        # Exploration decay parameters
        self.exploration_decay = exploration_decay
        self.exploration_min_rate = exploration_min_rate
        self.exploration_decay_rate = exploration_decay_rate
        self.exploration_decay_interval = exploration_decay_interval
        
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
        """Train the agent for the specified number of episodes."""
        pass
    
    @abstractmethod
    def get_optimal_switch_point(self) -> int:
        """Get the learned optimal switch point."""
        pass
    
    @abstractmethod
    def get_q_table(self) -> Dict:
        """Get the agent's value function (Q-table or equivalent)."""
        pass
    
    def get_training_statistics(self) -> Dict:
        """Get training statistics."""
        return {
            'training_history': self.training_history,
            'method': self.__class__.__name__
        }
    
    def update_exploration_rate(self, episode: int) -> None:
        """Update exploration rate using exponential decay at specified intervals."""
        if self.exploration_decay and episode > 0 and episode % self.exploration_decay_interval == 0:
            self.exploration_rate = max(
                self.exploration_min_rate,
                self.exploration_rate * self.exploration_decay_rate
            )
    
    def select_action(self, current_switch_point: Optional[int] = None) -> int:
        """
        Epsilon-greedy with step-based exploration.
        Returns: (selected_switch_point, exploration_flag)
        """
        exploration_flag = False
        if random.random() < self.exploration_rate:
            exploration_flag = True
            return self._explore_with_steps(current_switch_point), exploration_flag
        else:
            exploration_flag = False
            best_switch_point = self._get_best_switch_point(current_switch_point)
            # Snap to nearest available if needed
            if best_switch_point not in self.available_switch_points:
                best_switch_point = min(self.available_switch_points, key=lambda x: abs(x - best_switch_point))
            return best_switch_point, exploration_flag
    
    def _explore_with_steps(self, current_switch_point: int = None) -> int:
        """
        Explore using step-based exploration from the best action (positive direction).
        """
        best_switch_point = self._get_best_switch_point(current_switch_point)
        available_points = sorted(self.available_switch_points)
        if best_switch_point not in available_points:
            best_switch_point = min(available_points, key=lambda x: abs(x - best_switch_point))
        best_index = available_points.index(best_switch_point)
        
        # Probabilistically choose the step
        random_value = random.random()
        cumulative_prob = 0.0
        for step, prob in zip(EXPLORATION_STEPS, EXPLORATION_PROBABILITIES):
            cumulative_prob += prob
            if random_value <= cumulative_prob:
                target_index = best_index + step
                return available_points[min(target_index, len(available_points) - 1)]
    
    @abstractmethod
    def _get_best_switch_point(self, current_switch_point: int = None) -> int:
        """Return the best switch point based on learned policy."""
        pass
    
    @abstractmethod 
    def _update_q_values(self, *args, **kwargs) -> None:
        """Algorithm-specific Q-value update."""
        pass
    
    def _initialize_q_table(self) -> Dict[Tuple[int, int], float]:
        """Initialize Q-table for state-action pairs (only filling-process weights)."""
        q_table = {}
        filling_weights = set()
        for session in self.data_processor.sessions:
            # Find termination index
            try:
                termination_index = session.weight_sequence.index(TERMINATION_TOKEN)
            except ValueError:
                termination_index = len(session.weight_sequence)
            
            # Process only weights during filling process (before final weight)
            for i in range(termination_index):
                weight = session.weight_sequence[i]
                # Skip indicators
                if weight in (SWITCH_TOKEN, TERMINATION_TOKEN):
                    continue
                # Skip final weight (position just before TERMINATION_TOKEN)
                if i == termination_index - 1:
                    continue
                filling_weights.add(weight)
        
        # Initialize Q-values for filling-process state-action pairs
        for weight in filling_weights:
            q_table[(weight, 1)] = self.initial_q_value   # Fast
            q_table[(weight, -1)] = self.initial_q_value  # Slow
        
        self.states_with_fast_action_updated = set()
        return q_table
    
    def _create_policy_from_q_values(self) -> Dict[int, int]:
        """Create policy by choosing best action for each state."""
        policy = {}
        state_actions = {}
        for (weight, action), q_value in self.q_table.items():
            state_actions.setdefault(weight, {})[action] = q_value
        for weight, actions in state_actions.items():
            best_action = max(actions, key=actions.get)
            policy[weight] = best_action
        return policy
    
    def _create_policy_from_q_values_with_tie_breaking(self) -> Dict[int, int]:
        """
        Tie-breaking rule:
        - If equal, choose slow (-1); otherwise choose action with higher Q.
        """
        policy = {}
        state_actions = {}
        for (weight, action), q_value in self.q_table.items():
            state_actions.setdefault(weight, {})[action] = q_value
        for weight, actions in state_actions.items():
            if len(actions) == 2:
                q_fast = actions.get(1, float('-inf'))
                q_slow = actions.get(-1, float('-inf'))
                policy[weight] = -1 if q_slow > q_fast else 1
            else:
                policy[weight] = max(actions, key=actions.get)
        return policy
    
    def train_episode(self, current_switch_point: int) -> Tuple[int, int]:
        """
        Shared per-episode training for MC/TD/Standard Q-Learning.
        Returns (episode_length, final_weight).
        """
        unused_sessions = self.data_processor.get_unused_sessions_for_switch_point(current_switch_point)
        selected_session = random.choice(unused_sessions)
        episode = self._create_episode_trajectory(selected_session, current_switch_point)
        
        final_weight = selected_session.final_weight if selected_session.final_weight is not None else 0
        episode_length = selected_session.episode_length
        final_episode_reward = self.reward_calculator.calculate_reward(episode_length, final_weight)
        
        if episode:
            last_state, last_action, last_step_reward = episode[-1]
            episode[-1] = (last_state, last_action, last_step_reward + final_episode_reward)
        
        self._update_q_values_from_episode(episode)
        return episode_length, final_weight
    
    @abstractmethod
    def _update_q_values_from_episode(self, episode: List[Tuple[int, int, float]]) -> None:
        """Update Q-values from the episode trajectory."""
        pass
    
    def _create_episode_trajectory(self, session: FillingSession, switch_point: int) -> List[Tuple[int, int, float]]:
        """
        Build (state, action, reward) tuples over the filling process (excludes final weight).
        """
        trajectory = []
        try:
            termination_index = session.weight_sequence.index(TERMINATION_TOKEN)
        except ValueError:
            termination_index = len(session.weight_sequence)
        
        for i in range(termination_index):
            weight = session.weight_sequence[i]
            # Skip indicators
            if weight in (SWITCH_TOKEN, TERMINATION_TOKEN):
                continue
            # Skip final weight (position just before TERMINATION_TOKEN)
            if i == termination_index - 1:
                continue
            
            action = 1 if weight < switch_point else -1  # fast before switch, slow after
            step_reward = self._get_step_reward()
            trajectory.append((weight, action, step_reward))
        
        return trajectory
    
    @abstractmethod
    def _get_step_reward(self) -> float:
        """Return step reward for the specific agent."""
        pass
    
    def train(self, num_episodes: int, initial_switch_point: Optional[int] = None, logger=None) -> List[Dict]:
        """Common training loop."""
        current_switch_point = initial_switch_point
        
        for episode in range(num_episodes):
            self.update_exploration_rate(episode)
            episode_length, final_weight = self.train_episode(current_switch_point)
            termination_type = self._determine_termination_type(final_weight)
            model_selected_next_switch_point = self._get_best_switch_point(current_switch_point)
            next_switch_point, exploration_flag = self.select_action(current_switch_point)
            explored_switch_point = next_switch_point if exploration_flag else None
            
            episode_data = {
                'episode': episode + 1,
                'episode_num': episode + 1,
                'switch_point': current_switch_point,
                'model_selected_switching_point': model_selected_next_switch_point,
                'explored_switching_point': explored_switch_point,
                'episode_length': episode_length,
                'final_weight': final_weight,
                'termination_type': termination_type,
                'exploration_rate': self.exploration_rate
            }
            self.training_history.append(episode_data)
            
            if (episode + 1) % 100 == 0 or episode == 0:
                explored_point = next_switch_point if exploration_flag else None
                print(f"--- Episode {episode + 1}/{num_episodes} ---")
                print(f"Experienced Switching Point: {current_switch_point}")
                print(f"Termination Type: {termination_type}")
                print(f"Model-Selected Next Switching Point: {model_selected_next_switch_point}")
                print(f"Explored Switching Point: {explored_point}")
                print()
            
            if logger:
                logger.log_episode(
                    episode + 1, num_episodes, current_switch_point,
                    termination_type, model_selected_next_switch_point,
                    explored_switch_point
                )
            
            current_switch_point = next_switch_point
        
        return self.training_history
    
    def _determine_termination_type(self, final_weight: int) -> str:
        """Determine termination type against safe band (training scale)."""
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

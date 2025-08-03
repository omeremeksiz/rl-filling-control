from file_handler import FileHandler
from state import State

class QValueHandler:
    def __init__(self, final_arr=None, max_states_all_episodes=None, initial_q_value_value=-125, target_state_config=None, learning_rate=0.001, max_weight_index=None):
        self.qValue = {}
        self.count = {}
        self.a = learning_rate
        self.INITIAL_Q_VALUE = initial_q_value_value
        self.filtered_qValue = {}
        self.file_handler = FileHandler()
        self.final_arr = final_arr
        self.max_states_per_episode = max_states_all_episodes
        self.max_weight_index = max_weight_index
        self.qValues_for_target_state = []

        # Set target state
        self.target_state = State(weight=target_state_config["weight"], action=target_state_config["action"])

        # Initialize q-values and counts for all (weight, action) pairs
        for i in range(0, max_weight_index + 1):
            weight = i
            for action in [1, -1]:
                state = State(weight, action)
                self.qValue[state] = self.INITIAL_Q_VALUE
                self.count[state] = 0

    def update_q_values(self, final_arr):
        """
        Update Q-values using a single selected filling (episode).
        final_arr is expected shape (1, max_states_all_episodes, 3).
        """
        sample = final_arr[0]  # Only one episode at a time

        for i in range(0, self.max_states_per_episode):
            weight = sample[i][0]
            action = sample[i][1]
            reward = sample[i][2]

            state = State(weight, action)
            previous_q_value = self.qValue.get(state, self.INITIAL_Q_VALUE)

            if reward == 0:
                continue  # Skip zero rewards (optional)

            if action == 1:  # Always update (weight, 1)
                self.qValue[state] += self.a * (reward - self.qValue[state])
                self.count[state] += 1
            elif action == -1:  # Update only if positive action was seen
                positive_state = State(weight, 1)
                if self.count[positive_state] > 0:
                    self.qValue[state] += self.a * (reward - self.qValue[state])
                    self.count[state] += 1
                else:
                    continue  # Skip update for unseen positive state

            if state == self.target_state:
                difference = self.qValue[state] - previous_q_value
                self.qValues_for_target_state.append((self.qValue[state], difference, reward))

    def write_to_text(self, file_path):
        self.file_handler.write_q_values(file_path, self.qValue, self.count)

    def write_target_state_qvalues_to_text(self, file_path):
        self.file_handler.write_target_state_qvalues(file_path, self.qValues_for_target_state)

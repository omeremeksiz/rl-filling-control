from .file_handler import FileHandler
from .state import State

class QValueHandler:
    def __init__(self, final_arr, max_states_per_episode, learning_rate, initial_q_value_value, target_state_config, upper_tolerance_weight, initial_q_value=None):
        self.qValue = {}
        self.count = {}
        self.a = learning_rate
        self.INITIAL_Q_VALUE = initial_q_value_value
        self.filtered_qValue = {}
        self.file_handler = FileHandler()
        self.final_arr = final_arr
        self.max_states_per_episode = max_states_per_episode
        self.target_state = State(weight=target_state_config["weight"], action=target_state_config["action"])
        self.qValues_for_target_state = []

        max_weight_index = round(upper_tolerance_weight / 10000)
        for i in range(0, max_weight_index + 1):
            weight = i 
            for action in [1, -1]:
                state = State(weight, action)
                self.qValue[state] = self.INITIAL_Q_VALUE
                self.count[state] = 0

        if initial_q_value:
            for state, (count, q_val) in initial_q_value.items():
                self.qValue[state] = q_val
                self.count[state] = count

    def calculate_q_values(self):
        for sample in self.final_arr:
            for i in range(0, self.max_states_per_episode):
                weight = sample[i][0]
                action = sample[i][1]
                reward = sample[i][2]

                state = State(weight, action)
                if reward == 0:
                    continue
                
                previous_q_value = self.qValue.get(state, self.INITIAL_Q_VALUE)

                if action == 1: # Always update (weight, 1)
                    self.qValue[state] += self.a * (reward - self.qValue[state])
                    self.count[state] += 1                    
                elif action == -1: # Only update if (weight, 1) has been updated before
                    positive_state = State(weight, 1)
                    if self.count[positive_state] > 0:
                        self.qValue[state] += self.a * (reward - self.qValue[state])
                        self.count[state] += 1                        
                    else:
                        continue # skip update for unseen positive state

                if state == self.target_state:
                    difference = self.qValue[state] - previous_q_value
                    self.qValues_for_target_state.append((self.qValue[state], difference, reward))

    def write_to_text(self, file_path):
        self.file_handler.write_q_values(file_path, self.qValue, self.count)

    def write_target_state_qvalues_to_text(self, file_path):
        self.file_handler.write_target_state_qvalues(file_path, self.qValues_for_target_state)

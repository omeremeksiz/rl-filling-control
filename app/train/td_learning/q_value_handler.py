class QValueHandler:
    def __init__(self, final_arr, max_states_per_episode, FileHandler, State):
        self.qValue = {}
        self.count = {}
        self.a = 0.001
        self.gama = 0.99
        self.INITIAL_Q_VALUE = -125
        self.file_handler = FileHandler()
        self.State = State
        self.final_arr = final_arr
        self.max_states_per_episode = max_states_per_episode
        self.target_state = State(weight=53, action=1)
        self.qValues_for_target_state = []
        self.qValues_for_specific_episode = [] 
        self.target_episode_index = 1

    def calculate_q_values_for_all_episodes(self):
        for episode in self.final_arr:
            last_valid_index = len(episode) - 1
            for i in range(last_valid_index, -1, -1):
                if episode[i][2] != 0:
                    last_valid_index = i
                    break

            for i in range(0, last_valid_index + 1):
                weight = episode[i][0]
                action = episode[i][1]
                reward = episode[i][2]
                current_state = self.State(weight, action)

                if current_state not in self.qValue:
                    self.qValue[current_state] = self.INITIAL_Q_VALUE
                    self.count[current_state] = 0

                previous_q_value = self.qValue[current_state]

                if i == last_valid_index:
                    self.qValue[current_state] += self.a * (reward - self.qValue[current_state])
                    self.count[current_state] += 1

                    if current_state == self.target_state:
                        difference = self.qValue[current_state] - previous_q_value
                        self.qValues_for_target_state.append((self.qValue[current_state], difference, reward))
                    break

                next_weight = episode[i + 1][0]
                next_action = episode[i + 1][1]
                next_state = self.State(next_weight, next_action)

                if next_state not in self.qValue:
                    self.qValue[next_state] = self.INITIAL_Q_VALUE
                    self.count[next_state] = 0

                self.qValue[current_state] += self.a * (reward + self.gama * self.qValue[next_state] - self.qValue[current_state])
                self.count[current_state] += 1

                if current_state == self.target_state:
                    difference = self.qValue[current_state] - previous_q_value
                    self.qValues_for_target_state.append((self.qValue[current_state], difference, reward))

    def calculate_q_values_for_single_episode(self):
        if not (0 <= self.target_episode_index < len(self.final_arr)):
            print("Invalid target episode index")
            return

        episode = self.final_arr[self.target_episode_index]

        last_valid_index = len(episode) - 1
        for i in range(last_valid_index, -1, -1):
            if episode[i][2] != 0:
                last_valid_index = i
                break

        for i in range(0, last_valid_index + 1):
            weight = episode[i][0]
            action = episode[i][1]
            reward = episode[i][2]
            current_state = self.State(weight, action)

            if current_state not in self.qValue:
                self.qValue[current_state] = self.INITIAL_Q_VALUE
                self.count[current_state] = 0

            previous_q_value = self.qValue[current_state]

            if i == last_valid_index:
                self.qValue[current_state] += self.a * (reward - self.qValue[current_state])
                self.count[current_state] += 1

                difference = self.qValue[current_state] - previous_q_value
                self.qValues_for_specific_episode.append((current_state, self.qValue[current_state], difference, reward))

                break

            next_weight = episode[i + 1][0]
            next_action = episode[i + 1][1]
            next_state = self.State(next_weight, next_action)

            if next_state not in self.qValue:
                self.qValue[next_state] = self.INITIAL_Q_VALUE
                self.count[next_state] = 0

            self.qValue[current_state] += self.a * (reward + self.gama * self.qValue[next_state] - self.qValue[current_state])
            self.count[current_state] += 1

            difference = self.qValue[current_state] - previous_q_value
            self.qValues_for_specific_episode.append((current_state, self.qValue[current_state], difference, reward))

    def get_q_values(self):
        return self.qValue, self.count

    def write_to_text(self, file_path):
        self.file_handler.write_q_values(file_path, self.qValue, self.count)
    
    def write_target_state_qvalues_to_text(self, file_path):
        self.file_handler.write_target_state_qvalues(file_path, self.qValues_for_target_state)

    def write_specific_episode_qvalues_to_text(self, file_path):
        q_value_output = []
        for i, (state, q_value, difference, reward) in enumerate(self.qValues_for_specific_episode, start=1):
            q_value_output.append(f"{state.weight}\t{int(state.action)}\t{difference}\t{q_value}\t{reward}")

        header = "Weight\tAction\tDifference\tQ Value\tReward"
        self.file_handler.write_to_text(file_path, header, q_value_output)

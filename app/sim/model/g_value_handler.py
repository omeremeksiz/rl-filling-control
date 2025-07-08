import numpy as np
import logging
from model.tolerance_pair import TolerancePair

class GValueHandler:
    def __init__(self, db_handler, query, tolerance_limits, gama=0.99, overflow_scale=0.0, underflow_scale=0.0):
        self.db_handler = db_handler
        self.query = query
        self.tolerance_pairs = {}
        self.df = None
        self.num_episodes = None
        self.max_states_per_episode = None
        self.final_arr = None
        self.reward_arr = None
        self.gama = gama
        self.min_weight = round(tolerance_limits[0] / 10000)
        self.max_weight = round(tolerance_limits[1] / 10000)
        self.overflow_scale = overflow_scale
        self.underflow_scale = underflow_scale

    def process_g_values(self):
        # Fetch data from the database
        self.df = self.db_handler.fetch_data(self.query)
        
        if self.df is None or 'raw_data' not in self.df.columns:
            logging.warning("No data or 'raw_data' column not found in the database.")
            return
        
        # Initialize episode and state counts, adjusting max states for indicators (-1, 300)
        self.num_episodes = len(self.df)
        self.max_states_per_episode = max(len(row.split(',')) for row in self.df['raw_data']) - 2  # Exclude -1 and 300
        self.final_arr = np.zeros((self.num_episodes, self.max_states_per_episode, 3))

        # Calculate rewards
        self.reward_arr = np.zeros((self.max_states_per_episode, 1))
        for i in range(self.reward_arr.shape[0]):
            total_sum = np.sum(self.gama ** np.arange(i + 1))
            self.reward_arr[i] = (-1) * total_sum

        # Process each episode in self.df as a separate filling
        for col_ind, row in self.df.iterrows():
            # Split 'raw_data' into individual states
            parsed_data = [int(value) for value in row['raw_data'].split(',')]
            action = 1
            row_ind = 0

            for state in range(len(parsed_data)):
                current_value = parsed_data[state]

                # Handle termination conditions
                if state + 1 < len(parsed_data) and (parsed_data[state + 1] == 300 or current_value > self.max_weight):
                    terminate_state_ind = row_ind 
                    terminate_state_weight = current_value

                    if terminate_state_weight > self.max_weight:
                        tolerance_pair = TolerancePair(cutoff_weight, 1001)
                        if tolerance_pair in self.tolerance_pairs:
                            self.tolerance_pairs[tolerance_pair] = max(self.tolerance_pairs[tolerance_pair], terminate_state_weight - self.max_weight)
                        else:
                            self.tolerance_pairs[tolerance_pair] = terminate_state_weight - self.max_weight
                        row_ind = 0
                        for state in self.final_arr[col_ind]:
                            if row_ind == terminate_state_ind:
                                break
                            distance = terminate_state_ind - row_ind
                            if state[1] != -1:
                                penalty = (terminate_state_weight - self.max_weight) * (self.overflow_scale) * pow(self.gama, distance)
                            else: 
                                penalty = 0
                            state[2] = self.reward_arr[terminate_state_ind - row_ind] + penalty
                            row_ind += 1
                    elif terminate_state_weight < self.min_weight:
                        tolerance_pair = TolerancePair(cutoff_weight, -1001)
                        if tolerance_pair in self.tolerance_pairs:
                            self.tolerance_pairs[tolerance_pair] = max(self.tolerance_pairs[tolerance_pair], self.min_weight - terminate_state_weight)
                        else:
                            self.tolerance_pairs[tolerance_pair] = self.min_weight - terminate_state_weight
                        self.final_arr[col_ind][row_ind][0] = current_value
                        self.final_arr[col_ind][row_ind][1] = action
                        row_ind = 0
                        for state in self.final_arr[col_ind]:
                            if row_ind == terminate_state_ind + 1:
                                break
                            distance = terminate_state_ind - row_ind
                            if state[1] != -1:
                                penalty = (self.min_weight - terminate_state_weight) * (self.underflow_scale) * pow(self.gama, distance)
                            else:
                                penalty = 0
                            state[2] = self.reward_arr[terminate_state_ind - row_ind] + penalty
                            row_ind += 1
                    else:
                        self.final_arr[col_ind][row_ind][0] = current_value
                        self.final_arr[col_ind][row_ind][1] = action
                        row_ind = 0
                        for state in self.final_arr[col_ind]:
                            if row_ind == terminate_state_ind + 1:
                                break
                            penalty = 0
                            state[2] = self.reward_arr[terminate_state_ind - row_ind] + penalty
                            row_ind += 1
                    break

                # Handle action changes for -1 values
                if current_value == -1:
                    action = -1
                    cutoff_weight = parsed_data[state + 1]
                    continue

                # Populate final_arr with current values
                self.final_arr[col_ind][row_ind][0] = current_value
                self.final_arr[col_ind][row_ind][1] = action
                row_ind += 1

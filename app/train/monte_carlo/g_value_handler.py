import numpy as np
import random
import logging
from file_handler import FileHandler
from tolerance_pair import TolerancePair

class GValueHandler:
    def __init__(self, min_weight, max_weight, gama, overflow_penalty, underflow_penalty, excel_file_path):
        self.file_handler = FileHandler()
        self.excel_file_path = excel_file_path
        self.tolerance_pairs = {}
        self.df = None
        self.max_states_all_episodes = None
        self.final_arr = None
        self.reward_arr = None
        self.gama = gama
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.overflow_penalty = overflow_penalty 
        self.underflow_penalty = underflow_penalty
        self.clusters = {} 

    def prepare_clusters(self):
        """
        Prepare clusters and extract data informations.
        """
        self.df = self.file_handler.read_excel(self.excel_file_path)

        if self.df is None:
            logging.error(f"Failed to load Excel file from {self.excel_file_path}")
            raise ValueError(f"Failed to load Excel file from {self.excel_file_path}")

        num_episodes = self.df.shape[1]
        self.max_states_all_episodes = self.df.applymap(lambda x: str(x).strip()).stack().groupby(level=1).count().max() - 2

        for episode in self.df.columns:
            filling = self.df[episode].tolist()
            try:
                switching_index = filling.index(-1)
                switching_point = filling[switching_index - 1]
            except ValueError:
                continue  # if no -1 is found in this filling, skip

            if switching_point not in self.clusters:
                self.clusters[switching_point] = []

            self.clusters[switching_point].append(filling)

        # Log cluster summary
        logging.info("\n=== Cluster Information ===")
        logging.info(f"{'Switching Point':<20}{'Number of Fillings'}")
        logging.info(f"{'-'*20} {'-'*20}")
        for switching_point, fillings in sorted(self.clusters.items()):
            logging.info(f"{switching_point:<20}{len(fillings)}")

        logging.info(f"\nTotal Switching Points (Clusters): {len(self.clusters)}")
        logging.info(f"Total Episodes: {num_episodes}\n")

        return num_episodes, self.max_states_all_episodes

    def select_filling(self, switching_point):
        """
        Select a filling from the cluster corresponding to the switching point.
        """
        if switching_point in self.clusters and self.clusters[switching_point]:
            filling = random.choice(self.clusters[switching_point])
            self.clusters[switching_point].remove(filling)
            return switching_point, filling

        return None, None

    def process_single_filling(self, filling):
        """
        Calculate G Values for single filling.
        """
        # filling: list of weights (states)
        self.final_arr = np.zeros((1, self.max_states_all_episodes, 3))  # Single episode
        self.reward_arr = np.zeros((self.max_states_all_episodes, 1))

        for i in range(self.reward_arr.shape[0]):
            total_sum = np.sum(self.gama ** np.arange(i + 1))
            self.reward_arr[i] = (-1) * total_sum

        action = 1
        row_ind = 0
        termination_type = "Normal"

        for idx, state_value in enumerate(filling):
            if idx + 1 < len(filling) and (filling[idx + 1] == 300 or state_value > self.max_weight):
                terminate_state_ind = row_ind
                terminate_state_weight = state_value

                cutoff_weight = terminate_state_weight 

                if terminate_state_weight > self.max_weight:
                    # Overflow
                    termination_type = "Overflow"
                    tolerance_pair = TolerancePair(cutoff_weight, 1001)
                    self.tolerance_pairs.setdefault(tolerance_pair, terminate_state_weight - self.max_weight)
                    row_ind = 0
                    for state in self.final_arr[0]:
                        if row_ind == terminate_state_ind:
                            break
                        distance = terminate_state_ind - row_ind
                        penalty = (terminate_state_weight - self.max_weight) * self.overflow_penalty * pow(self.gama, distance)
                        state[2] = self.reward_arr[terminate_state_ind - row_ind] + penalty
                        row_ind += 1

                elif terminate_state_weight < self.min_weight:
                    # Underflow
                    termination_type = "Underflow"
                    tolerance_pair = TolerancePair(cutoff_weight, -1001)
                    self.tolerance_pairs.setdefault(tolerance_pair, self.min_weight - terminate_state_weight)
                    self.final_arr[0][row_ind][0] = state_value
                    self.final_arr[0][row_ind][1] = action
                    row_ind = 0
                    for state in self.final_arr[0]:
                        if row_ind == terminate_state_ind + 1:
                            break
                        distance = terminate_state_ind - row_ind
                        penalty = (self.min_weight - terminate_state_weight) * self.underflow_penalty * pow(self.gama, distance)
                        state[2] = self.reward_arr[terminate_state_ind - row_ind] + penalty
                        row_ind += 1

                else:
                    # Normal ending
                    termination_type = "Normal"
                    self.final_arr[0][row_ind][0] = state_value
                    self.final_arr[0][row_ind][1] = action
                    row_ind = 0
                    for state in self.final_arr[0]:
                        if row_ind == terminate_state_ind + 1:
                            break
                        penalty = 0
                        state[2] = self.reward_arr[terminate_state_ind - row_ind] + penalty
                        row_ind += 1

                break  # Filling ends

            if state_value == -1:
                action = -1
                continue

            self.final_arr[0][row_ind][0] = state_value
            self.final_arr[0][row_ind][1] = action
            row_ind += 1
        
        return termination_type

    def write_tolerance_to_text(self, file_path):
        self.file_handler.write_tolerance_pairs(file_path, self.tolerance_pairs)

    def write_final_arr_to_text(self, file_path):
        self.file_handler.write_final_arr(file_path, self.final_arr)

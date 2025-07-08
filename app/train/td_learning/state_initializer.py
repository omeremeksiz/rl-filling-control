import numpy as np

class StateInitializer:
    def __init__(self, excel_file_path, FileHandler):
        self.excel_file_path = excel_file_path
        self.file_handler = FileHandler()
        self.df = None
        self.num_episodes = None
        self.max_states_per_episode = None
        self.final_arr = None
        self.reward_arr = None
        self.overflow_constant = 0
        self.underflow_constant = 0
        self.max_weight = 76
        self.min_weight = 74

    def initialize_states(self):
        self.df = self.file_handler.read_excel(self.excel_file_path)
        self.num_episodes = self.df.shape[1]
        self.max_states_per_episode = self.df.applymap(lambda x: str(x).strip()).stack().groupby(level=1).count().max() - 2
        self.final_arr = np.zeros((self.num_episodes, self.max_states_per_episode, 3))
    
        col_ind = 0
        for episode in self.df.columns:
            action = 1
            row_ind = 0
            for state in self.df.index:
                if self.df[episode][state + 1] == 300 or self.df[episode][state] > self.max_weight:
                    terminate_state_ind = state - 1
                    terminate_state_weight = self.df[episode][state]
                    if terminate_state_weight > self.max_weight:
                        penalty_overflow = (terminate_state_weight - self.max_weight) * (self.overflow_constant)
                        self.final_arr[col_ind][switch_ind][2] += penalty_overflow
                        self.final_arr[col_ind][terminate_state_ind - 1][2] = - 1 
                    elif terminate_state_weight < self.min_weight:
                        self.final_arr[col_ind][terminate_state_ind][0] = self.df[episode][state]
                        self.final_arr[col_ind][terminate_state_ind][1] = action
                        penalty_underflow = (self.min_weight - terminate_state_weight) * (self.underflow_constant)
                        self.final_arr[col_ind][switch_ind][2] += penalty_underflow
                        self.final_arr[col_ind][terminate_state_ind][2] = - 1
                    else:
                        self.final_arr[col_ind][terminate_state_ind][0] = self.df[episode][state]
                        self.final_arr[col_ind][terminate_state_ind][1] = action
                        self.final_arr[col_ind][terminate_state_ind][2] = - 1
                    break

                if self.df[episode][state] == -1:
                    action = -1
                    switch_ind = state - 1
                    continue

                self.final_arr[col_ind][row_ind][0] = self.df[episode][state]
                self.final_arr[col_ind][row_ind][1] = action
                self.final_arr[col_ind][row_ind][2] = -1
                row_ind += 1
            col_ind += 1

    def get_final_arr(self):
        return self.final_arr

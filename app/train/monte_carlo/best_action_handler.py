from file_handler import FileHandler
from state import State

class BestActionHandler:
    def __init__(self, qValue, count):
        self.best_action = {}
        self.qValue = qValue
        self.count = count
        self.file_handler = FileHandler()

    def find_best_actions(self):
        """
        Find best actions per state in single filling
        """
        self.best_action = {}
        sorted_items = sorted(self.qValue.items(), key=lambda x: (x[0].weight, x[0].action))
        
        for state, reward in sorted_items:
            weight = state.weight
            if weight not in self.best_action:
                q_value_1 = self.qValue.get(State(weight, 1), float("-inf"))
                q_value_minus_1 = self.qValue.get(State(weight, -1), float("-inf"))

                if q_value_1 >= q_value_minus_1:
                    count_value = self.count[State(weight, 1)]
                    self.best_action[weight] = (1, q_value_1, count_value)  # Action +1
                else:
                    count_value = self.count[State(weight, -1)]
                    self.best_action[weight] = (-1, q_value_minus_1, count_value)  # Action -1

    def get_next_switching_state(self):
        """
        Based on best action sequence, find the next switching state
        by detecting action flip (from +1 to -1).
        """

        for weight, (action, _, _) in self.best_action.items():
            if action == -1:
                return weight  # Return weight where -1 happened after flip

        return None # Return initial switchin weight if no flip is found

    def write_to_text(self, file_path):
        self.file_handler.write_best_actions(file_path, self.best_action)

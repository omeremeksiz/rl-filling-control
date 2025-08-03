from .file_handler import FileHandler
from .state import State
import random

class BestActionHandler:
    def __init__(self, qValue, count):
        self.best_action = {}
        self.qValue = qValue
        self.count = count
        self.file_handler = FileHandler()

    def find_best_actions(self):
        sorted_items = sorted(self.qValue.items(), key=lambda x: (x[0].weight, x[0].action))
        for state, reward in sorted_items:
            weight = state.weight
            if weight not in self.best_action:
                q_value_1 = self.qValue.get(State(weight, 1), float("-inf"))
                q_value_minus_1 = self.qValue.get(State(weight, -1), float("-inf"))
                if q_value_1 > q_value_minus_1:
                    count_value = self.count[State(weight, 1)]
                    self.best_action[weight] = (1, q_value_1, count_value)
                else:
                    count_value = self.count[State(weight, -1)]
                    self.best_action[weight] = (-1, q_value_minus_1, count_value)

    def find_action_flip(self):
        prev_action = None
        prev_weight = None

        for weight, (action, _, _) in self.best_action.items():
            if prev_action == 1 and action == -1:
                return prev_weight  # Return previous weight where +1 happened before flip
            prev_action = action
            prev_weight = weight

        return weight # Return last weight if no flip is found
    
    def exploration_or_explotation(self, flip_weight, prob=0.5):
        chosen_weight = flip_weight if random.random() < prob else flip_weight + 1
        return chosen_weight

    def write_to_text(self, file_path):
        self.file_handler.write_best_actions(file_path, self.best_action)

class BestActionHandler:
    def __init__(self, qValue, count, FileHandler, State):
        self.best_action = {}
        self.qValue = qValue
        self.count = count
        self.file_handler = FileHandler()
        self.State = State

    def find_best_actions(self):
        sorted_items = sorted(self.qValue.items(), key=lambda x: (x[0].weight, x[0].action))
        for state, q_value in sorted_items:
            weight = state.weight
            if weight not in self.best_action:
                q_value_1 = self.qValue.get(self.State(weight, 1), float("-inf"))
                q_value_minus_1 = self.qValue.get(self.State(weight, -1), float("-inf"))
                if q_value_1 > q_value_minus_1:
                    count_value = self.count.get(self.State(weight, 1), 0)
                    self.best_action[weight] = (1, q_value_1, count_value)
                else:
                    count_value = self.count.get(self.State(weight, -1), 0)
                    self.best_action[weight] = (-1, q_value_minus_1, count_value)

    def write_to_text(self, file_path):
        self.file_handler.write_best_actions(file_path, self.best_action)

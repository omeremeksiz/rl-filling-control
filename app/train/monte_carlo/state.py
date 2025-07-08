class State:
    def __init__(self, weight, action):
        self.weight = weight
        self.action = action

    def __eq__(self, other):
        return (self.weight, self.action) == (other.weight, other.action)

    def __hash__(self):
        return hash((self.weight, self.action))

    def __repr__(self):
        return f"State(weight={self.weight}, action={self.action})"

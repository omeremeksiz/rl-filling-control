class TolerancePair:
    def __init__(self, cutoff_weight, error_type):
        self.cutoff_weight = cutoff_weight
        self.error_type = error_type

    def __eq__(self, other):
        return (self.cutoff_weight, self.error_type) == (other.cutoff_weight, other.error_type)

    def __hash__(self):
        return hash((self.cutoff_weight, self.error_type))
    
    def __repr__(self):
        return f"State(cutoff_weight={self.cutoff_weight}, error_type={self.error_type})"
    
    
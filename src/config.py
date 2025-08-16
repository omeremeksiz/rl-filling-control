# EXPERIMENT PARAMETERS

# File paths
DATA_FILE_PATH = "data/data.xlsx"

# RL Method selection
DEFAULT_RL_METHOD = "mab"  # Options: "mab", "mc", "td", "qlearning"

# Common parameters (used by all methods)
DEFAULT_LEARNING_RATE = 0.1 # Use 0.1 for MAB
DEFAULT_EXPLORATION_RATE = 0.5
DEFAULT_TRAINING_EPISODES = 1000
DEFAULT_RANDOM_SEED = 42
DEFAULT_STARTING_SWITCH_POINT = 45.0

# Exploration decay parameters
DEFAULT_EXPLORATION_DECAY = True  # Whether to use exploration decay
DEFAULT_EXPLORATION_MIN_RATE = 0.01  # Minimum exploration rate
DEFAULT_EXPLORATION_DECAY_RATE = 0.995  # Decay factor when decay occurs
DEFAULT_EXPLORATION_DECAY_INTERVAL = 10  # Decay every N episodes

# Safe weight range (to be configured based on domain requirements)
DEFAULT_SAFE_WEIGHT_MIN = 74
DEFAULT_SAFE_WEIGHT_MAX = 76

# Reward calculation parameters
DEFAULT_OVERFLOW_PENALTY_CONSTANT = 0.0 # Penalty per weight unit above safe max
DEFAULT_UNDERFLOW_PENALTY_CONSTANT = 0.0  # Penalty per weight unit below safe min

# Exploration parameters (common - only positive direction)
EXPLORATION_STEPS = [1, 2, 3, 4, 5]  # exploration steps (+1, +2, +3, +4, +5 from best)
EXPLORATION_PROBABILITIES = [0.33, 0.27, 0.20, 0.13, 0.07]  # Must sum to 1.0 for guaranteed step selection
# EXPLORATION_STEPS = [1]  # exploration steps (+1, +2, +3, +4, +5 from best)
# EXPLORATION_PROBABILITIES = [1.00]  # Must sum to 1.0 for guaranteed step selection

# Method-specific parameters (used by MC, TD, and Q-learning)
DEFAULT_DISCOUNT_FACTOR = 0.99
DEFAULT_MC_INITIAL_Q_VALUE = -300.0  # Also used as initial Q-value for TD and Q-learning 
# EXPERIMENT PARAMETERS

# File paths
DATA_FILE_PATH = "../data/data.xlsx"

# Q-learning parameters
DEFAULT_LEARNING_RATE = 0.1
DEFAULT_EXPLORATION_RATE = 0.5
DEFAULT_TRAINING_EPISODES = 1000

# Random seed for reproducibility
DEFAULT_RANDOM_SEED = 42

# Starting switching point (None means random selection)
DEFAULT_STARTING_SWITCH_POINT = 45.0  # Use random selection by default

# Exploration parameters
EXPLORATION_STEPS = [1]  # exploration steps
EXPLORATION_PROBABILITIES = [1.0]  # probability weight
 
# Safe weight range (to be configured based on domain requirements)
DEFAULT_SAFE_WEIGHT_MIN = 74
DEFAULT_SAFE_WEIGHT_MAX = 76

# Reward calculation parameters
DEFAULT_PENALTY_MULTIPLIER = -10000.0 
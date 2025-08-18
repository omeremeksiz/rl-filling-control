"""
Configuration file for the RL filling control system.
Contains all default parameters for training, testing, and real-world operation.
"""

# Data and file paths
DATA_FILE_PATH = "data/data.xlsx"

# Operation mode
DEFAULT_OPERATION_MODE = "train"  # "train" for simulation or "test" for real device

# Training parameters
DEFAULT_TRAINING_EPISODES = 1000
DEFAULT_LEARNING_RATE = 0.1
DEFAULT_EXPLORATION_RATE = 0.9
DEFAULT_DISCOUNT_FACTOR = 0.99
DEFAULT_INITIAL_Q_VALUE = -125.0

# Safe weight range (for reward calculation)
DEFAULT_SAFE_WEIGHT_MIN = 74
DEFAULT_SAFE_WEIGHT_MAX = 76

# Penalty constants
DEFAULT_OVERFLOW_PENALTY_CONSTANT = 0.0
DEFAULT_UNDERFLOW_PENALTY_CONSTANT = 0.0

# Agent parameters
DEFAULT_RANDOM_SEED = 42
DEFAULT_STARTING_SWITCH_POINT = 45.0
DEFAULT_RL_METHOD = "qlearning"

# Exploration decay parameters
DEFAULT_EXPLORATION_DECAY = True
DEFAULT_EXPLORATION_MIN_RATE = 0.01
DEFAULT_EXPLORATION_DECAY_RATE = 0.995
DEFAULT_EXPLORATION_DECAY_INTERVAL = 5

# Exploration step parameters
EXPLORATION_STEPS = [1, 2, 3, 4, 5]
EXPLORATION_PROBABILITIES = [0.33, 0.27, 0.2, 0.13, 0.07]

# TESTING CONFIGURATION

# Network configuration for real-world testing
DEFAULT_TCP_IP = "127.0.0.1"
DEFAULT_TCP_PORT = 5050
DEFAULT_MODBUS_IP = "127.0.0.1"
DEFAULT_MODBUS_PORT = 1502
DEFAULT_MODBUS_REGISTER = 40001

# Database configuration for saving real episodes
DEFAULT_DB_CONFIG = {
    "name": "filling_data",
    "user": "root", 
    "password": "6637",
    "host": "127.0.0.1",
    "port": 3306
}

# Real-world device parameters
DEFAULT_WEIGHT_QUANTIZATION_STEP = 10000    # For converting real weights to model format (multiply simulation values by this)
DEFAULT_TCP_TIMEOUT = 0.1  # 100ms timeout for TCP communication
DEFAULT_TESTING_EPISODES = 10

# Note: Uses DEFAULT_SAFE_WEIGHT_MIN/MAX * DEFAULT_WEIGHT_QUANTIZATION_STEP for real device tolerance
# Note: Uses DEFAULT_STARTING_SWITCH_POINT for both training start and testing switching point

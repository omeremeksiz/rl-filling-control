# Reinforcement Learning for Filling Control

RL system for finding optimal switching points in container filling systems using MAB, Monte Carlo, TD, and Q-Learning methods.

## Problem

Find optimal switching point from fast to slow filling mode to:
- Keep final weight in safe range (74-76)
- Minimize filling time

## Usage

### Quick Start
```bash
pip install -r requirements.txt

# Training mode (set DEFAULT_OPERATION_MODE = "train" in config.py)
python main.py

# Testing mode (set DEFAULT_OPERATION_MODE = "test" in config.py)  
python main.py
```

### Configuration-Based Operation
The system operates based on `src/config.py` settings:

**Training Mode**:
- Set `DEFAULT_OPERATION_MODE = "train"`
- Configure RL method, episodes, learning parameters
- Run: `python main.py`

**Testing Mode**:
- Set `DEFAULT_OPERATION_MODE = "test"`
- Configure device IP, ports, switching points
- Run: `python main.py`

### Optional CLI Overrides
```bash
# Override method and episodes
python main.py --method qlearning --episodes 1000
```

**Methods**: MAB, Monte Carlo, TD, Q-Learning  
**Real-World Testing**: See [TEST.md](TEST.md)  
**Method Details**: See [METHODS.md](METHODS.md)

## Code Structure

### Project Layout
```
src/                    # Core source code directory
├── Core System
│   ├── config.py       # Global configuration parameters
│   └── filling_control_system.py  # Main orchestrator class with CLI
├── RL Agents
│   ├── base_agent.py              # Abstract base class for all RL methods
│   ├── agent_factory.py           # Factory for creating different agents
│   ├── q_learning_agent.py        # Multi-Armed Bandit (MAB) implementation
│   ├── monte_carlo_agent.py       # Monte Carlo method implementation
│   ├── td_agent.py                # Temporal Difference (TD) learning
│   └── qlearning_standard_agent.py # Standard Q-Learning implementation
├── Data & Rewards
│   ├── data_processor.py          # Excel data loading and preprocessing
│   ├── real_data_processor.py     # Real-world data preprocessing
│   └── reward_calculator.py       # Reward computation logic
├── Real-World Testing
│   ├── tcp_client.py              # TCP communication with device
│   ├── modbus_client.py           # Modbus communication for switching points
│   ├── database_handler.py        # Database operations for episodes
│   └── real_world_tester.py       # Real-world testing orchestrator
└── Utilities
    ├── logger.py                  # Training progress logging
    └── visualizer.py              # Plotting and visualization
    
main.py         # Entry point (delegates to src/)
notebooks/      # Data analysis and visualization notebooks
output/         # Training results (auto-generated)
data/           # Input data directory
METHODS.md      # Detailed method documentation
TEST.md         # Real-world testing guide
requirements.txt # Python dependencies
```

### File Responsibilities

#### Core System
- **`main.py`**: Simple entry point that delegates to `src/filling_control_system.py`
- **`config.py`**: Centralized configuration including learning rates, exploration parameters, reward penalties, and file paths
- **`filling_control_system.py`**: Main orchestrator that coordinates data loading, agent creation, training, logging, and visualization

#### RL Agents
- **`base_agent.py`**: Abstract base class defining the common interface for all RL methods (train_episode, get_best_action, etc.)
- **`agent_factory.py`**: Factory pattern implementation for creating different RL agents based on method selection
- **`q_learning_agent.py`**: Multi-Armed Bandit implementation treating each switch point as an independent arm
- **`monte_carlo_agent.py`**: Monte Carlo method with episode-based learning and return calculation
- **`td_agent.py`**: Temporal Difference learning with step-by-step value updates
- **`qlearning_standard_agent.py`**: Standard Q-Learning with state-action value function

#### Data Processing & Rewards
- **`data_processor.py`**: Handles Excel file loading, session parsing, and extraction of switch points and final weights
- **`real_data_processor.py`**: Processes real-world device data and converts it to model format
- **`reward_calculator.py`**: Computes rewards based on episode length and safety constraints (overflow/underflow penalties)

#### Real-World Testing
- **`tcp_client.py`**: TCP communication client for receiving filling data from physical device
- **`modbus_client.py`**: Modbus communication client for sending switching points to device
- **`database_handler.py`**: Database operations for storing real-world episodes and statistics
- **`real_world_tester.py`**: Orchestrates real-world testing sessions with device communication

#### Utilities
- **`logger.py`**: Manages training progress logging, file output, and experiment tracking
- **`visualizer.py`**: Creates plots for Q-values, switch point trajectories, and result analysis

**Data**: Located in `data/data.xlsx` - contains filling session data with columns as sessions and special tokens (-1 for switch, 300 for termination)

## Data

Excel file with filling sessions:
- Columns = sessions
- Values = weight measurements  
- `-1` = switch point, `300` = termination

## Methods

| Method | Type | Best For |
|--------|------|----------|
| MAB | Direct optimization | Simple problems |
| MC | Episode-based | Detailed analysis |
| TD | Step-by-step (conservative) | Safe policies |
| Q-Learning | Step-by-step (optimal) | Best performance |

## Configuration

Edit `src/config.py` for parameters like learning rate, exploration rate, episodes, and penalties.

### Real-World Testing

1. Set `DEFAULT_OPERATION_MODE = "test"` in `src/config.py`
2. Configure device IP/ports
3. Install: `pip install pymodbus mysql-connector-python`
4. Run: `python main.py`

**Setup Guide**: See [TEST.md](TEST.md)

## Output

Results saved to `output/{timestamp}/`:
- `training_process.log` - training logs
- `qvalue_vs_state.png` - Q-value plots  
- `switching_point_trajectory.png` - learning progress

**Notebooks**: Analysis and visualizations available in `notebooks/` directory 
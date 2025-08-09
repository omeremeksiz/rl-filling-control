# Reinforcement Learning for Filling Control

RL system for finding optimal switching points in container filling systems using MAB, Monte Carlo, TD, and Q-Learning methods.

## Problem

Find optimal switching point from fast to slow filling mode to:
- Keep final weight in safe range (74-76)
- Minimize filling time

## Usage

```bash
pip install -r requirements.txt
python main.py --method [mab|mc|td|qlearning] --episodes [number]
```

**Methods**: MAB, Monte Carlo, TD, Q-Learning  
**Details**: See [METHODS.md](METHODS.md)

## Code Structure

### Project Layout
```
src/                    # Core source code directory
├── Core System
│   ├── main.py         # Entry point and CLI argument parsing
│   ├── config.py       # Global configuration parameters
│   └── filling_control_system.py  # Main orchestrator class
├── RL Agents
│   ├── base_agent.py              # Abstract base class for all RL methods
│   ├── agent_factory.py           # Factory for creating different agents
│   ├── q_learning_agent.py        # Multi-Armed Bandit (MAB) implementation
│   ├── monte_carlo_agent.py       # Monte Carlo method implementation
│   ├── td_agent.py                # Temporal Difference (TD) learning
│   └── qlearning_standard_agent.py # Standard Q-Learning implementation
├── Data & Rewards
│   ├── data_processor.py          # Excel data loading and preprocessing
│   └── reward_calculator.py       # Reward computation logic
└── Utilities
    ├── logger.py                  # Training progress logging
    └── visualizer.py              # Plotting and visualization
    
output/         # Training results (auto-generated)
data/           # Input data directory
main.py         # Main entry point
METHODS.md      # Detailed method documentation
requirements.txt # Python dependencies
```

### File Responsibilities

#### Core System
- **`main.py`**: Simple entry point that delegates to `filling_control_system.py`
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
- **`reward_calculator.py`**: Computes rewards based on episode length and safety constraints (overflow/underflow penalties)

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

## Output

Results saved to `output/{timestamp}/`:
- `training_process.log` - training logs
- `qvalue_vs_state.png` - Q-value plots  
- `switching_point_trajectory.png` - learning progress
- `cluster_histogram.png` - switch point distribution 
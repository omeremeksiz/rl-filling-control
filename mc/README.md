# Monte Carlo Reinforcement Learning for Filling Control

This module implements Monte Carlo reinforcement learning for industrial filling process optimization.

## Project Structure

```
mc/
├── train/                 # Training implementation
│   ├── main.py           # Main training script
│   ├── config.yaml       # Training configuration
│   ├── best_action_handler.py
│   ├── g_value_handler.py
│   ├── q_value_handler.py
│   ├── plot_handler.py
│   ├── file_handler.py
│   ├── state.py
│   ├── tolerance_pair.py
│   └── output/           # Training outputs
├── sim/                  # Simulation environment
│   ├── main.py           # Simulation main script
│   ├── config.yaml       # Simulation configuration
│   ├── db_handler.py     # Database operations
│   ├── com/              # Communication modules
│   ├── data_preprocessing/
│   ├── model/            # RL model components
│   └── output/           # Simulation outputs
└── README.md
```

## Quick Start

### Training
```bash
cd mc/train
python main.py
```

### Simulation
```bash
cd mc/sim
python main.py
```

## Key Features

- **Monte Carlo Learning**: Implementation of MC methods for filling control
- **Simulation Environment**: Real-time process simulation with communication protocols
- **Data Processing**: Comprehensive data preprocessing and analysis tools
- **Visualization**: Plotting and analysis of training results

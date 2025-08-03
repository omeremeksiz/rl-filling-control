# RL-FILLING-Control

Reinforcement Learning approaches for industrial filling process optimization and control.

## Overview

This repository contains implementations of various reinforcement learning methods for optimizing industrial container filling processes. The project includes Monte Carlo learning and Multi-Armed Bandit approaches for determining optimal switching points between fast and slow filling modes.

## Project Structure

```
rl-filling-control/
├── mc/                     # Monte Carlo Reinforcement Learning
│   ├── train/             # Training implementation
│   ├── sim/               # Simulation environment
│   └── README.md
├── mab/                    # Multi-Armed Bandit approach
│   ├── src/               # Source code
│   ├── output/            # Results
│   └── README.md
├── data/                   # Data files (gitignored)
│   └── data.xlsx          # Training data
└── README.md              # This file
```

## Key Components

- **Monte Carlo Learning**: Implementation of MC methods for filling control optimization
- **Multi-Armed Bandit**: Q-learning based approach for optimal switching point determination
- **Simulation Environment**: Real-time process simulation with communication protocols
- **Data Analysis**: Comprehensive analysis and visualization tools

## Quick Start

### Monte Carlo Learning
```bash
cd mc/train
python main.py
```

### Multi-Armed Bandit
```bash
cd mab
python main.py
```

## Publications

1. Ö. S. Emeksiz, E. Maşazade, S. Selim, "AI-Driven Optimization of the Filling Process: A Comparison of Reinforcement Learning Methods," 33rd Signal Processing and Communications Applications Conference (SIU), 2025 (to appear)

2. Ö. S. Emeksiz, M. E. Ağcabay, E. Maşazade, S. Selim, T. Boysan, "A Reinforcement Learning Model for Industrial Filling Process Control," IEEE 30th International Conference on Electronics, Circuits and Systems (ICECS), 2023. [IEEE Xplore](https://ieeexplore.ieee.org/abstract/document/10382724)

## Acknowledgment

This work was supported by TÜBİTAK TEYDEB 1501 program under the project titled "Development of Data Analysis Device for Development of AI-Supported Weighing Device in Manufacturing Filling Systems (BX30Fill Analyzer)" (Project No: 3230048).

## License

All rights reserved. This project is proprietary and confidential. 
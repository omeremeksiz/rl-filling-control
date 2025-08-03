# Multi-Armed Bandit for Optimal Switching Point Learning

Reinforcement learning system for determining optimal switching points in industrial container filling systems using Multi-Armed Bandit approach.

## Problem

Industrial container filling systems operate in two modes:
- **Fast mode**: For speed
- **Slow mode**: For precision

Find the optimal switching point that:
- Ensures final weight in safe region
- Minimizes total filling time

## Quick Start

```bash
pip install -r requirements.txt
python main.py
```

## Project Structure

```
mab/
├── src/                    # Source code
│   ├── config.py          # Essential experiment parameters
│   ├── data_processor.py  # Data loading and processing
│   ├── reward_calculator.py # Reward function implementation
│   ├── q_learning_agent.py # Q-learning agent
│   ├── visualizer.py      # Plotting and visualization
│   ├── logger.py          # Training logging
│   └── filling_control_system.py # Main system orchestrator
├── output/               # Training outputs (auto-generated)
├── main.py               # Entry point
├── requirements.txt      # Python dependencies
└── README.md           # This file
```

**Note**: Data files are located in the root `data/` directory.

## Data Format

`data/data.xlsx` contains filling sessions where:
- Each **column** = separate session
- Values = weight measurements (non-negative integers)
- Special tokens: `-1` (switch point), `300` (termination)

## Algorithm

**Multi-Armed Bandit Q-learning** with:
- Q-table for each possible switch point
- Epsilon-greedy exploration
- Reward: `-length + β × penalty`

## Configuration

Edit `src/config.py` for essential experiment parameters:
- Safe weight range (`DEFAULT_SAFE_WEIGHT_MIN/MAX`)
- Learning parameters (`DEFAULT_LEARNING_RATE`, `DEFAULT_EXPLORATION_RATE`)
- Training episodes (`DEFAULT_TRAINING_EPISODES`)
- Starting switch point (`DEFAULT_STARTING_SWITCH_POINT`)

## Output

Training results are saved to `output/{training_id}/` with:
- Training logs
- Q-table analysis
- Visualization plots
- Optimal policy summary 
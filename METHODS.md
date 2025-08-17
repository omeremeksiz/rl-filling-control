# Reinforcement Learning Methods for Filling Control System

This document explains the four reinforcement learning methods implemented in our filling control system: Multi-Armed Bandit (MAB), Monte Carlo (MC), Temporal Difference (TD), and Standard Q-Learning.

## Problem Setup

### State-Action Definition
- **States**: Weight measurements during filling (e.g., 33, 45, 67)
- **Actions**: 
  - `1` = Fast filling mode
  - `-1` = Slow filling mode
- **Goal**: Find optimal switching point from fast to slow mode

### Reward Structure
```
For MAB (Multi-Armed Bandit):
  - Safe: -episode_length
  - Overflow: -episode_length + overflow_penalty × (final_weight - safe_max)
  - Underflow: -episode_length + underflow_penalty × (safe_min - final_weight)

For Monte Carlo (MC):
  - Step reward: -1.0 per step
  - Final reward: penalty only (overflow/underflow based on safe_max/safe_min)

For TD and Q-Learning:
  - Step reward: 0.0 per step  
  - Final reward: penalty only (overflow/underflow based on safe_max/safe_min)
```

## Method Comparison Overview

| Method | Type | Update Rule | Key Characteristic |
|--------|------|-------------|-------------------|
| **MAB** | Simple | Direct Q-update | Switch point optimization |
| **MC** | On-policy | Episode-based | Complete trajectory learning |
| **TD** | On-policy | Step-by-step | SARSA-style updates |
| **Q-Learning** | Off-policy | Step-by-step | Optimal policy learning |

---

## 1. Multi-Armed Bandit (MAB)

### Concept
Treats each possible switching point as a separate "arm" of a bandit. Directly learns Q-values for switching points.

### State-Action Space
- **States**: Switching points (45, 46, ..., 72)
- **Actions**: Choose switching point
- **Q-table**: `Q[switching_point] → expected_reward`

### Update Rule
```
Q(switch_point) = Q(switch_point) + α × (reward - Q(switch_point))
```

### Policy
```
best_switch_point = argmax Q(switch_point)
```

### Reward Structure
```
Safe: -episode_length
Overflow: -episode_length + overflow_penalty × (final_weight - safe_max)
Underflow: -episode_length + underflow_penalty × (safe_min - final_weight)
```

### Characteristics
- Simple and direct
- Fast convergence
- Limited to switching point optimization only
- Uses episode length as base penalty instead of per-step penalty

---

## 2. Monte Carlo (MC)

### Concept
Learns from complete episodes by calculating returns (G-values) and updating Q-values for all state-action pairs experienced.

### State-Action Space
- **States**: Weight measurements (0, 1, 2, ..., ~76)
- **Actions**: {1 (fast), -1 (slow)}
- **Q-table**: `Q[(weight, action)] → expected_return`

### G-value Calculation
For each state-action pair in an episode:
```
G(s,a) = r_t + γ×r_{t+1} + γ²×r_{t+2} + ... + γ^T×r_T
```
Where:
- `r_t = -1.0` (intermediate steps)
- `r_T = -1.0 + penalty` (final step)
- `γ = 0.99` (discount factor)

### Q-value Update
```
Q(s,a) = Q(s,a) + α × (G(s,a) - Q(s,a))
```

### Policy Extraction
1. For each weight, compare Q(weight, 1) vs Q(weight, -1)
2. Choose action with higher Q-value (tie-break: prefer -1)
3. Find first transition from action 1 → -1
4. Return that weight as switching point

### Special Implementation Tricks
- **Initialization**: All Q-values start at -125.0
- **Update Constraint**: Q(weight, -1) only updates if Q(weight, 1) was updated at least once
- **Data Filtering**: Exclude indicators (-1, 300) and final weights from state space

### Characteristics
- Learns detailed state-action policy
- Handles uncertainty well
- Requires complete episodes
- Slower convergence

---

## 3. Temporal Difference (TD) - True SARSA

### Concept
On-policy learning that updates Q-values after each step using the actual next action taken (SARSA-style).

### State-Action Space
Same as Monte Carlo:
- **States**: Weight measurements
- **Actions**: {1 (fast), -1 (slow)}

### Update Rule
```
Q(s,a) = Q(s,a) + α × [r + γ×Q(s',a') - Q(s,a)]
```
Where:
- `s'` = next state (weight)
- `a'` = **actual next action taken** (on-policy)
- `r = 0.0` (intermediate) or `penalty` (final)

### Policy
Same extraction method as Monte Carlo.

### Characteristics
- Learns from partial episodes
- On-policy (learns value of followed policy)
- Conservative (follows current policy)
- May converge to suboptimal policy

---

## 4. Standard Q-Learning

### Concept
Off-policy learning that updates Q-values using the maximum possible next Q-value, regardless of the action actually taken.

### State-Action Space
Same as Monte Carlo and TD.

### Update Rule
```
Q(s,a) = Q(s,a) + α × [r + γ×max_a'Q(s',a') - Q(s,a)]
```
Where:
- `max_a'Q(s',a')` = **best possible next action** (off-policy)
- `r = 0.0` (intermediate) or `penalty` (final)
- Always optimistic about future rewards

### Policy
Same extraction method as Monte Carlo.

### Characteristics
- Learns optimal policy regardless of exploration
- Fast convergence to optimal solution
- Can be overly optimistic
- May overestimate values during learning

---

## Common Implementation Details

### Exploration Strategy
All methods use **epsilon-greedy with step-based exploration**:
```python
if random() < exploration_rate:
    # Guaranteed exploration: probabilistically select step size
    steps = [1, 2, 3, 4, 5] with probabilities [0.2, 0.2, 0.2, 0.2, 0.2]
    return best_switch_point + selected_step
else:
    return best_switch_point  # Exploitation
```

### Bounds Checking
If selected switching point is outside available range:
```python
if switch_point > max_available:
    return max_available
elif switch_point < min_available:
    return min_available
```

### Reward Calculation
```python
def calculate_reward(episode_length, final_weight, method="standard"):
    if method.lower() == "mab":
        # MAB-specific reward calculation
        base_reward = -episode_length
        
        if final_weight > safe_max:  # Overflow
            penalty = (final_weight - safe_max) × overflow_penalty_constant
        elif final_weight < safe_min:  # Underflow  
            penalty = (safe_min - final_weight) × underflow_penalty_constant
        else:  # Safe
            penalty = 0.0
            
        return base_reward + penalty
    else:
        # Standard final reward calculation (step rewards handled separately)
        penalty = 0.0
        
        if final_weight > safe_max:  # Overflow
            penalty = (final_weight - safe_max) × overflow_penalty_constant
        elif final_weight < safe_min:  # Underflow  
            penalty = (safe_min - final_weight) × underflow_penalty_constant
            
        return penalty  # Step rewards: MC=-1.0, TD/Q-Learning=0.0
```

### Data Processing
- **Filter Indicators**: Remove -1 and 300 from weight sequences
- **Exclude Final Weight**: Don't use final weight as a state
- **Available Switch Points**: Extracted from actual data (typically 45-72)

---
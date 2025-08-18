# Real-World Testing Guide

## Quick Start

**Training**: Set `DEFAULT_OPERATION_MODE = "train"` → Run `python main.py`
**Testing**: Set `DEFAULT_OPERATION_MODE = "test"` → Configure IPs → Run `python main.py`

## Setup

### 1. Install
```bash
pip install -r requirements.txt
pip install pymodbus mysql-connector-python
```

### 2. Configure `src/config.py`
```python
# Switch mode
DEFAULT_OPERATION_MODE = "test"

# Device network
DEFAULT_TCP_IP = "192.168.1.100"
DEFAULT_TCP_PORT = 5050
DEFAULT_MODBUS_IP = "192.168.1.100"
DEFAULT_MODBUS_PORT = 502
DEFAULT_MODBUS_REGISTER = 40001

# Testing parameters
DEFAULT_TESTING_EPISODES = 10
DEFAULT_STARTING_SWITCH_POINT = 45.0
DEFAULT_WEIGHT_QUANTIZATION_STEP = 10000
```

## Communication

**TCP Data**: `"weight,time;weight,time;...;30,30;final_weight,time;300,300;timing"`
**Modbus**: Send switching point × 1000 to register 40001

## Expected Output
```
=== Real-World Testing Configuration ===
[TESTING]
rl_method: qlearning
num_episodes: 10
starting_switch_point: 45.0

--- Episode 1/10 ---
Experienced Switching Point: 45.0
Termination Type: overweight
Model-Selected Next Switching Point: 47.0
Explored Switching Point: None

=== Testing Completed ===
Total Episodes: 10
Results saved to: output/20250818_123456_abc123
Final optimal switching point learned: 48.5
```

## Troubleshooting
- **Connection failed**: Check IP/port/network
- **Data parsing failed**: Check TCP format
- **Development**: Use `cd sim/com/ && python tcp_mock_server.py`

**Config → Set "test" mode → Run `python main.py`**
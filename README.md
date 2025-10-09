## RL Filling Control

RL system for finding optimal switching points in container filling systems using MAB, Monte Carlo, TD, and Q-Learning methods.

### Problem

Find the optimal fast→slow switch point that
- keeps the final weight inside a safe band (74–76), and
- minimizes overall filling time.

### Layout

- `configs/` – YAML configs for each algorithm (`*_train.yaml`, `*_test.yaml`)
- `scripts/` – entry-point scripts (`train_*.py`, `test_*.py`)
- `utils/` – shared helpers (data loading, plotting, Excel export, logging, comms)
- `data/` – Excel dataset used by the trainers
- `outputs/` – auto-created run folders with logs, plots, Excel summaries
- `notebooks/` – optional analysis notebooks

### Train

```bash
pip install -r requirements.txt

python scripts/train_mab.py   # or train_mc.py / train_td.py / train_q.py
```

- Edit the matching file in `configs/` to tweak episodes, α/γ/ε, safety bounds, penalties, etc.
- Each run writes plots + `<algo>_qvalue_updates.xlsx` to `outputs/<timestamp>/`.

### Test

```bash
# Configure communication + episode count in configs/*_test.yaml
python scripts/test_mab.py    # or test_mc.py / test_td.py / test_q.py
```

Test scripts stream weights from TCP/Modbus endpoints (or their mocks), log to the same `outputs` directory, and record per-episode Q tables in Excel.

### Method Reminders

| Method | Type | Update | Reward Target | Highlight |
|--------|------|--------|---------------|-----------|
| **MAB** | Simple bandit | `Q ← Q + α (r − Q)` | Immediate reward | Switch-point exploration |
| **Monte Carlo** | On-policy | `G = Σ γ^t r_t`<br>`Q ← Q + α (G − Q)` | Full return | Trajectory averaging |
| **TD (SARSA)** | On-policy | `Q ← Q + α [r + γ Q(s', a') − Q]` | One-step return | Follows behaviour policy |
| **Q-Learning** | Off-policy | `Q ← Q + α [r + γ max_a Q(s', a) − Q]` | Greedy next-state value | Optimal-policy bias |

Overflow/underflow penalties and safe-weight bounds are injected through the config files and appear in the Excel reports so you can track convergence quickly.

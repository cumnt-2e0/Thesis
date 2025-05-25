# Thesis
Safe Reinforcement Learning for Real-time Microgrid Reconfiguration and Load Shedding Under Nonlinear Power Flow Constraints

# Thesis
Safe Reinforcement Learning for Real-time Microgrid Reconfiguration and Load Shedding Under Nonlinear Power Flow Constraints

Thesis/
│
├── env/                        # Core simulation logic
│   ├── __init__.py
│   ├── microgrid_env.py        # Gym environment
│   ├── builder.py              # Network construction (16-node)
│   ├── faults.py               # Fault/event injection
│   ├── reward.py               # Reward shaping logic
│   ├── constants.py            # Node indices, switch IDs priority levels
│   ├── visualisation.py        # Custom Plotly plotting
│
├── agents/                     # Future RL models (optional for now)
│   └── baseline.py             # Rule-based agent (fallback logic)
│
├── tests/
│   └── test_env_basic.py       # Smoke tests, unit tests
│
├── data/                       # Logs, scenarios, configs
│   └── scenarios.json          # Predefined fault/event scenarios
│
├── run_simulation.py           # CLI to run environment manually
├── train_rl_agent.py           # Future use (SB3 PPO/DQN training loop)
├── requirements.txt
├── README.md
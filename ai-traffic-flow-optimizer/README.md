# AI Traffic Flow Optimizer 🚦🚗

A beginner-friendly reinforcement learning project that learns to optimize traffic light phases at a single 4-way intersection. The goal is to minimize total vehicle waiting time and queue lengths using a tabular Q-learning agent on a lightweight, discrete simulator (no external traffic simulators required).

> Perfect for a student portfolio: clean code, reproducible experiments, plots, and extensions for future work (multi-intersection, SUMO/CityFlow integration).

## ✨ Features
- Discrete-time 4-way intersection simulator (N, S, E, W).
- Actions: `0 = keep NS green`, `1 = keep EW green`, with configurable **switch penalty & yellow time**.
- Stochastic arrivals (Bernoulli), capped service rate (vehicles passed per step).
- Reward = **- (total wait + switch penalty when changing phase)**.
- **Q-learning** with epsilon-greedy exploration + simple state aggregation (discretize queues).
- Training, evaluation, and visualization scripts.
- Reproducible config + saved artifacts (policy, metrics, plots).

## 🚀 Quickstart
```bash
# 1) (Optional) create a virtual environment
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 2) Install requirements
pip install -r requirements.txt

# 3) Train
python src/train.py --episodes 300 --render-every 0

# 4) Evaluate a trained policy
python src/evaluate.py --episodes 20 --load artifacts/q_table.pkl

# 5) Visualize queues over a single rollout
python src/visualize.py --steps 500 --load artifacts/q_table.pkl
```

## 📂 Project Structure
```
ai-traffic-flow-optimizer/
├── README.md
├── requirements.txt
├── src/
│   ├── env.py
│   ├── agent.py
│   ├── train.py
│   ├── evaluate.py
│   └── visualize.py
├── data/
├── artifacts/
├── .gitignore
└── LICENSE
```

## 🔧 Config (important hyperparameters)
- `arrivals`: per-direction arrival probability per step (default 0.2)
- `service_rate`: max cars you can pass per green direction per step (default 1)
- `switch_penalty`: extra penalty when changing phase (mimics yellow/all-red safety delay)
- `yellow_time`: forced dwell steps when switching (no cars pass)
- `queue_clip`: max queue bucket used for discretization

## 📈 Metrics
- Episode reward (higher is better).
- Mean total wait per step.
- Average queue length.
Saved to `artifacts/metrics.jsonl` and plots under `artifacts/plots/`.

## 🧩 State Representation
`(qN, qS, qE, qW, phase)` where queue lengths are **discretized** `[0..queue_clip]` and `phase in {0:NS, 1:EW, 2:YELLOW}`.

## 🏁 Actions
- `0`: prefer/keep NS green
- `1`: prefer/keep EW green
If action differs from current green, environment applies `yellow_time` steps (phase=2).

## 🧪 Experiments to try
- Increase arrival rates to create congestion and observe learned switching.
- Change `service_rate` (e.g., two-lane green).
- Replace reward with **pressure** (incoming - outgoing queues).
- Multi-intersection grid (independent agents).

## 🔌 Future Integration
- Swap `env.py` with a wrapper around **SUMO** or **CityFlow** for realism.
- Replace tabular agent with **DQN** (PyTorch).

## 📜 License
MIT

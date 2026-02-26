## 🚀 Agentic Load Balancer using Deep Reinforcement Learning

An intelligent, reinforcement learning–based cloud load balancer that dynamically allocates workloads across servers and performs adaptive autoscaling to optimize latency, load balance, and SLA compliance.

This project compares a Deep Q-Network (DQN) agent against traditional load balancing heuristics under stochastic cloud workloads.

---

## 📌 Overview

Modern cloud systems face dynamic and unpredictable workloads. Traditional load balancing strategies (e.g., Round Robin, Least Connections) do not adapt optimally to fluctuating demand and SLA constraints.

This project implements an Agentic Load Balancer using Deep Reinforcement Learning that:

- Learns optimal routing policies
- Performs intelligent autoscaling
- Minimizes latency and SLA violations
- Maintains balanced CPU utilization
- Adapts to stochastic request patterns

---

## 🧠 Key Features

- ✅ Deep Q-Network (DQN) based load balancing
- ✅ Configurable autoscaling (enable/disable)
- ✅ Poisson-distributed stochastic workload simulation
- ✅ GPU optional (automatic CPU fallback)
- ✅ Replay buffer with experience replay
- ✅ Target network stabilization
- ✅ Baseline comparisons:
  - Random Allocation
  - Round Robin
  - Least Connections
- ✅ Convergence analysis & reward visualization

---

## 🏗️ Project Architecture
```
CC Project
│
├── cloud_env_simulator.py            # Cloud environment simulation
├── configuration.py                  # All configurable hyperparameters
├── dqn_agent.py                      # DQN model + replay buffer
├── dqn_agent_trainer.py              # Training loop
├── baselines_vs_dqn_evaluator.ipynb  # Evaluation & plotting
├── requirements.txt
├── .gitignore
```

---

## ⚙️ How It Works
### Environment (cloud_env_simulator.py)

Simulates:
  - Server CPU utilization
  - Request queues
  - Poisson-distributed incoming workload
  - Processing rates
  - SLA violation penalties
  - Autoscaling actions
  - State Representation

Includes:
  - CPU utilization per server
  - Queue length per server
  - Average latency
  - Number of active servers
  - Load variance
  - Action Space

If autoscaling enabled:

  - 0 → MAX_SERVER_COUNT - 1   : Route to server
  - MAX_SERVER_COUNT           : Scale up
  - MAX_SERVER_COUNT + 1       : Scale down

Otherwise:

  - 0 → MAX_SERVER_COUNT - 1   : Route to server

---

## 🎯 Reward Function

The agent minimizes:
  - Average latency
  - SLA violations
  - Load imbalance
  - Excessive server usage
  - Reward= − [Latency + SLA_Penalty + LoadVariance + ScalingCost]

This encourages stable, balanced, low-latency operation.

---

## 🖥️ Installation

### 1️⃣ Clone Repository
git clone https://github.com/HaZaRdOuSDeVeLoPeR/CC-Project.git  
cd CC-Project

### 2️⃣ Create Virtual Environment
python -m venv venv
venv\Scripts\activate

### 3️⃣ Install Dependencies
pip install -r requirements.txt

### 4️⃣ (Optional) Install CUDA-enabled PyTorch for GPU

If you have an NVIDIA GPU:
  - pip install torch --index-url https://download.pytorch.org/whl/cu121

---

### baselines_vs_dqn_evaluator.ipynb
  - This notebook evaluates all baselines
  - Compares performance
  - Plots reward distributions
  - Demonstrates convergence

## 🔧Configuration Options

All configurable parameters are centralized in: configuration.py  
Including:
  - Server counts
  - Processing rate
  - SLA thresholds
  - Incoming workload parameters
  - DQN hyperparameters
  - Training episode count
  - Device selection

This makes experimentation and ablation studies easy.

---

## 🧪 Research Observations

  - DQN converges to a steady-state control strategy.
  - Evaluation with ε = 0 yields deterministic performance.
  - Autoscaling significantly improves stability under stochastic load.
  - Learned policy absorbs workload randomness better than heuristics.
  - Replay buffer and target network stabilize training.

---

## 📌 Future Improvements

  - Double DQN implementation
  - Prioritized Experience Replay
  - Parallelized environments
  - Multi-agent distributed control
  - Carbon-aware scheduling integration
  - Serverless simulation extension

---

## 📚 Technologies Used

  - Python 3.10
  - PyTorch
  - NumPy
  - Matplotlib
  - Reinforcement Learning (DQN)

---

## 👨‍💻 Author

Aditya Vimal  
B.Tech CSE  
NIT Warangal

# 🤖 Agentic Load Balancer using Deep Q-Network (DQN)

A reinforcement learning–based load balancer for cloud systems that dynamically routes incoming requests and adapts to varying workloads and heterogeneous server capacities.

---

## 📌 Overview

Traditional load balancing algorithms such as **Least Connections**, **Round Robin**, and **Random Routing** work well under simple conditions but struggle under:

- High load scenarios
- Heterogeneous server capacities
- Dynamic traffic patterns

This project introduces a **Deep Q-Network (DQN)** based agent that learns an optimal routing policy by interacting with a simulated cloud environment.

---

## 🎯 Key Features

- 🧠 **DQN-based intelligent routing**
- ⚖️ Comparison with classical baselines:
  - Least Connections (LC)
  - Round Robin (RR)
  - Random Policy
- 🏗️ **Heterogeneous server simulation**
- 📈 Real-time **training & evaluation dashboard**
- 🔄 Support for **variable traffic conditions (λ variation)**
- ⚡ Optional **autoscaling support**
- 📊 Visualization of:
  - Reward trends
  - Latency
  - SLA violations
  - CPU utilization
  - Action distribution

---

## 🧠 Problem Formulation

The system is modeled as a **Markov Decision Process (MDP)**:

- **State**:
  - Server queues
  - CPU utilization
  - Processing rates
  - Latency
  - Incoming request rate (λ)

- **Action**:
  - Select a server (or server pair) for routing

- **Reward**:
  Designed to balance:
  - Throughput
  - Latency
  - Load balancing
  - SLA violations

---

## ⚙️ System Architecture

```
cloud_env_simulator.py   → Environment (MDP)
dqn_agent.py            → Neural Network (Q-function)
dqn_agent_trainer.py    → Training Loop
app.py                  → Web Interface (Flask)
templates/              → UI (Jinja2 + Chart.js)
```

---

## 📊 Experimental Insights

### 🔹 Under Low Load (λ ≈ 5)
- Classical methods perform competitively
- System is underutilized
- RL offers marginal improvement

### 🔹 Under Moderate Load (λ ≈ 8–10)
- DQN begins to outperform baselines
- Better load distribution

### 🔹 Under High Load (λ ≥ 10)
- LC becomes unstable (high SLA violations)
- DQN maintains:
  - Lower latency
  - Significantly fewer SLA violations
  - Stable system behavior

---

## 🧪 Key Results

- ✅ Up to **10–70× reduction in SLA violations**
- ✅ Lower latency under high load
- ✅ More stable system performance
- ⚖️ Comparable performance under low load (expected behavior)

---

## 🚀 Getting Started

### 1. Clone Repository

```bash
git clone https://github.com/HaZaRdOuSDeVeLoPeR/CC-Project.git
cd CC-Project
```

---

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 3. Run the Application

```bash
python app.py
```

Open in browser:

http://localhost:5000

---

## 📈 Evaluation Metrics

- Average Reward
- Latency
- SLA Violations
- CPU Utilization
- Action Distribution

---

## ⚠️ Design Notes

- Reward is used **only for training**, not final evaluation
- Evaluation is based on **system-level metrics**
- Training noise is expected due to stochastic environment

---

## 🧩 Future Work

- Continuous action space (PPO / Actor-Critic)
- Multi-objective optimization
- Real-world traffic traces
- Advanced autoscaling policies

---

## 👨‍💻 Author

Aditya Vimal  
B.Tech CSE, NIT Warangal  

---

## ⭐ If you found this useful

Give it a ⭐ on GitHub!
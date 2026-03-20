import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque


class DQN(nn.Module):
    def __init__(self, state_size, action_size, hidden_dim=128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_size, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, action_size),
        )

    def forward(self, x):
        return self.network(x)


class ReplayBuffer:
    def __init__(self, capacity=10_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, ns, d = zip(*batch)
        return (
            torch.tensor(np.array(s,  dtype=np.float32)),
            torch.tensor(np.array(a,  dtype=np.int64)),
            torch.tensor(np.array(r,  dtype=np.float32)),
            torch.tensor(np.array(ns, dtype=np.float32)),
            torch.tensor(np.array(d,  dtype=np.float32)),
        )

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """Double DQN agent — fixed action space, no masking needed."""

    def __init__(self, state_size, action_size, *,
                 hidden_dim=128, gamma=0.99, lr=0.0005,
                 batch_size=128, buffer_size=10_000,
                 epsilon_start=1.0, epsilon_min=0.05, n_episodes=500,
                 target_update_steps=500, device="cpu"):

        self.action_size   = action_size
        self.gamma         = gamma
        self.batch_size    = batch_size
        self.epsilon       = epsilon_start
        self.epsilon_min   = epsilon_min
        self.epsilon_decay = (epsilon_min / epsilon_start) ** (1.0 / max(1, n_episodes))
        self.device        = torch.device(device)
        self.step_count    = 0
        self.target_update_steps = target_update_steps

        self.policy_net = DQN(state_size, action_size, hidden_dim).to(self.device)
        self.target_net = DQN(state_size, action_size, hidden_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory    = ReplayBuffer(buffer_size)

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)
        s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return int(self.policy_net(s).argmax(1).item())

    def train_step(self):
        if len(self.memory) < self.batch_size:
            return None
        s, a, r, ns, d = self.memory.sample(self.batch_size)
        s, a, r, ns, d = s.to(self.device), a.to(self.device), r.to(self.device), \
                         ns.to(self.device), d.to(self.device)

        q_cur = self.policy_net(s).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_a = self.policy_net(ns).argmax(1)
            q_next = self.target_net(ns).gather(1, next_a.unsqueeze(1)).squeeze(1)
            q_tgt  = r + self.gamma * q_next * (1.0 - d)

        loss = nn.SmoothL1Loss()(q_cur, q_tgt)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        self.step_count += 1
        if self.step_count % self.target_update_steps == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        return float(loss.item())

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

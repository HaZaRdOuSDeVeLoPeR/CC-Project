"""
Standalone training script — run directly with: python dqn_agent_trainer.py
The web app (app.py) uses the same logic but with live progress reporting.
"""
import os
import numpy as np
import torch

from cloud_env_simulator import CloudEnvironment
from dqn_agent import DQNAgent
from configuration import (
    SERVER_COUNT, TRAIN_EPISODE_COUNT, EPISODE_LENGTH, GAMMA, LEARNING_RATE,
    BATCH_SIZE, REPLAY_BUFFER_SIZE, EPSILON_START, EPSILON_MIN,
    TARGET_UPDATE_STEPS, HIDDEN_DIM, DEVICE, MODEL_PATH, REWARD_PATH,
)


def train():
    env = CloudEnvironment(num_servers=SERVER_COUNT)

    state_size  = len(env.reset())
    action_size = env.num_servers  # one action per server

    agent = DQNAgent(
        state_size=state_size,
        action_size=action_size,
        hidden_dim=HIDDEN_DIM,
        gamma=GAMMA,
        lr=LEARNING_RATE,
        batch_size=BATCH_SIZE,
        buffer_size=REPLAY_BUFFER_SIZE,
        epsilon_start=EPSILON_START,
        epsilon_min=EPSILON_MIN,
        n_episodes=TRAIN_EPISODE_COUNT,
        target_update_steps=TARGET_UPDATE_STEPS,
        device=DEVICE,
    )

    print(f"Device:      {agent.device}")
    print(f"State size:  {state_size}")
    print(f"Action size: {action_size}  (one per server)")
    print()

    rewards_per_episode = []

    for ep in range(TRAIN_EPISODE_COUNT):
        state        = env.reset()
        total_reward = 0.0

        for _ in range(EPISODE_LENGTH):
            action                   = agent.select_action(state)
            next_state, reward, done = env.step(action)
            agent.memory.push(state, action, reward, next_state, done)
            agent.train_step()
            state         = next_state
            total_reward += reward
            if done:
                break

        agent.decay_epsilon()
        rewards_per_episode.append(total_reward)

        if ep % 10 == 0:
            avg = np.mean(rewards_per_episode[-20:])
            print(f"Ep {ep+1:4d}/{TRAIN_EPISODE_COUNT}  "
                  f"reward={total_reward:10.1f}  avg20={avg:10.1f}  ε={agent.epsilon:.3f}")

    print("\nTraining complete.")
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    torch.save(agent.policy_net.state_dict(), MODEL_PATH)
    np.save(REWARD_PATH, np.array(rewards_per_episode))
    print(f"Model saved → {MODEL_PATH}")
    return rewards_per_episode


if __name__ == "__main__":
    train()

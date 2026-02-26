from cloud_env_simulator import CloudEnvironment
from dqn_agent import DQNAgent
import numpy as np
import torch

from configuration import AUTOSCALING, TRAIN_EPISODE_COUNT, EPISODE_LENGTH, MODEL_PATH, REWARD_PATH, MAX_SERVER_COUNT

def train():

    autoscale = AUTOSCALING 
    episodes = TRAIN_EPISODE_COUNT
    max_steps = EPISODE_LENGTH

    env = CloudEnvironment(autoscale=autoscale)

    state_size = len(env.reset())
    if autoscale:
        action_size = MAX_SERVER_COUNT + 2
    else:
        action_size = MAX_SERVER_COUNT

    agent = DQNAgent(state_size, action_size)
    print("Using device:", agent.device)

    rewards_per_episode = []

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0

        for step in range(max_steps):

            action = agent.select_action(state)
            next_state, reward, done = env.step(action)
            agent.memory.push(state, action, reward, next_state, done)
            agent.train()

            state = next_state
            total_reward += reward

            if done:
                break

        rewards_per_episode.append(total_reward)

        # decay epsilon per episode
        if agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.epsilon_decay

        if episode % 10 == 0:
            print("\nSample Q-values:",
                agent.policy_net(
                    torch.as_tensor(state, dtype=torch.float32)
                    .unsqueeze(0)
                    .to(agent.device)
                ).detach().cpu().numpy())
            print()

        print(f"Episode {episode+1} | Total Reward: {total_reward:.2f} | Epsilon: {agent.epsilon:.3f}")

    print("\nTraining Complete.")

    torch.save(agent.policy_net.state_dict(), MODEL_PATH)
    print("Model saved as dqn_trained_model.pth")

    np.save(REWARD_PATH, np.array(rewards_per_episode))
    print("Training rewards saved as training_rewards.npy")

    return rewards_per_episode

if __name__ == "__main__":
    train()
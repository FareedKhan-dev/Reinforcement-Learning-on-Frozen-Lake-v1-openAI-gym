###########################################################################
# This file contains the code for the Q-Learning agent for the FrozenLake-v1 environment
###########################################################################

# Importing Libraries
import argparse
import numpy as np
import gym
import matplotlib.pyplot as plt

# Define command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=0.8, help='learning rate')
parser.add_argument('--gemma', type=float, default=0.95, help='discount factor')
parser.add_argument('--num_episodes', type=int, default=10000, help='number of episodes')
args = parser.parse_args()

# Set the learning rate, discount factor, and number of episodes
lr = args.lr
gemma = args.gemma
num_episodes = args.num_episodes

# Create the environment
env = gym.make('FrozenLake-v1')

# Initialize the Q-table
Q = np.zeros([env.observation_space.n, env.action_space.n])

# lists to store performance metrics
total_rewards = []
total_steps = []
success_rate = []
test_reward = []

# Run the Q-learning algorithm
for i in range(num_episodes):
    s = env.reset()
    done = False
    episode_reward = 0
    num_steps = 0
    
    # The Q-Table learning algorithm
    while not done:
        a = np.argmax(Q[s,:] + np.random.randn(1,env.action_space.n)*(1./(i+1)))
        s_, r, done, _ = env.step(a)
        
        # update Q values
        Q[s,a] = Q[s,a] + lr*(r + num_episodes*np.max(Q[s_,:]) - Q[s,a])
        s = s_
        episode_reward += r
        num_steps += 1
    
    # append performance metrics to lists
    total_rewards.append(episode_reward/num_steps)
    total_steps.append(num_steps)
    success_rate.append(int(episode_reward > 0))

    # Print episode metrics
    print("Episode:", i+1, "Reward:", episode_reward, "Steps:", num_steps)


# Plotting the averge rewards and episode Lengths gained throughout each episode per episode
fig, axs = plt.subplots(1, 2, figsize=(200, 5))
axs[0].plot(total_rewards, 'tab:green')
axs[0].set_title('Average Reward per Episode')
axs[1].plot(total_steps, 'tab:purple')
axs[1].set_title('number of steps taken per episode')

plt.show()

# Print overall metrics
print('----------------------------------------------------------')
print("Overall Average reward:", np.mean(total_rewards))
print("Overall Average number of steps:", np.mean(total_steps))
print("Success rate (%):", np.mean(success_rate)*100)
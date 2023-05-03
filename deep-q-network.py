###########################################################################
# This file contains the code for the Deep Q-Learning agent for the FrozenLake-v1 environment
###########################################################################


# Importing the required libraries
import numpy as np
from keras.layers import Input, Dense
from keras.models import Model
from keras.optimizers import Adam
import gym
import matplotlib.pyplot as plt
import argparse

# Initialize argparse to get the input values
parser = argparse.ArgumentParser(description='Q-Learning agent for FrozenLake-v1 environment')
parser.add_argument('--gamma', type=float, default=0.95, help='discount factor to take future rewards into account')
parser.add_argument('--alpha', type=float, default=0.8, help='learning rate, to update the Q-table')
parser.add_argument('--n_hl1', type=int, default=32, help='number of units in the first hidden layer')
parser.add_argument('--n_hl2', type=int, default=32, help='number of units in the second hidden layer')
parser.add_argument('--epsilon', type=float, default=0.2, help='epsilon value for epsilon-greedy policy')
parser.add_argument('--num_episodes', type=int, default=1000, help='number of episodes to train the agent')
args = parser.parse_args()

# Initializing the agent's parameters
env = gym.make("FrozenLake-v1")
state_size = env.observation_space.n
action_size = env.action_space.n

gamma = args.gamma
alpha = args.alpha
n_hl1 = args.n_hl1
n_hl2 = args.n_hl2
epsilon = args.epsilon
num_episodes = args.num_episodes
action_space = np.arange(action_size)

# The network is initialized to output Q-Values for each action associated with a given state.
inputs = Input(shape=[state_size,])
X = Dense(n_hl1)(inputs)
X = Dense(n_hl2)(X)
outputs = Dense(action_size)(X)
network = Model(inputs=inputs, outputs=outputs)
network.compile(optimizer=Adam(learning_rate=alpha),
loss="categorical_crossentropy")

# Following the epsilon greedy policy to choose actions
def epsilon_greedy_action(state, epsilon=0.2):
    state = np.eye(state_size)[state]
    qvalues = network.predict(state.reshape([1, state_size]))
    A = np.zeros((qvalues.shape[1]))+epsilon/action_size
    greedy_action = np.argmax(qvalues[0])
    A[greedy_action] += 1-epsilon
    action = np.random.choice(action_space, p=A)
    return action

# Getting the target Q-Values for a particular state, and next_state pair(under a specific action)
def target_qvalues(qvalues, action, next_state, reward):
    next_state = np.eye(state_size)[next_state]
    q_nextstate = network.predict(
    next_state.reshape([1, state_size]))
    max_q = np.argmax(q_nextstate[0])
    target_qvalues = qvalues
    target_qvalues[action] = reward+gamma*q_nextstate[0, max_q]
    return target_qvalues

# Updating the network for each set
def update_network(state, action, reward, next_state):
    state = np.eye(state_size)[state]
    qvalues = network.predict(state.reshape([1, state_size]))
    target_qvalues_up = target_qvalues(
    qvalues[0], action, next_state, reward).reshape([1, action_size])
    state = state.reshape([1, state_size])
    network.fit(state, target_qvalues_up, epochs=1)

# Training the agent through a series of episodes and storing their values
reward_history = []
episode_lengths = []

successes = 0
for i in range(num_episodes):

    # Resetting the environment
    reward_buffer = 0
    j = 0
    state_now = env.reset()
    
    # Running the episode
    while True:
        # Selecting an epsilon greedy action
        action = epsilon_greedy_action(state_now)
        # Going to the next state on the basis of the chosen action
        state_next, reward, done, _ = env.step(action)
        # Updating the reward buffer and episode length
        reward_buffer += reward
        j += 1
        # Updating the network
        update_network(state_now, action, reward, state_next)

        # Checking if the episode is done
        if done == True:
            print(f'episode {i} is done\n')
            reward_history.append(reward_buffer/j)
            episode_lengths.append(j)
            print("Average number of steps per episode is {}".format(np.mean(episode_lengths)))

            # Checking if the episode was successful
            if reward_buffer == 1:
                successes += 1
            break
        else:
            state_now = state_next


# Plotting the averge rewards and episode Lengths gained throughout each episode per episode
fig, axs = plt.subplots(1, 2, figsize=(200, 5))
axs[0].plot(reward_history, 'tab:green')
axs[0].set_title('Average Reward per Episode')
axs[1].plot(episode_lengths, 'tab:purple')
axs[1].set_title('number of steps taken per episode')

plt.show()

# Printing the average reward and success rate
print('Average Reward throughout all episodes is {}'.format(
    sum(reward_history)/len(reward_history)))

# Printing the success rate
print('Success rate is {}'.format(successes/num_episodes))
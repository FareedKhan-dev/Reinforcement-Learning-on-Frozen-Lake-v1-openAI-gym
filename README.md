# Reinforcement Learning on Frozen Lake v1 - OpenAI Gym

### Table of Contents
-   Introduction
-   Implementation 1 - Q Learning
    -   Algorithm Approach
    -   Training
-   Implementation 2 – Deep Q Network
    -   Algorithm Approach
    -   Training
-   Installation
-   How to Run

## Introduction

Using the OpenAI Gym library, I implemented two reinforcement learning algorithms in the Frozen Lake environment (Figure 1.1). The environment requires the agent to navigate through a grid of frozen lake tiles, avoiding holes, and reaching the goal in the bottom-right corner.

![Frozen-Lake-and-table](/images/Picture4.svg)

## Implementation 1 - Q Learning

**Algorithm Approach**

Creating the Frozen Lake environment using the openAI gym library and initialized a Q-table with zeros. By default, the values of learning rate, discount factor, and number of episodes are 0.8, 0.95, and 10000 respectively in the given Python script. However, these values can also be changed using command-line arguments to customize the behavior of the script according to the requirements.
For selecting an action in each state according to an epsilon-greedy policy, we used the following equation:

$a = argmax(Q(s,a) + N(0, 1/(i+1)))$

    np.argmax(Q[s,:] + np.random.randn(1,env.action_space.n)*(1./(i+1)))

The second equation is used to update the Q-value for a state-action pair based on the observed reward and the maximum Q-value for the next state.

$Q(s,a) = Q(s,a) + α * [r + γ * max(Q(s',a')) - Q(s,a)]$

    Q[s,a] = Q[s,a] +  lr*(r + num_episodes*np.max(Q[s_,:]) - Q[s,a])

**Training**

During training, the agent took actions based on the highest Q-value for the current state with some random exploration, and the Q-table was updated after each action. The training metrics, including the total rewards per episode, number of steps taken per episode, and success rate, were recorded and printed after each episode.

After training the agent on default parameters for the specified number of episodes (i.e., 10,000), I printed the overall metrics. The overall average reward was 0.5263, the overall average number of steps was 49.7155, and the success rate was 52.692%. Despite increasing the number of episodes, the success rate could not surpass 55%.

The following diagram illustrates the average reward and number of steps per episode observed after running the script for 1000 episodes.

![q-learning-graphs](/images/Picture1.svg)


## Implementation 2 – Deep Q Network

**Algorithm Approach**

Creating the Frozen Lake environment using the openAI gym library and initialized the parameters of the agent including the environment, state size, action size, discount factor (0.95), learning rate (0.8), number of units in each hidden layer (32), and the action space. However, these values can also be changed using command-line arguments to customize the behavior of the script according to the requirements.

Neural network that takes the state as input and outputs Q-values for each action. The network is compiled with the Adam optimizer and the loss function is set to categorical crossentropy.

There are four main components of the applied approach:

 - epsilon_greedy_action - Returns an action according to the
   epsilon-greedy policy for a given state. 
 - target_qvalues - Calculates the target Q-values for a particular state, next_state pair under a specific action
 - update_network - Updates the network for each set.
 - For loop - Trains the agent through a series of episodes and prints the average reward and success rate.

**Training**

During training, the agent through a series of episodes. For each episode, the agent selects an action using the epsilon-greedy policy, updates the network based on the received reward, and continues until the episode is complete. The training metrics, including the total rewards per episode, number of steps taken per episode, and success rate, were recorded and printed after each episode.

After training the agent on default parameters for the specified number of episodes (i.e., 10,000), I printed the overall metrics. The overall average reward was 0.016, the overall average number of steps was 23.123, and the success rate was 31.251%.

The following diagram illustrates the average reward and number of steps per episode observed after running the script for 1000 episodes.

<svg src="/images/Picture2.sv"  width="600" height="300">

## Installation

    git clone https://github.com/FareedKhan-dev/Reinforcement-Learning-on-Frozen-Lake-v1-openAI-gym
    cd Reinforcement-Learning-on-Frozen-Lake-v1-openAI-gym
    pip install -r requirements.txt

## How to Run
virtual environment is recommended not necessary

#### For Q Learning 
 `python q-learning.py`
 
| Optional Arguments | Description | 
|--------------------|------------------------------------------------------| 
| --lr | Sets the learning rate (default: 0.8) | 
| --gamma | Sets the discount factor (default: 0.95) | 
| --num_episodes | Sets the number of episodes (default: 10000) |

#### For Deep Q Networks
 `python deep-q-network.py`
 
| Optional Arguments | Description                                               |
|--------------------|-----------------------------------------------------------|
| --gamma       | Sets the discount factor to take future rewards into account (default: 0.95) |
| --alpha      | Sets the learning rate, to update the Q-table (default: 0.1) |
| --n_hl1       | Sets the number of units in the first hidden layer (default: 32) |
| --n_hl2       | Sets the number of units in the second hidden layer (default: 32) |
| --epsilon   | Sets the epsilon value for epsilon-greedy policy (default: 1.0) |
| --num_episodes | Sets the number of episodes to train the agent (default: 10000) |

You can star the project, only if you like it: 

[![Star on GitHub](https://img.shields.io/github/stars/FareedKhan-dev/Reinforcement-Learning-on-Frozen-Lake-v1-openAI-gym.svg?style=social)](https://github.com/FareedKhan-dev/Reinforcement-Learning-on-Frozen-Lake-v1-openAI-gym/stargazers)

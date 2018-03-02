#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 21:15:42 2018

@author: lukaskemmer
"""

import numpy as np
import matplotlib.pyplot as plt
from q_s_agent import q_s_agent
from approximate_sarsa import approximate_sarsa_agent
from approximate_sarsa_V2 import approximate_sarsa_agent_V2
from approximate_sarsa_V3 import approximate_sarsa_agent_V3
from supply_distribution import SupplyDistribution
from reinforce import REINFORCE_agent

# ========================= 0. Function definitions ========================= #

def print_step(step, state, action, reward, state_new, total_reward, freq = 100):
    print("========= Step: %3s =========" % step)
    print("State:         ", state)
    print("Action:        ", action)
    print("Reward:        ", round(reward, 2))
    print("Next state:    ", state_new)
    print("Episode reward:", round(total_reward,2))
    
# ========================== 1. Setting parameters ========================== #

# Set seed
np.random.seed(10108)

# Simulation parameters
n_episodes = 5001
max_steps = 24  # 2 years = 52 * 2 weeks ( 2 week steps )

# Visualization parameters
output=1
status_freq = 10 # Print status (current episode) every X episodes
print_freq = 100 # Print current step every X episodes

# Instantiate environment
env = SupplyDistribution(n_stores=1, cap_truck=5, prod_cost=0, max_prod=10,
                 store_cost=np.array([0.5, 0]),
                 truck_cost=np.array([0]),
                 cap_store=np.array([20, 10]),
                 penalty_cost=4, price=0, gamma=0.90)

# Select agent

#agent = q_s_agent(threshold = np.array([10, 3, 3, 3]), reorder_quantity = np.array([11, 3, 3, 3]))
#agent = approximate_sarsa_agent(env)
#agent = approximate_sarsa_agent_V2(env)
agent = approximate_sarsa_agent_V3(env)
#agent = REINFORCE_agent(env,10,3, max_steps)

# ============================ 2. Evaluate agent ============================ #

# Initialize array with rewards
rewards = np.zeros(n_episodes)
stocks = np.zeros((n_episodes*max_steps, env.n_stores+1))
demands = np.zeros((n_episodes*max_steps, env.n_stores))

#Initialize epsilon
epsilon = 0.999

# Simulate n_episodes with max_steps each
for episode in np.arange(n_episodes):
    # Print number current episode each 100 episodes
    if episode % status_freq == 0:
        print("Episode ", episode)
    if episode == 5000:
        print("HERE!")
    # Reset environment, select initial action and reset episode reward
    state = env.reset()
    #action = agent.get_action(state, epsilon)
    action = agent.get_action(state)
    episode_reward = 0
    epsilon *= 0.999

    for step in np.arange(max_steps):
        
        # Log environment
        stocks[episode*max_steps+step, :] = env.s
        demands[episode*max_steps+step, :] = env.demand

        # Update environment
        state_new, reward, done, info = env.step(action)

        # Select a new action
        # action_new = agent.get_action(state_new, epsilon)
        action_new = agent.get_action(state_new)
        #action_new = agent.get_action(state)
        # Update episode reward
        episode_reward += np.power(env.gamma, step) * reward
        
        # Update agent
        agent.update(state, action, reward, state_new, action_new)

        # Print information
        if output and episode % print_freq == 0:
            print_step(step, state, action, reward, state_new, episode_reward,print_freq)
        
        # Update state
        state = state_new
        action = action_new
      
    # Add episodes reward to rewards list
    rewards[episode] = episode_reward

# ============================ 3. Output results ============================ #    

# Output results
print("Average reward: ", round(np.mean(rewards),2))

# Receive information from agent
ns = [agent.log[i][0] for i in range(len(agent.log))]
alphas = [agent.log[i][1] for i in range(len(agent.log))]
epsilons = [agent.log[i][2] for i in range(len(agent.log))]
deltas = [agent.log[i][3] for i in range(len(agent.log))]
thetas = [agent.log[i][4] for i in range(len(agent.log))]

# Plot results
fig = plt.figure(figsize=(5, 10), dpi=120)
fig.add_subplot(6, 1, 1)
plt.plot(rewards)
fig.add_subplot(6, 1, 2)
plt.plot(thetas)
fig.add_subplot(6, 1, 3)
plt.plot(deltas)
fig.add_subplot(6, 1, 4)
plt.plot(alphas)
fig.add_subplot(6, 1, 5)
plt.plot(epsilons)

# Plot some behavior
seq = np.arange(24*900,24*902)
fig = plt.figure(figsize=(5, 10), dpi=120)
fig.add_subplot(1, 1, 1)
plt.plot(stocks[seq,1])
plt.plot(demands[seq,0])

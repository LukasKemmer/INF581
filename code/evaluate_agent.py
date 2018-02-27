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
np.random.seed(10107)

# Simulation parameters
n_episodes = 10000
max_steps = 24 # 2 years = 2 * 52 weeks -- maybe better use 24 months = 2 years

# Visualization parameters
output=0
status_freq = 10 # Print status (current episode) every X episodes
print_freq = 1 # Print current step every X episodes

# Instantiate environment
store_cost = np.array([0.01, 0.1, 0.1, 0.1], dtype=np.float32)
truck_cost = np.array([2, 3, 4], dtype=np.float32)
cap_store=np.array([20, 5, 5, 5], dtype=np.int32)

env = SupplyDistribution(n_stores=3, cap_truck=100, prod_cost=1, max_prod=10,
                 store_cost=store_cost, truck_cost=truck_cost,
                 cap_store=cap_store, penalty_cost=1, price=5, gamma=0.9)

# Select agent
#agent = q_s_agent(threshold = np.array([10, 3, 3, 3]), reorder_quantity = np.array([11, 3, 3, 3]))
agent = approximate_sarsa_agent(env)

# ============================ 2. Evaluate agent ============================ #

# Initialize array with rewards
rewards = np.zeros(n_episodes)

# Simulate n_episodes with max_steps each
for episode in np.arange(n_episodes):
    # Print number current episode each 100 episodes
    if episode % status_freq == 0:
        print("Episode ", episode)

    # Reset environment, select initial action and reset episode reward
    state = env.reset()
    action = agent.get_action(state)
    episode_reward = 0
    
    for step in np.arange(max_steps):
        
        # Update environment
        state_new, reward, done, info = env.step(action)
        
        # Select a new action
        action_new = agent.get_action(state)
        
        # Update episode reward
        episode_reward += np.power(env.gamma, step) * reward
        
        # Update agent
        agent.update(state, action, reward, state_new, action_new)

        # Print information
        if output:
            print_step(step, state, action, reward, state_new, episode_reward,print_freq)
        
        # Update state
        state = state_new
        action = action_new
        
    # Add episodes reward to rewards list
    rewards[episode] = episode_reward
        
# ============================ 3. Output results ============================ #    

# Output results
print("Average reward: ", round(np.mean(rewards),2))
plt.plot(rewards)
plt.hist(rewards, normed=True)
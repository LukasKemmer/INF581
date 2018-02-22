#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 21:15:42 2018

@author: lukaskemmer
"""

import numpy as np
import matplotlib.pyplot as plt
from q_s_agent import q_s_agent
from supply_distribution import SupplyDistribution

# ========================= 0. Function definitions ========================= #

def print_step(step, state, action, reward, state_new, freq = 100):
    print("========= Step: %3s =========" % step)
    print("State:     ", state)
    print("Action:    ", action)
    print("Reward:    ", round(reward, 2))
    print("Next state:", state_new)
    
# ========================== 1. Setting parameters ========================== #

# Set seed
np.random.seed(10107)

# Simulation parameters
n_episodes = 50000
max_steps = 104 # 2 years = 2 * 52 weeks

# Visualization parameters
output=False
status_freq = 1000 # Print status (current episode) every X episodes
print_freq = 1 # Print current step every X episodes

# E
env = SupplyDistribution()

# Select agent
agent = q_s_agent(threshold = np.array([10, 3, 3, 3]), reorder_quantity = np.array([11, 3, 3, 3]))

# ============================ 2. Evaluate agent ============================ #

# Initialize array with rewards
rewards = np.zeros(n_episodes)

# Simulate n_episodes with max_steps each
for episode in np.arange(n_episodes):
    # Print number current episode each 100 episodes
    if episode % status_freq == 0:
        print("Episode ", episode)

    # Reset environment and episode reward
    state = env.reset()
    episode_reward = 0
    
    for step in np.arange(max_steps):
        # Select an action
        action = agent.get_action(state)
        
        # Update environment
        state_new, reward, done, info = env.step(action)
        
        # Update episode reward
        episode_reward += np.power(env.gamma, step) * reward
        
        # Update agent
        agent.update(state, action)
        
        # Print information
        if output:
            print_step(step, state, action, reward, state_new, print_freq)
        
        # Update state
        state = state_new
        
    # Add episodes reward to rewards list
    rewards[episode] = episode_reward
        
# ============================ 3. Output results ============================ #    

# Output results
print("Average reward: ", np.mean(rewards))
plt.plot(rewards)
plt.hist(rewards, normed=True)
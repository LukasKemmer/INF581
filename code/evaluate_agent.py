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

# Set seed
np.random.seed(10107)

# Set parameters
n_episodes = 1000
max_steps = 104 # 2 years = 2 * 52 weeks
output=False

# Initialize environment
env = SupplyDistribution()

# Select agent
agent = q_s_agent(threshold = np.array([10, 3, 3, 3]), reorder_quantity = np.array([11, 3, 3, 3]))

# Initialize array with rewards
rewards = np.zeros(n_episodes)

# Simulate n_episodes with max_steps each
for episode in np.arange(n_episodes):
    # Print number current episode each 100 episodes
    if episode % 100 == 0:
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
        
        # Update state
        state = state_new
        
    # Add episodes reward to rewards list
    rewards[episode] = episode_reward
        
# Output results
print("Average reward: ", np.mean(rewards))
#plt.plot(rewards)
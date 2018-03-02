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
#from approximate_sarsa_V2 import approximate_sarsa_agent_V2
#from approximate_sarsa_V3 import approximate_sarsa_agent_V3
from supply_distribution import SupplyDistribution
from reinforce2 import REINFORCE_agent

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
n_episodes = 10000
max_steps = 24 # 24 Months

# Visualization parameters
output=1
status_freq = int(n_episodes/100) # Print status (current episode) every X episodes
print_freq = int(n_episodes/5) # Print current step every X episodes
log_freq = int(n_episodes / 10) # Helper variable for when 

# Instantiate environment
env = SupplyDistribution(n_stores=1, 
                         cap_truck=4, 
                         prod_cost=1, 
                         max_prod=4,
                         store_cost=np.array([0, 0]),
                         truck_cost=np.array([0]),
                         cap_store=np.array([40, 14]),
                         penalty_cost=4, 
                         price=5, 
                         gamma=1, 
                         max_demand = 8,
                         episode_length = max_steps)

# Select agent
#agent = q_s_agent(threshold = np.array([10, 3, 3, 3]), reorder_quantity = np.array([11, 3, 3, 3]))
#agent = approximate_sarsa_agent(env)
#agent = approximate_sarsa_agent_V2(env)
#agent = approximate_sarsa_agent_V3(env)
agent = REINFORCE_agent(env, actions_per_store = 3, max_steps = max_steps)

# ============================ 2. Evaluate agent ============================ #

# Initialize array with rewards
rewards = np.zeros(n_episodes)
stocks = np.zeros((int(n_episodes / log_freq), max_steps, env.n_stores+1))
demands = np.zeros((int(n_episodes / log_freq), max_steps, env.n_stores))

# Simulate n_episodes with max_steps each
for episode in np.arange(n_episodes):
    
    

    # Reset environment, select initial action and reset episode reward
    state = env.reset()
    action = agent.get_action(state)
    episode_reward = 0
        
    for step in np.arange(max_steps):
        
        # Update environment
        state_new, reward, done, info = env.step(action)

        # Select a new action
        action_new = agent.get_action(state_new)

        # Update episode reward
        episode_reward += np.power(env.gamma, step) * reward
        
        # Update agent
        agent.update(state, action, reward, state_new, action_new)

        # Print information
        if output and episode % print_freq == 0:
            print_step(step, state, action, reward, state_new, episode_reward,print_freq)
            
        # Log environment
        if (episode+1) % log_freq == 0:
            stocks[int((episode+1) / log_freq - 1), step, :] = env.s
            demands[int((episode+1) / log_freq - 1), step, :] = env.demand
                
        # Update state
        state = state_new
        action = action_new
      
    # Add episodes reward to rewards list
    rewards[episode] = episode_reward
    
    # Print number current episode each 100 episodes
    if episode % status_freq == 0:
        print("Episode ", episode," Reward: ", episode_reward)

# ============================ 3. Output results ============================ #    

# Output results
print("Average reward: ", round(np.mean(rewards),2))

# Print rewards
plt.plot(rewards)

# Create plots from agent (e.g. parameter development over time)
agent.create_plots(rewards)

# Plot some behavior

fig = plt.figure(figsize=(10, 4), dpi=120)
for i in range(1,11):
    fig.add_subplot(5, 2, i)
    plt.plot(stocks[i-1,:,1])
    plt.plot(demands[i-1,:,0])

fig2 = plt.figure(figsize=(10, 4), dpi=120)
plt.plot(rewards)

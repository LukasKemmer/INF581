#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import numba as nb
from supply_distribution import SupplyDistribution

# Set environment
env = SupplyDistribution()

# Set input parameters
alpha = 0.2
n_episodes = 1000
max_steps = 104 # 2 years = 2 * 52 weeks
output=False

# Policy parameters
threshold = np.array([10, 3, 3, 3])
reorder_quantity = np.array([11, 3, 3, 3])

# Initialize value function
V = np.zeros(env.cap_store+1)

# Define heuristical policy
#@nb.jit(nopython=True, cache=True)
def s_q_policy(state, threshold, reorder_quantity):
    '''
        Heuristic based on s_q policy
    '''
    a = np.zeros(len(state))
    disposable_produce = state[0]
    # Set actions for warehouses
    for i in np.arange(1, len(state)):
        # If current stock is below threshold, replenish
        if state[i] < threshold[i]:
            # replenish as much as possible and update disposable produce
            a[i] = min(reorder_quantity[i], disposable_produce)
            disposable_produce -= a[i]

    # Set action for factory
    if disposable_produce < threshold[0]:
        a[0] = reorder_quantity[0]
    return a.astype(np.int)

# Temporal difference algorithm
returns_per_episode = np.zeros(n_episodes)

for i in np.arange(n_episodes):
    # Print current status
    if i % 100 == 0:
        print("Iteration ", i+1)

    # Reset environment
    s = env.reset()
    
    # Set return of episode to 0
    episode_return = 0
    
    for t in np.arange(max_steps):
        # Find action
        a = s_q_policy(s, threshold, reorder_quantity)

        # Take next step
        s_new, r, done, info = env.step(a)
        
        # Update episode return
        episode_return += np.power(env.gamma, t) * r
        
        # Print output if wanted
        if output:
            print("State: ", s)
            print("Action: ", a)
            print(info)
            print("New state: ", s_new)
            print("--")
        
        # Update value function
        V[tuple(s)] += alpha*(r + env.gamma*V[tuple(s_new)] - V[tuple(s)])

        # Update s
        s = s_new
    
    # Append return of last episode to returns_per episode
    returns_per_episode[i] = episode_return
    
print(np.mean(episode_return))
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 21:15:42 2018

@author: lukaskemmer
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from s_q_agent import s_q_agent
from approximate_sarsa import approximate_sarsa_agent
#from approximate_sarsa_V2 import approximate_sarsa_agent_V2
from approximate_sarsa_V3 import approximate_sarsa_agent_V3
from approximate_sarsa_V4 import approximate_sarsa_agent_V4
from supply_distribution import SupplyDistribution
from reinforce3 import REINFORCE_agent

# ========================= 0. Function definitions ========================= #

def print_step(step, state, action, reward, state_new, total_reward, freq = 100):
    print("========= Step: %3s =========" % step)
    print("State:         ", state)
    print("Action:        ", action)
    print("Reward:        ", round(reward, 2))
    print("Next state:    ", state_new)
    print("Episode reward:", round(total_reward,2))


def evaluate_agent(agent, env, n_episodes, max_steps, output, status_freq, print_freq, log_freq, results_folder_path, result_file_name):

    # Set seed
    np.random.seed(10108)


    # Initialize array with rewards
    rewards = np.zeros(n_episodes)
    stocks = np.zeros((int(n_episodes / log_freq), max_steps, env.n_stores+1))
    demands = np.zeros((int(n_episodes / log_freq), max_steps, env.n_stores))
    actions = np.zeros((int(n_episodes / log_freq), max_steps, env.n_stores+1))
    reward_log = np.zeros(int(n_episodes / log_freq))
    maxreward = -100000 # save the maximum reward episode

    current_stocks = np.zeros(( max_steps+1, env.n_stores+1))
    current_demands = np.zeros(( max_steps+1, env.n_stores))
    current_actions = np.zeros(( max_steps+1, env.n_stores+1))
    # Simulate n_episodes with max_steps each
    for episode in np.arange(n_episodes):
        # Reset environment, select initial action and reset episode reward
        state = env.reset()
        action = agent.get_action(state)
        episode_reward = 0

        # save some stuff
        current_stocks[0,:] = env.s
        current_demands[0,:] = env.demand
        current_actions[0,:] = action

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
            current_stocks[step,:] = env.s
            current_demands[step,:] = env.demand
            current_actions[step,:] = action_new

            if (episode+1) % log_freq == 0:
                stocks[int((episode+1) / log_freq - 1), step, :] = env.s
                demands[int((episode+1) / log_freq - 1), step, :] = env.demand
                actions[int((episode+1) / log_freq - 1), step, :] = action_new
                reward_log[int((episode+1) / log_freq - 1)] = episode_reward
            # Update state
            state = state_new
            action = action_new



        # Add episodes reward to rewards list
        rewards[episode] = episode_reward

        # Save for best reward actions
        if episode_reward >= max(episode_reward, maxreward):
            best_stocks = current_stocks.copy()
            best_demands= current_demands.copy()
            best_actions= current_actions.copy()
            maxreward = episode_reward



        # Print number current episode each 100 episodes
        if episode % status_freq == 0:
            print("Episode ", episode," Reward: ", episode_reward)

    # ============================ 3. Output results ============================ #

    # Output results
    print("Average reward: ", round(np.mean(rewards),2))

    # Print rewards
    #plt.plot(rewards)

    # Create plots from agent (e.g. parameter development over time)
    #agent.create_plots(rewards)

    # Plot some behavior

    #fig = plt.figure(figsize=(10, 4), dpi=120)
    #for i in range(1,11):
    #    fig.add_subplot(5, 2, i)
        #plt.title("after ",i*10,"% of the episodes")
    #    plt.plot(stocks[i-1,:,1], 'g', label='Stock warehouse 1')
    #    plt.plot(demands[i-1,:,0], 'r', label='Demand warehouse 1')
    #    plt.plot(stocks[i-1,:,0], 'b', label='Stock factory')
    #    plt.xlabel('time (months)')
    #    plt.ylabel('stock/price')
        # plt.legend()

    #fig4 = plt.figure(figsize=(10, 4), dpi=120)
    #for i in range(1,11):
    #    fig4.add_subplot(5, 2, i)
    #    plt.title("after %s of the episodes. Reward: %s"%(i*10,reward_log[i-1]))
    #    plt.plot(actions[i-1,:,0], 'b', label='production')
    #    plt.plot(actions[i-1,:,1], 'g', label='sending to warehouse')
    #    plt.xlabel('time (months)')
    #    plt.ylabel('stock/price')


    #fig2 = plt.figure(figsize=(10, 4), dpi=120)
    #fig2.add_subplot(2, 1, 1)
    #plt.title("For the best policy: reward = %s"%maxreward)
    #plt.plot(best_stocks[:,1], 'g', label='Stock warehouse 1')
    #plt.plot(best_demands[:,0], 'r', label='Demand warehouse 1')
    #plt.plot(best_stocks[:,0], 'b', label='Stock factory')
    #plt.xlabel('time (months)')
    #plt.ylabel('stock/price')
    #plt.legend()

    #fig2.add_subplot(2, 1, 2)
    #plt.plot(best_actions[:,0], 'b', label='production')
    #plt.plot(best_actions[:,1], 'g', label='sending to warehouse')
    #plt.xlabel('time (months)')
    #plt.ylabel('stock')
    #plt.legend()

    #fig3 = plt.figure(figsize=(10, 4), dpi=120)
    #plt.title("Rewards")
    #plt.plot(rewards)
    #plt.show()

    # Save all data:
    pd.DataFrame(best_stocks).to_csv(results_folder_path + result_file_name + "_best_stocks.csv", header=None, index=None)
    pd.DataFrame(best_demands).to_csv(results_folder_path + result_file_name + "_best_demands.csv", header=None, index=None)
    pd.DataFrame(best_actions).to_csv(results_folder_path + result_file_name + "_best_actions.csv", header=None, index=None)
    pd.DataFrame(rewards).to_csv(results_folder_path + result_file_name + "_rewards.csv", header=None, index=None)
    pd.DataFrame(current_stocks).to_csv(results_folder_path + result_file_name + "_last_stock.csv", header=None, index=None)
    pd.DataFrame(current_demands).to_csv(results_folder_path + result_file_name + "_last_demand.csv", header=None, index=None)
    pd.DataFrame(current_actions).to_csv(results_folder_path + result_file_name + "_last_actions.csv", header=None, index=None)

    # Test the saving of the rewards
    #df_rewards_loaded = pd.read_csv(results_folder_path + result_file_name + "_rewards.csv", header=None)
    #rewards_loaded = df_rewards_loaded.values.flatten()

    #if len(rewards_loaded) != len(rewards):
    #    print("error saving rewards!")

    #for i in range(len(rewards_loaded)):
    #    if rewards[i] != rewards_loaded[i]:
    #        print("error saving rewards!")

    #figTest = plt.figure(figsize=(10, 4), dpi=120)
    #figTest.add_subplot(2, 1, 1)
    #plt.title("Rewards")
    #plt.plot(rewards)

    #figTest.add_subplot(2, 1, 2)
    #plt.title("Rewards Loaded")
    #plt.plot(rewards_loaded)
    #plt.show()

    data_sets = np.array([])
    try:
        data_sets = pd.read_csv(results_folder_path + "data_sets.csv", header=None).values.flatten()
    except:
        data_sets = np.array([])
        print("files file not created, creating it")
    finally:
        pd.DataFrame(np.unique(np.append(data_sets, result_file_name))).to_csv(results_folder_path + "data_sets.csv", header=None, index=None)
        print(pd.read_csv(results_folder_path + "data_sets.csv", header=None).values.flatten())


# ========================== 1. Setting parameters ========================== #

# results path name
results_folder_path = "../results/"
result_file_name = "test_DIEGO"

# Simulation parameters
n_episodes = 20000
max_steps = 24 # 24 Months

# Visualization parameters
output=1
status_freq = int(n_episodes/100) # Print status (current episode) every X episodes
print_freq = int(n_episodes/5) # Print current step every X episodes
log_freq = int(n_episodes / 10) # Helper variable for when 

# Instantiate environment
env = SupplyDistribution(n_stores=1,
                         cap_truck=3,
                         prod_cost=1, 
                         max_prod=2,
                         store_cost=np.array([0, 0]),
                         truck_cost=np.array([0]),
                         cap_store=np.array([30, 10]),
                         penalty_cost=1,
                         price=1,
                         gamma=1,
                         max_demand = 4,
                         episode_length = max_steps)

env = SupplyDistribution(n_stores=1,
                         cap_truck=4,
                         prod_cost=1,
                         max_prod=1,
                         store_cost=np.array([0, 0.1]),
                         truck_cost=np.array([5]),
                         cap_store=np.array([10, 5]),
                         penalty_cost=1,
                         price=0,
                         gamma=0.9,
                         max_demand = 2,
                         episode_length = max_steps/2)

#env = SupplyDistribution(n_stores=1,
#                         cap_truck=3,
#                         prod_cost=1,
#                         max_prod=2,
#                         store_cost=np.array([0.5, 1]),
#                         truck_cost=np.array([1]),
#                         cap_store=np.array([30, 10]),
#                         penalty_cost=1,
#                         price=1,
#                         gamma=1,
#                         max_demand = 8,
#                         episode_length = max_steps)

# Select agent
#agent = q_s_agent(threshold = np.array([10, 3, 3, 3]), reorder_quantity = np.array([11, 3, 3, 3]))
#agent = approximate_sarsa_agent(env)
#agent = approximate_sarsa_agent_V2(env)
agent = approximate_sarsa_agent_V3(env)
#agent = approximate_sarsa_agent_V4(env)
#agent = REINFORCE_agent(env, actions_per_store = 3, max_steps = max_steps)

# ============================ 2. Evaluate agent ============================ #

#evaluate_agent(agent, env, n_episodes, max_steps, output, status_freq, print_freq, log_freq, results_folder_path, result_file_name)



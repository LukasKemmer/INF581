#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from q_s_agent import q_s_agent
from approximate_sarsa import approximate_sarsa_agent
#from approximate_sarsa_V2 import approximate_sarsa_agent_V2
from approximate_sarsa_V3 import approximate_sarsa_agent_V3
from approximate_sarsa_V4 import approximate_sarsa_agent_V4
from supply_distribution import SupplyDistribution
from reinforce3 import REINFORCE_agent
from evaluate_agent import evaluate_agent
from evaluate_agent import print_step

# results path name
results_folder_path = "../results/"

# Simulation parameters
n_episodes = 10000
max_steps = 24 # 24 Months

# Visualization parameters
output=1
status_freq = int(n_episodes/100) # Print status (current episode) every X episodes
print_freq = int(n_episodes/5) # Print current step every X episodes
log_freq = int(n_episodes / 10) # Helper variable for when

environments = []
env_names = []

add_q_s = False
add_sarsa = False
add_sarsa_V3 = False
add_reinforce_1 = True
add_reinforce_2 = False
add_reinforce_3 = False
add_reinforce_4 = False

# Simple2,3,4, Medium,2,3,4, weird,2, Difficult
test_to_run = [False, False, False, True, False, False, False, False, True, False]

# Instantiate environment
environments.append(SupplyDistribution(n_stores=1,
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
                                    episode_length = max_steps))
env_names.append("simple_environment_2")

environments.append(SupplyDistribution(n_stores=1,
                         cap_truck=4,
                         prod_cost=0.5,
                         max_prod=1,
                         store_cost=np.array([0, 0.1]),
                         truck_cost=np.array([0]),
                         cap_store=np.array([10, 10]),
                         penalty_cost=1,
                         price=1,
                         gamma=1,
                         max_demand = 3,
                         episode_length = max_steps))
env_names.append("simple_environment_3")

environments.append(SupplyDistribution(n_stores=1,
                         cap_truck=2,
                         prod_cost=0.5,
                         max_prod=1,
                         store_cost=np.array([0, 0.1]),
                         truck_cost=np.array([0]),
                         cap_store=np.array([10, 10]),
                         penalty_cost=1,
                         price=1,
                         gamma=1,
                         max_demand = 3,
                         episode_length = max_steps))
env_names.append("simple_environment_4")

environments.append(SupplyDistribution(n_stores=1,
                                       cap_truck=3,
                                       prod_cost=1,
                                       max_prod=2,
                                       store_cost=np.array([0, 0.5]),
                                       truck_cost=np.array([3]),
                                       cap_store=np.array([30, 10]),
                                       penalty_cost=1,
                                       price=1,
                                       gamma=1,
                                       max_demand = 4,
                                       episode_length = max_steps))
env_names.append("medium_environment_2")

environments.append(SupplyDistribution(n_stores=3,
                         cap_truck=3,
                         prod_cost=1,
                         max_prod=6,
                         store_cost=np.array([0, 0.4, 0.5, 0.6]),
                         truck_cost=np.array([2, 3, 4]),
                         cap_store=np.array([30, 6, 6, 6]),
                         penalty_cost=1,
                         price=3,
                         gamma=1,
                         max_demand = 4,
                         episode_length = max_steps))
env_names.append("medium_environment_3stores")

environments.append(SupplyDistribution(n_stores=3,
                         cap_truck=2,
                         prod_cost=0.5,
                         max_prod=5,
                         store_cost=np.array([0, 0.1, 0.1, 0.1]),
                         truck_cost=np.array([0, 0, 0]),
                         cap_store=np.array([30, 10, 10, 10]),
                         penalty_cost=1,
                         price=1,
                         gamma=1,
                         max_demand = 3,
                         episode_length = max_steps))
env_names.append("medium_environment_3stores2")

environments.append(SupplyDistribution(n_stores=3,
                                       cap_truck=3,
                                       prod_cost=1,
                                       max_prod=6,
                                       store_cost=np.array([0, 0.5, 0.5, 0.5]),
                                       truck_cost=np.array([3, 3, 3]),
                                       cap_store=np.array([90, 10, 10, 10]),
                                       penalty_cost=1,
                                       price=2.5,
                                       gamma=1,
                                       max_demand = 4,
                                       episode_length = max_steps))
env_names.append("medium_hard_environment_2")

environments.append(SupplyDistribution(n_stores=3,
                                       cap_truck=2,
                                       prod_cost=1,
                                       max_prod=3,
                                       store_cost=np.array([0, 2, 0, 0]),
                                       truck_cost=np.array([3, 3, 0]),
                                       cap_store=np.array([50, 10, 10, 10]),
                                       penalty_cost=1,
                                       price=2.5,
                                       gamma=1,
                                       max_demand = 3,
                                       episode_length = max_steps))
env_names.append("weird_environment")


environments.append(SupplyDistribution(n_stores=3,
                                       cap_truck=2,
                                       prod_cost=1,
                                       max_prod=3,
                                       store_cost=np.array([0, 2, 0, 0]),
                                       truck_cost=np.array([3, 3, 0]),
                                       cap_store=np.array([50, 10, 10, 10]),
                                       penalty_cost=1,
                                       price=2.5,
                                       gamma=1,
                                       max_demand = 4,
                                       episode_length = max_steps))
env_names.append("weird_environment_2")

environments.append(SupplyDistribution(n_stores=3,
                         cap_truck=2,
                         prod_cost=1,
                         max_prod= 5,
                         store_cost=np.array([0.1, 0.5, 0, 1]),
                         truck_cost=np.array([2, 4, 6]),
                         cap_store=np.array([30, 5, 10, 20]),
                         penalty_cost=1,
                         price=5,
                         gamma=1,
                         max_demand = 3,
                         episode_length = max_steps))
env_names.append("difficult_environment")





for test_num in range(len(test_to_run)):
    if test_to_run[test_num]:
        result_file_names = []
        agents = []
        env = environments[test_num]
        test_name = env_names[test_num]
        if add_q_s:
            result_file_names.append(test_name + "_q_s")
            agents.append(q_s_agent(threshold=np.array(env.cap_store/2), reorder_quantity=np.array([env.max_prod, env.cap_truck, env.cap_truck, env.cap_truck])))

        if add_reinforce_1:
            result_file_names.append(test_name + "_reinforce3")
            agents.append(REINFORCE_agent(env, actions_per_store=3, max_steps=max_steps))

        if add_sarsa:
            result_file_names.append(test_name + "_sarsa_V1")
            agents.append(approximate_sarsa_agent(env))

        if add_sarsa_V3:
            result_file_names.append(test_name + "_sarsa_V3")
            agents.append(approximate_sarsa_agent_V3(env))

        if add_reinforce_2:
            result_file_names.append(test_name + "_reinforce3_phi2")
            agents.append(REINFORCE_agent(env, actions_per_store=3, max_steps=max_steps, type_of_phi=2))

        if add_reinforce_3:
            result_file_names.append(test_name + "_reinforce3_phi3")
            agents.append(REINFORCE_agent(env, actions_per_store=3, max_steps=max_steps, type_of_phi=3))

        if add_reinforce_4:
            result_file_names.append(test_name + "_reinforce3_phi4")
            agents.append(REINFORCE_agent(env, actions_per_store=3, max_steps=max_steps, type_of_phi=0))


        for i in range(len(agents)):
            evaluate_agent(agents[i], env, n_episodes, max_steps, output, status_freq, print_freq, log_freq,
                           results_folder_path, result_file_names[i])

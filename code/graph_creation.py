import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def add_graph_to_show(agent_code_name, agent_label_name):
    agents.append(agent_code_name)
    agent_labels.append(agent_label_name)

###
# Graph creation file
# To create a graph for an agent, set the agent boolean variable to true.
# To select the environment, put the correct boolean variable to true in "tests_to_graph"
# If you want to run your own environment, append the name to "tests" and a boolean to "test to graph"


# Variables
add_s_q = True
add_sarsa = True
add_sarsa_V3 = False
add_reinforce_1 = False
add_reinforce_2 = True
add_reinforce_3 = False
add_reinforce_4 = False
tests = ["simple_environment_2", "simple_environment_3", "simple_environment_4", "medium_environment", "medium_environment_3stores", "medium_environment_3stores2", "medium_hard_environment_2", "special_environment", "special_environment_2", "difficult_environment"]
tests_to_graph = [False, False, False, True, False, False, False, True, False, False]
results_folder_path = "../results/"
#reward_plot_step = 10
colors = ['g', 'c', 'm', 'y', 'r', 'b']


# Create auxiliary lists
agents = []
agent_labels = []
add_s_q and add_graph_to_show("s_q", "s_q")
add_sarsa and add_graph_to_show("sarsa_V1", "Sarsa")
add_sarsa_V3 and add_graph_to_show("sarsa_V3", "Sarsa V3")
add_reinforce_1 and add_graph_to_show("reinforce3", "REINFORCE")
add_reinforce_2 and add_graph_to_show("reinforce3_phi2", "REINFORCE_phi2")
add_reinforce_3 and add_graph_to_show("reinforce3_phi3", "REINFORCE_phi3")
add_reinforce_4 and add_graph_to_show("reinforce3_phi4", "REINFORCE_phi4")

for test_num in range(len(tests)):
    test = tests[test_num]
    if tests_to_graph[test_num]:
        fig1 = plt.figure(figsize=(10, 4), dpi=120)
        for agent_num in range(len(agents)):
            agent = agents[agent_num]
            rewards = pd.read_csv(results_folder_path + test + "_" + agent + "_rewards.csv", header=None).values.flatten()
            #plt.title("Rewards for " + test)
            plt.plot(rewards[::10], colors[agent_num], label=agent_labels[agent_num])
            plt.xlabel('episodes/10')
            plt.ylabel('reward')
            plt.legend()
        plt.show()
        if False:
            fig2 = plt.figure(figsize=(10, 4), dpi=120)
            for agent_num in range(len(agents)):

                agent = agents[agent_num]

                best_actions = pd.read_csv(results_folder_path + test + "_" + agent + "_best_actions.csv", header=None).values
                plt.title("Action of best case scenario for " + agent_labels[agent_num])
                plt.plot(best_actions[:-1,0], 'b', label='production')
                plt.plot(best_actions[:-1,1], 'g', label='sending to warehouse 1')
                if best_actions.shape[1] > 2:
                    plt.plot(best_actions[:-1, 2], 'm', label='sending to warehouse 2')
                if best_actions.shape[1] > 3:
                    plt.plot(best_actions[:-1, 3], 'y', label='sending to warehouse 1')
                plt.xlabel('steps')
                plt.ylabel('stock')
                plt.legend()
                plt.show()
        if True:
            #fig3 = plt.figure(figsize=(10, 4), dpi=120)
            for agent_num in range(len(agents)):
                fig3 = plt.figure(figsize=(10, 4), dpi=120)
                #subplot = fig3.add_subplot(2, 1, agent_num + 1)
               # if agent_num == 0:
                #    subplot.set_title("Stocks for the (&; Q)-Policy")
                #if agent_num == 1:
                #    subplot.set_title("Stocks for the REINFORCE")
                agent = agents[agent_num]
                best_stocks = pd.read_csv(results_folder_path + test + "_" + agent + "_best_stocks.csv", header=None).values
                # best_demands = pd.read_csv(results_folder_path + test + "_" + agent + "_best_demands.csv", header=None).values
                #plt.title("For the best policy for " + agent_labels[agent_num])
                plt.plot(best_stocks[:-1,1], 'g', label='Stock warehouse 1')
                #plt.plot(best_demands[:,0], 'r', label='Demand warehouse 1')
                if best_stocks.shape[1] > 2:
                    plt.plot(best_stocks[:-1, 2], 'm', label='Stock warehouse 2')
                if best_stocks.shape[1] > 3:
                    plt.plot(best_stocks[:-1, 3], 'y', label='Stock warehouse 3')
                plt.plot(best_stocks[:-1,0], 'b', label='Stock factory')
                plt.plot(np.zeros(len(best_stocks)), 'k')
                plt.xlabel('steps')
                plt.ylabel('stock')
                #if agent_num==0:
                plt.legend()
            plt.show()
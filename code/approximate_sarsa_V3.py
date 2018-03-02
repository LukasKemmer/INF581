#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 24 13:06:37 2018
@author: lukaskemmer
"""
#from supply_distribution import SupplyDistribution
import numpy as np
import numba as nb

class approximate_sarsa_agent_V3(object):
    
    def __init__(self, env):
        # Set environment
        self.env = env
        
        # Initialize theta random
        self.theta = np.random.rand(29)
        self.thetas = [self.theta.copy()]
        # Initialize the stepsize alpha
        self.alpha = 0.01
        # Initialize Epsilon for epsilon greedy
        self.epsilon = 0.999
        # Initialize agent parameters for stepsize rule        
        self.n=1
        self.stepsizes = [self.alpha]
        # Initialize environment params
        self.env_params = (env.prod_cost, env.store_cost, env.price, env.n_stores)
        # Initialize status logger
        self.log = []
    
    def phi(self, state, action):
    
        # Copy variables for easier to read code
        n_stores = self.env.n_stores
        price = self.env.price
        prod_cost = self.env.prod_cost
        store_cost = self.env.store_cost.reshape(n_stores+1,1)
        penalty_cost = self.env.penalty_cost
        cap_truck = self.env.cap_truck
        truck_cost = self.env.truck_cost.reshape(n_stores,1)
        theta_size =  self.theta.shape[0]
        action_dim = action.ndim
        store_cap = self.env.cap_store
        
        # Initialize phi
        if action_dim==1:
            phi = np.zeros((theta_size, 1))
            action = action.reshape(1, n_stores+1) # reshape action so matrix operation is possible
        else:
            phi = np.zeros((theta_size, action.shape[0]))
            
        # Create simple estimates for demand and storage levels in the next episode
        d_next = (2*state[n_stores+1:2*n_stores+1] - state[2*n_stores+1:3*n_stores+1])
        s_next = ((state[0:n_stores+1] - np.hstack((0, d_next))).T + action).T
        
        # Save size of s_next
        s_shape = (s_next.shape[0]-1, s_next.shape[1])

        # Create simple scenarios for errors within s_next estimation
        s_next_plus = s_next.copy()
        s_next_plus[1:n_stores+1,:] += 1
        s_next_minus = s_next.copy()
        s_next_minus[1:n_stores+1,:] -= 1

        # Add bias
        phi[0, :] = 1
        # How much is being produce
        phi[1, :] = action[:, 0]
        # All negative states
        phi[2, :] = ((state[1:self.env.n_stores + 1] < 0) * 1).reshape(self.env.n_stores, 1)
        # Demand cannot be satisfied
        phi[3, :] = (state[1:self.env.n_stores + 1] + action[:, 1:self.env.n_stores + 1] - 2 * state[
                                                                                     self.env.n_stores + 1:2 * self.env.n_stores + 1] + state[
                                                                                                                              2 * self.env.n_stores + 1:] < 0).T
        # How much will there be after demand(only positive, 0 if can't be satisfied)
        # Stock will be more than X after producing
        # phi[2 + self.env.n_stores * 2, :] = (state[0] + action[:, 0] - sum(action[:, 1:].T) > 7)
        # phi[2 + self.env.n_stores * 2 + 1, :] = (state[0] + action[:, 0] - sum(action[:,1:].T) > 15)
        # Number of trucks being send
        # phi[5, :] = (np.ceil(action[:, 1:self.env.n_stores + 1] / self.env.cap_truck)).T
        # Will there be more than one truck of goods in the store?
        # phi[9, :] = state[1] + (action[:, 1] - (state[2] + state[3]) / 2).T > self.env.cap_truck
        # Will there be less or equal than one truck of goods in the store but more than 0?
        # phi[10, :] = np.logical_and((state[1] + (action[:, 1] - (state[2] + state[3]) / 2).T) <= self.env.cap_truck, (state[1] + (action[:, 1] - (state[2] + state[3]) / 2).T) >= 0)
        # More than 2 trucks of goods
        # phi[11, :] = state[1] + (action[:, 1] - (state[2] + state[3]) / 2).T > 2*self.env.cap_truck
        # How full is the truck
        phi[12, :] = np.ceil(action[:, 1] / self.env.cap_truck) - action[:, 1] / self.env.cap_truck
        # Penalty cost
        phi[13,:] = np.minimum(np.zeros(s_shape), s_next[1:,:]) / 100
        # Factory stock can satisfy next estimated demand
        phi[14,:] = (s_next[0] >= np.sum(d_next))*1
        # Format output in case of single action input
        production_rest = (state[0] + action[:, 0] - sum(action[:, 1:].T))
        for i in range(10):
            phi[15+i, :] = np.logical_and(production_rest > (store_cap[0]*0.1*i), production_rest <= (store_cap[0]*0.1*(i+1)))
        stock_rest = state[0] + action[:, 0] - sum(action[:, 1:].T)
        for i in range(4):
            phi[25+i, :] = np.logical_and(stock_rest > (store_cap[1] * 0.1 * i),  stock_rest <= (store_cap[1] * 0.1 * (i + 1)))
        if action_dim == 1:
            return phi.reshape((theta_size,))
        return phi
            
    def get_action(self, state):
        # Find all possible actions
        actions = np.array(self.env.action_space())
        
        # With probability epsilon, select random action
        if np.random.rand() < self.epsilon:
            return actions[np.random.randint(0, len(actions))]

        # With probability 1-epsilon, select greedy action
        return actions[np.argmax(np.dot(self.theta, self.phi(state, actions)))]        
    
    def update(self, state, action, reward, state_new, action_new):
        # Calculate delta
        delta = reward + self.env.gamma * np.dot(self.theta, self.phi(state_new, action_new)) - np.dot(self.theta, self.phi(state, action))

        # Update theta
        self.theta += self.alpha * delta * self.phi(state, action)        
        
        # LOG TODO: implement "logger"
        self.thetas.append(self.theta.copy())
        
        # Update alpha, epsilon and n
        self.epsilon *= 0.9999
        # self.alpha = update_alpha(self.alpha, self.n)
        #self.stepsizes.append(self.alpha)
        self.n+=1
        
        # Save information for log
        self.log.append([self.n, self.alpha, self.epsilon, delta, self.theta.copy()])

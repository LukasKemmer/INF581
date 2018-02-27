#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 24 13:06:37 2018

@author: lukaskemmer
"""
from supply_distribution import SupplyDistribution
import numpy as np
import numba as nb

'''
def phi(state, action, prod_cost, price, n_stores):
    # Create helper for parameters
    parameter_helper = np.zeros(1+n_stores)
    parameter_helper[0] = price - prod_cost
    parameter_helper[1:] = price
    
    # Create variable for result
    result = action.copy()

    # If action is 1d use simple computation
    if action.ndim == 1:
        result[1:] += state[1:n_stores+1] - state[n_stores+1:]
        return np.hstack((1, result*parameter_helper))
    
    # If action is 2d use matrix computation
    result[:, 1:] += (state[1:n_stores+1] - state[n_stores+1:])
    return np.vstack((np.ones(action.shape[0]), (result*parameter_helper).T))
'''

def phi(state, action, prod_cost, store_cost, price, n_stores):
    # Create helper for parameters
    parameter_helper = np.zeros(1+n_stores)
    parameter_helper[0] = price - prod_cost -store_cost[0]
    parameter_helper[1:] = price - store_cost[1:]
    
    # Create variable for result
    result = action.copy()

    # If action is 1d use simple computation
    if action.ndim == 1:
        result[1:] += state[1:n_stores+1] - state[n_stores+1:]
        return np.hstack((1, result*parameter_helper))
    
    # If action is 2d use matrix computation
    result[:, 1:] += (state[1:n_stores+1] - state[n_stores+1:])
    return np.vstack((np.ones(action.shape[0]), (result*parameter_helper).T))

@nb.njit(cache=True)
def update_alpha(alpha, n):
    return geometric_stepsize(alpha)

@nb.njit(cache=True)
def geometric_stepsize(alpha, beta=0.99):
    return alpha*beta

@nb.njit(cache=True)
def generalized_harmonic_stepsize(n, a=20):
    return a/(a+n-1)

@nb.njit(cache=True)
def mcclains_formular(alpha, n, target=0.005):
    return alpha/(1+alpha-target)

@nb.njit(cache=True)
def stc_stepsize(n, alpha_0=1, a=10, b=100, beta=0.75):
    return alpha_0 * (b/n+a) / (b/n+a+np.power(n, beta))

class approximate_sarsa_agent(object):
    
    def __init__(self, env):
        # Set environment
        self.env = env
        
        # Initialize theta random
        self.theta = np.ones(env.n_stores+2)#np.random.rand(5) # currently hard coded
        self.theta /= np.sum(self.theta)
        # Initialize the stepsize alpha
        self.alpha = 1#0.02
    
        # Initialize agent parameters for stepsize rule
        self.n=1
        self.stepsizes = [1]
        
        # Initialize environment params
        #self.env_params = (env.prod_cost, env.price, env.n_stores)
        self.env_params = (env.prod_cost, env.store_cost, env.price, env.n_stores)
    
    def get_action(self, state, epsilon=0.1):
        # Find all possible actions
        actions = np.array(self.env.action_space()) # TODO: Should be changed in supply_chain to return numpy array
        
        # With probability epsilon, select random action
        if np.random.rand() < epsilon:
            return actions[np.random.randint(0, len(actions))]
        
        # With probability 1-epsilon, select greedy action
        return actions[np.argmax(np.dot(self.theta, phi(state, actions, *self.env_params)))]        
    
    def update(self, state, action, reward, state_new, action_new):
        # Calculate delta
        delta = reward + self.env.gamma * np.dot(self.theta, phi(state_new, action_new, *self.env_params)) - np.dot(self.theta, phi(state, action, *self.env_params))

        # Update theta
        self.theta += self.alpha * delta * phi(state, action, *self.env_params)
        
        # Normalize theta to [0,1] (only relative weights should be important for policy selection)
        self.theta /= np.sum(self.theta)
        
        # Update learning rate alpha
        self.alpha = update_alpha(self.alpha, self.n)
        self.stepsizes.append(self.alpha)
        self.n+=1
        
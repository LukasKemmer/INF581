#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 24 13:06:37 2018

@author: lukaskemmer
"""
from supply_distribution import SupplyDistribution
import numpy as np


def phi(state, action, prod_cost, price, n_stores):
    # Create helper for parameters
    helper = np.zeros(1+n_stores)
    helper[0] = -prod_cost
    helper[1:] = price
    
    # Create helper for result
    result = action.copy()

    # If action is 1d use simple computation
    if action.ndim == 1:
        result[1:] += (state[1:n_stores+1] - state[n_stores+1:])
        return np.hstack((1, result*helper))
    
    # If action is 2d use matrix computation
    result[:, 1:] += (state[1:n_stores+1] - state[n_stores+1:])
    return np.vstack((np.ones(action.shape[0]), (result*helper).T))

    '''
    # If action is one-dimensional return hstack of state and action
    if action.ndim == 1:
        return np.hstack((state, action))

    # If matrix with multiple actions is entered, return matrix stack
    helper = np.array([state for i in np.arange(action.shape[0])])
    return np.vstack((helper.T, action.T))
    '''

def update_alpha(alpha):
    return alpha*0.99


class approximate_sarsa_agent(object):
    
    def __init__(self, env):
        # Set environment
        self.env = env
        
        # Initialize theta random
        self.theta = np.random.rand(5) # currently hard coded
        
        # Initialize the stepsize alpha
        self.alpha = 0.01
        
        # Initialize environment params
        self.env_params = (env.prod_cost, env.price, env.n_stores)
    
    
    def get_action(self, state, epsilon=0.1):
        # Find all possible actions
        actions = np.array(self.env.action_space()) # TODO: Should be changed in supply_chain to return numpy array
        
        # With probability epsilon, select random action
        if np.random.rand() < epsilon:
            return actions[np.random.randint(0, len(actions))]
        
        # With probability 1-epsilon, select greedy action
        return actions[np.argmax(np.dot(self.theta, phi(state, actions, *self.env_params)))]        
    
    
    def update(self, state, action, reward, state_new, action_new):
        # Find action for next iteration
        action_new = self.get_action(state)

        # Find calculate delta
        delta = reward + self.env.gamma * np.dot(self.theta, phi(state_new, action_new, *self.env_params)) - np.dot(self.theta, phi(state, action, *self.env_params))

        # Update theta
        self.theta += self.alpha * delta * phi(state, action, *self.env_params)
        
        # Update learning rate alpha
        self.alpha = update_alpha(self.alpha)
        
        
        
        
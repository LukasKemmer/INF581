#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 24 13:06:37 2018

@author: lukaskemmer
"""
#from supply_distribution import SupplyDistribution
import numpy as np
import numba as nb

def phi(state, action, env):
    return phi3(state, action, env)

def basic_phi(state, action):
    return np.hstack((state, action))

def phi1(state, action, prod_cost, store_cost, price, n_stores):
    # Create helper for parameters
    parameter_helper = np.zeros(1+n_stores)
    parameter_helper[0] = price - prod_cost - store_cost[0]
    parameter_helper[1:] = price - store_cost[1:]
    
    # Create variable for result
    result = action.copy()

    # If action is 1d use simple computation
    if action.ndim == 1:
        result[1:] += state[1:n_stores+1] - state[n_stores+1:2*n_stores+1]
        return np.hstack((1, result*parameter_helper))
    
    # If action is 2d use matrix computation
    result[:, 1:] += (state[1:n_stores+1] - state[n_stores+1:2*n_stores+1])
    return np.vstack((np.ones(action.shape[0]), (result*parameter_helper).T))

def phi2(state, action, prod_cost, store_cost, price, n_stores):
    # Create helper for parameters
    parameter_helper = np.zeros(1+n_stores)
    parameter_helper[0] = price - prod_cost
    parameter_helper[1:] = price
    
    # Create variable for result
    result = action.copy()

    # If action is 1d use simple computation
    if action.ndim == 1:
        result[1:] += state[1:n_stores+1] - state[n_stores+1:2*n_stores+1]
        return np.hstack((1, result*parameter_helper))
    
    # If action is 2d use matrix computation
    result[:, 1:] += (state[1:n_stores+1] - state[n_stores+1:2*n_stores+1])
    return np.vstack((np.ones(action.shape[0]), (result*parameter_helper).T))

def phi3(state, action, env):
    
    if action.ndim==1:
        phi = np.zeros(13) # +1 extra for bias. +1 to see

        # Add bias
        phi[0] = 1
    
        # How much is being produce
        phi[1] = action[0] #(state[0] + action[0] >= 5)*1
        # How much is not being produce
       # phi[2] = -action[0] # TODO delete

        # All positive states
       # phi[3] = (state[1:env.n_stores+1] >= 0)*1
        # All negative states
        phi[4] = (state[1:env.n_stores + 1] < 0) * 1 # TODO delete

        # trucks for demand
      #  phi[5] = min(np.ceil(state[1:env.n_stores+1] + action[1:env.n_stores+1] - 2*state[env.n_stores+1:2*env.n_stores+1] + state[2*env.n_stores+1:] / env.cap_truck), 0)
        # Demand cannot/can be satisfied
        phi[5] = state[1:env.n_stores + 1] + action[1:env.n_stores + 1] - 2 * state[env.n_stores + 1:2 * env.n_stores + 1] + state[2 * env.n_stores + 1:] < 0
        phi[6] = state[1:env.n_stores + 1] + action[1:env.n_stores + 1] - 2 * state[env.n_stores + 1:2 * env.n_stores + 1] + state[2 * env.n_stores + 1:] >= 0

        # Stock in main warehouse less than 25% and 50% after actions
        #phi[2 + env.n_stores * 2] = state[0] + action[0] - sum(action[1:]) > 7
        #phi[2 + env.n_stores*2 + 1] = state[0] + action[0] - sum(action[1:]) > 15

        # Number of trucks being send
        phi[7] = np.ceil(action[1:env.n_stores+1] / env.cap_truck)
        # Negative nunber of truck being send
       # phi[8] = -1*np.ceil(action[1:env.n_stores + 1] / env.cap_truck) # TODO delete
        # Will there be more than one truck of goods in the store?
        phi[9] = state[1] + action[1] - (state[2] + state[3])/2 > env.cap_truck
        # Will there be less or equal than one truck of goods in the store?
        phi[10] = state[1] + action[1] - (state[2] + state[3])/2 <= env.cap_truck\
                  and state[1] + action[1] - (state[2] + state[3])/2 >= 0
        # More than 2 trucks of goods
        phi[11] = state[1] + action[1] - (state[2] + state[3]) / 2 > 2*env.cap_truck
        # How full is the truck
        phi[12] = np.ceil(action[1]/env.cap_truck) - action[1]/env.cap_truck

        # Return result
        return phi
    
    # Matrix version
    phi = np.zeros((13, action.shape[0])) # 8 x 504
    phi[0,:] = 1
    phi[1,:] = action[:,0] #(state[0] + action[:, 0] >= 5)*1
  #  phi[2, :] = -action[:, 0]
  #  phi[3,:] = ((state[1:env.n_stores+1] >= 0)*1 ).reshape(env.n_stores, 1)
    phi[4, :] = ((state[1:env.n_stores + 1] < 0) * 1).reshape(env.n_stores, 1)
   # phi[5, :] = np.minimum(state[1:env.n_stores+1] + action[:,1:env.n_stores+1] - 2*state[env.n_stores+1:2*env.n_stores+1] + state[2*env.n_stores+1:], np.zeros((action.shape[0],1))).T
    phi[5, :] = (state[1:env.n_stores+1] + action[:,1:env.n_stores+1] - 2*state[env.n_stores+1:2*env.n_stores+1] + state[2*env.n_stores+1:] < 0).T
    phi[6, :] = (state[1:env.n_stores+1] + action[:,1:env.n_stores+1] - 2*state[env.n_stores+1:2*env.n_stores+1] + state[2*env.n_stores+1:] >= 0).T
    #phi[2 + env.n_stores * 2, :] = (state[0] + action[:, 0] - sum(action[:, 1:].T) > 7)
    #phi[2 + env.n_stores * 2 + 1, :] = (state[0] + action[:, 0] - sum(action[:,1:].T) > 15)
  # phi[2 + env.n_stores * 2 + 2: 2 + env.n_stores * 3 + 2, :] = ((action[:, 1:env.n_stores + 1] / env.cap_truck).round()).T
    phi[7, :] = (np.ceil(action[:, 1:env.n_stores + 1] / env.cap_truck)).T
   # phi[8, :] = (-1*np.ceil(action[:, 1:env.n_stores + 1] / env.cap_truck)).T
    phi[9, :] = state[1] + (action[:, 1] - (state[2] + state[3]) / 2).T > env.cap_truck
    phi[10, :] = np.logical_and((state[1] + (action[:, 1] - (state[2] + state[3]) / 2).T) <= env.cap_truck, (state[1] + (action[:, 1] - (state[2] + state[3]) / 2).T) >= 0)
    phi[11, :] = state[1] + (action[:, 1] - (state[2] + state[3]) / 2).T > 2*env.cap_truck
    phi[12, :] = np.ceil(action[:, 1] / env.cap_truck) - action[:, 1] / env.cap_truck
    return phi

class approximate_sarsa_agent(object):
    
    def __init__(self, env):
        # Set environment
        self.env = env
        
        # Initialize theta random
        self.theta = np.random.rand(13)#np.ones(env.n_stores+2)#
        #self.theta /= np.sum(self.theta)
        self.thetas = [self.theta.copy()]
        # Initialize the stepsize alpha
        self.alpha = 0.02 #1#0.02
    
        # Initialize agent parameters for stepsize rule
        self.n=1
        self.stepsizes = [1]
        
        # Initialize environment params
        #self.env_params = (env.prod_cost, env.price, env.n_stores)
        self.env_params = (env.prod_cost, env.store_cost, env.price, env.n_stores)

    def get_action(self, state, epsilon=0.2):
        # Find all possible actions
        actions = np.array(self.env.action_space()) # TODO: Should be changed in supply_chain to return numpy array

        # With probability epsilon, select random action
        if np.random.rand() < epsilon:
            return actions[np.random.randint(0, len(actions))]

        # With probability 1-epsilon, select greedy action
        return actions[np.argmax(np.dot(self.theta, phi(state, actions, self.env)))]
    
    def update(self, state, action, reward, state_new, action_new):
        # Calculate delta
        delta = reward + self.env.gamma * np.dot(self.theta, phi(state_new, action_new, self.env)) - np.dot(self.theta, phi(state, action, self.env))

        # Update theta
        self.theta += self.alpha * delta * phi(state, action, self.env)
        
        # Normalize theta to [0,1] (only relative weights should be important for policy selection)
        #self.theta /= np.sum(self.theta)
        #self.theta /= np.max(np.abs(self.theta))
        self.thetas.append(self.theta.copy())
        
        # Update learning rate alpha
    #    self.alpha = update_alpha(self.alpha, self.n) TODO diego 1/03 not updating alpha
        self.stepsizes.append(self.alpha)
        self.n+=1
        
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

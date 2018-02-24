#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 24 13:06:37 2018

@author: lukaskemmer
"""

import numpy as np

def phi(state, action):
    return np.hstack((state, action))

class approximate_sarsa_agent(object):
    
    def __init__(theta_size):
        # Initialize theta random
        self.theta = np.random.rand(theta_size)
    
    def get_action(self, state, epsilon=0.1):
        if np.random.rand() < epsilon:
            return 0 # TODO: Sample random action from environment
        return np.argmax(np.dot(self.theta, phi(state, actions))) # ToDo: Implement actions as matrix where each column is one possible action
        
    
    def update(self, state, action):
        pass
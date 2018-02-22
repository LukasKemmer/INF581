#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 21:16:21 2018

@author: lukaskemmer
"""

import numpy as np

class q_s_agent(object):
    
    def __init__(self, threshold, reorder_quantity):
        self.threshold = threshold
        self.reorder_quantity = reorder_quantity
        

    def get_action(self, state):
        '''
            Heuristic based on s_q policy
        '''
        # Initialize output and helper variable
        a = np.zeros(len(state))
        disposable_produce = state[0]
        
        # 1. Set actions for individual warehouses
        for i in np.arange(1, len(state)):
            # Check if current stock is below replenishment threshold
            if state[i] < self.threshold[i]:
                # Replenish with reorder quantity if possible
                a[i] = min(self.reorder_quantity[i], disposable_produce)

                # Update remaining disposable produce
                disposable_produce -= a[i]
    
        # 2. Set action for factory
        if disposable_produce < self.threshold[0]:
            a[0] = self.reorder_quantity[0]

        return a.astype(np.int)

    def update(self, state, action):
        '''
            update function not required for q-s-policy
        '''
        pass
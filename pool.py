# -*- coding: utf-8 -*-
"""
Created on Mon Sep  5 15:47:11 2016

@author: aurelia
"""
import numpy as np
from collections import deque

class Pool(object):
    """
    pool where we save trajectories...
    """
    
    def __init__(self,total_count,var):
        self.total_count = total_count
        self.var = var #list of the variables to save
        # it can be ob, action, reward, length
        self.reini()
        
    def reini(self):
        """
        delete all trajectories
        """
        self.actual_count = 0
        self.values = {}
        for var in self.var:
            self.values[var] = deque()
        
    def add(self,exp):
        """
        add a trajectory
        """
        for var in self.var:
            self.values[var].append(exp[var])
        if self.actual_count >= self.total_count:
            for var in self.var :
                self.values[var].popleft()
        else:
            self.actual_count += 1
            
    def minibatch(self,batch_size):
        """ 
        sample a minibatch of trajectories
        """
        batch_size = min(self.actual_count,batch_size)
        rand = np.random.choice(self.actual_count,batch_size,replace=False)
        r = {}
        for var in self.var :
            r[var] = np.stack([self.values[var][i] for i in rand], axis=0)
        return r
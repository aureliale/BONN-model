# -*- coding: utf-8 -*-
"""
Created on Fri Jun 24 16:21:59 2016

@author: aurelia
"""
import numpy as np

class Baseline(object):
    def __init__(self):
        raise NotImplementedError 
        
    def fit(self):
        pass
    
    def predict(self):
        raise NotImplementedError 
        
    def set_session(self,session):
        self.session = session
       
class MovingAverage(Baseline):
    """ 
    time-dependent moving average baseline
    """
    def __init__(self,config,b=0.4):
        self.meanRewards = np.zeros(config.max_length)
        self.b = b
        self.max_length = config.max_length
    
    def fit(self,rewards):
        pad = self.max_length - len(rewards)
        meanRewards = np.pad(rewards,(0,pad),'constant',constant_values=(0))      
        self.meanRewards = (1-self.b)*self.meanRewards + self.b*meanRewards
    
    def predict(self,trajs):
        if np.size(np.shape(trajs["reward"]))>1:
            r = []
            for le in trajs["length"]:
                b = self.meanRewards[:le]
                pad = self.max_length - le
                b = np.pad(b,(0,pad),'constant',constant_values=(0))
                r.append(b)
        else:
            r = self.meanRewards
        return r
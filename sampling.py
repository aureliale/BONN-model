# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 13:21:01 2016

@author: aurelia
"""
import numpy as np

"""
sampling : when we have the probability of each action, how do we choose one
"""

class categorical_sample(object):
    """
    Sample from categorical distribution,
    specified by a vector of class probabilities
    """
    def __init__(self):
        pass
    
    def get_action(self,prob_n):
        return np.random.choice(np.size(prob_n),p=prob_n)

class best_sample_class(object):
    """ always sample the best action """
    def __init__(self):
        pass
 
    def get_action(self,prob_n):
        return prob_n.argmax()
        
def best_sample(prob_n):
        return prob_n.argmax()
        
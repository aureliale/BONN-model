# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 11:40:22 2016

@author: aurelia
"""

from scipy.signal import lfilter
import gym
import copy

import envs  

def get_env(config,stocha=None):
    if type(config)==str:
        env = config
        stocha = stocha
    else:
        env = config.env
        stocha = config.stocha
    if env=="rooms2":
        env = envs.Rooms2()
    elif env=="rooms4":
        env = envs.Rooms4()
    elif env=="rooms9":
        env = envs.Rooms9()
    elif env=="rooms2stocha":
        env = envs.Rooms2Stocha(stocha)
    elif env=="rooms4stocha":
        env = envs.Rooms4Stocha(stocha)
    elif env=="rooms9stocha":
        env = envs.Rooms9Stocha(stocha)
    elif env == "cartpolestocha":
        env = envs.CartPoleStochaEnv(stocha)
    else:
        env = gym.make(env)  
    return env

def isBONN(config):
    if config.model=="BONN" or config.model=="D-BONN":
        return True
    else:
        return False

def isDBONN(config):
    if config.model=="D-BONN":
        return True
    else:
        return False
        
def discount(x, gamma):
    """
    Given vector x, computes a vector y:
    y[i] = x[i] + gamma * x[i+1] + gamma^2 x[i+2] + ...
    """
    return lfilter([1],[1,-gamma],x[::-1], axis=0)[::-1] 

def fit_config(config,env):
    
    try:
        config.input_dim = env.state.size 
        config.x_dim = env.state.size
    except:
        config.input_dim = env.observation_space.shape[0]
        config.x_dim = env.observation_space.shape[0]
    config.output_dim = env.action_space.n
    
    if isBONN(config) and (config.use_x=="True" or config.use_x==True) :
        try:
            config.y_dim = env.y_dim
        except:
            config.y_dim = 0
        config.x_dim = config.input_dim - config.y_dim
    elif isBONN(config):
        config.y_dim = copy.deepcopy(config.x_dim)
        config.x_dim = 0
        
    try:    
        config.max_length = env.spec.timestep_limit
    except:
        try :
            config.max_length = env.timestep_limit
        except:
            config.max_length = 200
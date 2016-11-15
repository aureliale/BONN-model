# -*- coding: utf-8 -*-
"""
Created on Fri Jun 24 15:12:47 2016

@author: aurelia
"""
import numpy as np
import copy

class PolicyRecurrent(object):
    
    def __init__(self):
        raise NotImplementedError
    
    def _acts_mask(self,a,env):
        var = np.zeros([1,1,env.action_space.n])
        var[0,0,a]=1
        return var
        
    def get_action(self, obs):
        raise NotImplementedError         
        
    def reini(self):
        """ 
        if the policy is recurrent, must define "reini" to 
        initialize the state of the recurrent network
        (to use get_action at the beginning of an episode )
        """
        raise NotImplementedError  
    
    def learn(self, minibatch):
        """
        an epoch of learning on minibatch
        """
        raise NotImplementedError
        
    def set_session(self,session):
        self.session = session
        
    def get_traj(self, env, episode_max_length, render=False):
        """
        run the policy for one trajectory and return it
        """
        
        ob = env.reset()
        obs = [copy.deepcopy(ob)]
        rews = []
        acts = []
        self.reini()
        for i in range(episode_max_length):          
            ob1 = np.reshape(ob,[1,1,-1])
            a = self.get_action(ob1)  
            (ob, rew, done, _) = env.step(a)
            obs.append(copy.deepcopy(ob))
            rews.append(copy.deepcopy(rew))
            acts.append(copy.deepcopy(a))
            if done: 
                break
            if render: 
                env.render()
        return {"reward" : rews,
                "ob" : obs,
                "action" : acts,
                "length" : i+1
                }
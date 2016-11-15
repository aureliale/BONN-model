# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 11:43:17 2016

@author: aurelia
"""
import numpy as np   

from tools_RL import discount
from pool import Pool
import tools_RL
         
class Agent(object):

    """
    Base agent; discrete action space
    """

    def __init__(self, policy, baseline, ob_space, action_space, config, **usercfg):
        self.policy = policy     
        self.baseline = baseline
        self.config = config
        self.pool = Pool(config.n_batch, var = self.varpool())

    def act(self, obs):
        """
        choose an action
        """
        action = self.policy.get_action(obs)
        return action

    def learn(self, env, observer):
        """
        learning algorithm: take trajectories then learn from them
        """
        cfg = self.config
        
        #for observer
        trajs_resume = {}
        rewardsEval = []
        if tools_RL.isBONN(self.config):
            sigmaEval = []
            
        self.epoch = 0
        for iteration in range(cfg.iter_max):
            
            #update cost lambda if necessary
            self.update_lambd()
            
            #sample new trajectories and save in pool
            for s in range(cfg.n_batch):
                traj = self.policy.get_traj(env, cfg.max_length)
                self.add_traj_to_pool(traj)
                
            #sample trajectories from pool then learn from them
            trajs = self.pool.minibatch(cfg.n_batch)            
            self.policy.learn(trajs)

            #observer: sample trajectories with current policy then evaluate them
            if iteration%cfg.obs_each_n_trajs == 0:
                print (iteration)
                for a in range(cfg.nb_iter_for_obs):
                    trajEval = self.policy.get_traj(env, cfg.max_length)
                    rewardsEval.append(np.sum(trajEval["reward"]))
                    if tools_RL.isBONN(self.config):
                        sigmaEval.append(np.sum(trajEval['sigma'])/trajEval['length'])
                
                trajs_resume['reward'] = np.array(rewardsEval) 
                if tools_RL.isBONN(self.config):
                    trajs_resume['sigma'] = np.array(sigmaEval) 
                observer.observe(iteration ,trajs_resume)                    
                trajs_resume = {}
                rewardsEval = []
                if tools_RL.isBONN(self.config):
                    sigmaEval = []
                    
            self.epoch = iteration        
                
    def varpool(self):
        """ variables to save in the pool """
        return ["ob","action","reward"]  
    
    def update_lambd(self):
        pass
        
    def add_traj_to_pool(self,traj):
        """
        non-reccurrent agent : save all observations separatly
        """
        rews = traj["reward"]
        if self.config.discount :
            rews = discount(rews, self.config.gamma) 
        if self.baseline:    
            self.baseline.fit(traj,rews)     
            rews = rews - self.baseline.predict(traj)[:len(rews)]
            
        for o,a,r in zip(traj["ob"][:-1],traj["action"],rews):
            traj = {"ob":o.flatten(),"action":a,"reward":r}            
            self.pool.add(traj)


class AgentRecurrent(Agent):

    """
    Reccurrent agent: save whole trajectory in pool
    """

    def varpool(self):
        """ var to save for the pool """
        return ["ob","action","reward","length"]
        
    def add_traj_to_pool(self,traj):
        """
        save whole trajectory in pool (with length of trajectory)
        """
        pad = self.config.max_length - traj["length"]
        
        obs = [o.flatten() for o in traj['ob']]
        obs = np.stack(obs[:-1],axis=0)
        obs = np.pad(obs,((0,pad),(0,0)),'constant',constant_values=(0)) 
            
        acts = np.pad(traj["action"],(0,pad),'constant',constant_values=(0)) 
        
        rews = traj["reward"]
        if self.config.discount :
            rews = discount(rews, self.config.gamma)   
        if self.baseline:    
            self.baseline.fit(rews)   
        rews = np.pad(rews,(0,pad),'constant',constant_values=(0)) 
        
        length = traj["length"]
        
        traj = {"ob":obs,"action":acts,"reward":rews,"length":length}            
        self.pool.add(traj)

class AgentBONN(Agent):

    """
    BONN (Budgeted options neural network) agent.
    Need to update self.lambd and a baseline cost.
    """
    
    def __init__(self, policy, baseline, baseline_cost, ob_space, action_space, config, **usercfg):
        super().__init__(policy, baseline, ob_space, action_space, config, **usercfg)
        max_lambd = self.config.lambd
        self.exploration_epoch_min = self.config.exploration_epoch_min        
        self.exploration_epoch_nb = self.config.exploration_epoch_nb
        if self.exploration_epoch_nb==0:
            self.lambd_update = max_lambd
        else:
            self.lambd_update = max_lambd / self.exploration_epoch_nb
        self.lambd = 0
        self.baseline_cost = baseline_cost
        

    def varpool(self):
        """ variables to save in the pool """
        if tools_RL.isDBONN(self.config):
            return ["ob","action","reward","length", "sigma","alpha"]
        else:
            return ["ob","action","reward","length", "sigma"]
        
    def update_lambd(self):
        if self.epoch>self.exploration_epoch_nb+self.exploration_epoch_min:
            pass
        elif self.epoch>=self.exploration_epoch_min:
            self.lambd += self.lambd_update
        else:
            pass
               
    def add_traj_to_pool(self,traj):
        pad = self.config.max_length - traj["length"]
        
        obs = [o.flatten() for o in traj['ob']]
        obs = np.stack(obs[:-1],axis=0)
        obs = np.pad(obs,((0,pad),(0,0)),'constant',constant_values=(0)) 
        rews = traj["reward"]   
        
        if self.config.discount :
            rews = discount(rews, self.config.gamma)    
            
        if self.baseline:    
            self.baseline.fit(rews)  
            
        if self.config.lambd!=0:
            cost = [-self.lambd * c for c in traj["sigma"]]
            if self.config.discount :
                cost = discount(cost, self.config.gamma)
            self.baseline_cost.fit(cost) 
            rews = [r+c for (r,c) in zip(rews,cost)] 
            
        rews = np.pad(rews,(0,pad),'constant',constant_values=(0)) 
        
        length = traj["length"]
        
        if tools_RL.isDBONN(self.config):
            traj = {"ob":obs,"action":traj["action"],"reward":rews,"length":length, "sigma":traj["sigma"], "alpha":traj["alpha"]}               
        else:
            traj = {"ob":obs,"action":traj["action"],"reward":rews,"length":length, "sigma":traj["sigma"]}            

        self.pool.add(traj) 

                    
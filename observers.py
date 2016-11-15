# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 13:58:19 2016

@author: aurelia
"""
import os.path
import pickle
from copy import deepcopy
import tools_RL

class Observer(object):
    def __init__(self,config):
        self.config = config
    
    def parameters(self):
        params = []
        params.append('iteration')
        params = params + ['min_rew','max_rew','mean_rew','std_rew']
        if tools_RL.BONN(self.config):
                var = 'sigma'
                params += [':min_'+var,':max_'+var,':mean_'+var,':std_'+var]
        return params         

class ObserverCSV(Observer):
    def __init__(self,config):
        self.directory = config.directory
        self.config = config
        self.filename = self.directory+"/"+config.filename
                
        os.makedirs(self.directory,exist_ok=True) #dont create directory if it already exists        
        
        #change the name if the file already exists
        i = 1
        filename = deepcopy(self.filename)
        while os.path.isfile(filename+".csv"):
            filename = "%sv%s" %(self.filename,i)
            i += 1
        self.filename = filename
        if i>1:
            config.filename += "v%s"%(i-1)
        
        #write names of the parameters observed on the first line
        with open(self.filename+".csv",'w') as file:
            file.write(",".join(self.parameters())+"\n")       
            
        #write a file with same name +"_infos" with the config
        with open(self.filename+"_infos.pkl",'wb') as file:
            pickle.dump(config,file)  
                
    
    def observe(self,iteration,obs):
        rew = obs['reward']
        print ("iteration : %i" %iteration)
        write = "%s," %iteration
        write += "%s,%s,%s,%s," %(rew.min(),rew.max(),rew.mean(),rew.std())
        if tools_RL.BONN(self.config):
            val = obs['sigma']
            write += "%s,%s,%s,%s," %(val.min(),val.max(),val.mean(),val.std())
        write = write[:-1]
        with open(self.filename+".csv",'a') as file:
            file.write(write+"\n")

class ObserverConsole(Observer):            
    
    def observe(self,iteration,obs):
        rew = obs['reward']
        print ("-----------------")
        print ("Iteration: \t %i"%iteration)
        print ("MaxRew: \t %s"%rew.max())
        print ("MeanRew: \t %s +- %s"%(rew.mean(), rew.std()))

        if tools_RL.isBONN(self.config):
            sigma = obs['sigma']
            print ("Mean sigma: \t %s +- %s"%(sigma.mean(), sigma.std()))
        print ("-----------------")  
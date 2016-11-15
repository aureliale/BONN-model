# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 11:00:12 2016

@author: aurelia
"""
import tensorflow as tf
import argparse

from tools_RL import fit_config
from agents import AgentRecurrent, AgentBONN
from policies.bonn_policy import BONNPolicy
from policies.gru_policy import GRUPolicy
from observers import ObserverCSV, ObserverConsole
import baselines as baselines
import sampling as sampl 
import tools_RL



def main():
    
    parser = argparse.ArgumentParser(description='Running expriment (recurrent policy)')
    
    parser.add_argument('--log', choices=["csv","console"], help="Save results to a file or print in console",required=True)
    parser.add_argument('--directory', help="Directory where to save results")
    parser.add_argument('--filename', help="Name of the file to save results")

    parser.add_argument('--env', help="Environment to use",required=True)
    parser.add_argument('--stocha', type=float, help="stochasticity")
    
    parser.add_argument('--model', choices=["GRUReinforce","BONN","D-BONN"], help="Wich model to use",required=True)
    parser.add_argument('--use_x', type=str, help="_")
    parser.add_argument('--num_hidden', type=int, help="number of hidden units")
    parser.add_argument('--nH', type=int, help="number of different discretes policies for model:D-BONN")
    parser.add_argument('--representation', help="representation of x_data")
    parser.add_argument('--representation_goal', help="representation of y_data")    
    parser.add_argument('--gamma', type=float, help="Discount rate for reward",default=0.99)
    
    parser.add_argument('--lr', type=float, help="Learning rate for policy",required=True)
    parser.add_argument('--weight_regu', type=float, help="weight l2 regularization for learning",default=0.)
    parser.add_argument('--activation', choices=["tanh","sigmoid","softsign","relu","None","none"], help="activation to use in RNN cell",default="tanh")
    
    parser.add_argument('--lambd', type=float, help="(maximum) cost to observe y")
    parser.add_argument('--exploration_epoch_min', type=int, help="when to begin to use a cost for y")
    parser.add_argument('--exploration_epoch_nb', type=int, help="number of epoch to increase the cost to lambd")  
    
    parser.add_argument('--n_batch', type=int, help="Number of trajectories for one epoch",default=32)
    parser.add_argument('--iter_max', type=int, help="Maximum number of learning iteration",default=1000)
    parser.add_argument('--nb_iter_for_obs', type=int, help="number of trajs used for observations",default=10)
    parser.add_argument('--obs_each_n_trajs', type=int, help="observe rewards (+sigma) each obs_each_n_trajs iterations",default=100)
    
    
    config = parser.parse_args()   
    
    env = tools_RL.get_env(config)
    fit_config(config,env)    
    
    sampling = sampl.categorical_sample()
    baseline = baselines.MovingAverage(config,0.4)        
    if tools_RL.isBONN(config):
        baseline_cost = baselines.MovingAverage(config,0.6)        
    config.discount=True
        
    if config.model=="GRUReinforce":
        policy = GRUPolicy(config,sampling,baseline)
    elif config.model=="BONN" :
        policy = BONNPolicy(config,sampling,baseline, baseline_cost)
    elif config.model=="D-BONN" :
        policy = BONNPolicy(config,sampling,baseline, baseline_cost)
       
    if config.log=="console":
        observer = ObserverConsole(config)
    elif config.log == "csv":
        observer = ObserverCSV(config)        
        
    NUM_THREADS = 1
    configSession = tf.ConfigProto(intra_op_parallelism_threads=NUM_THREADS)
    with tf.Session(config=configSession) as sess:
        policy.set_session(sess)
        baseline.set_session(sess)
        if  tools_RL.isBONN(config):
            baseline_cost.set_session(sess)
            agent = AgentBONN(policy, baseline, baseline_cost, env.observation_space, env.action_space, config=config)
        else:
            agent =  AgentRecurrent(policy, baseline, env.observation_space, env.action_space, config=config)     
            
        sess.run(tf.initialize_all_variables())
        agent.learn(env, observer)

if __name__ == "__main__":
    main()
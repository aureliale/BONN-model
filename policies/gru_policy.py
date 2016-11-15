# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 11:36:33 2016

@author: aurelia

Recurrent policy gradient [Wiestra Schmidhuber 2007-2009]
"""

import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
from tensorflow.python.ops.rnn import dynamic_rnn

from policies.base import PolicyRecurrent
import tools

class Model(object):
    def __init__(self,config,is_training=True,scope=None):
        with tf.variable_scope(scope or type(self).__name__) as vs:
            input_dim=config.input_dim
            output_dim=config.output_dim
            max_length=config.max_length
            if not is_training:
                max_length = 1            
            num_hidden=config.num_hidden
            lr=config.lr
            weight_regu = config.weight_regu
            activation = tools.get_activation(config.activation)
            
            # Placeholders
            if is_training:
                self.targets = tf.placeholder(tf.int64, [None, max_length],"targets")
                self.adv_data = tf.placeholder(tf.float32, [None, max_length],"adv")
                self.seq_length = tf.placeholder(tf.int64, [None],"seq_length") 
                self.mask = tf.placeholder(tf.float32, [None, max_length], "mask")
            self.input_data = tf.placeholder(tf.float32, [None, max_length,input_dim],"input")
            data = self.input_data  
            
            #data representation
            data = tf.reshape(data, [-1, input_dim])
            data = tools.representation(data, config.representation, is_training, scope="input_representation")
            data = tf.reshape(data, [-1, max_length, int(data.get_shape()[1])])    
            
            #recurrent network
            self.cell = tf.nn.rnn_cell.GRUCell(num_hidden,activation=activation)
            if is_training:
                data, state = dynamic_rnn(self.cell, data, sequence_length=self.seq_length,dtype=tf.float32,scope="RNN")            
            else:    
                self.initial_state = tf.placeholder(tf.float32,[None,self.cell.state_size],"ini_state")
                data, state = dynamic_rnn(self.cell, data, initial_state=self.initial_state ,scope="RNN")            
                self.state = state
            
            #probabilities of actions
            data = tf.reshape(data, [-1, num_hidden])
            data = slim.fully_connected(data, output_dim, activation_fn=None, scope="prediction")
            self.predictions = tf.reshape(tf.nn.softmax(data),[-1, max_length, output_dim])
            
            #training
            if is_training:
                variable_list = [v for v in tf.trainable_variables() if v.name.startswith(vs.name)]  
                loss = tools.loss_policy_gradient_recurrent(data, self.targets, self.adv_data, self.seq_length, self.mask, weight_regu, variable_list)
                self.train_step = tools.training(loss, lr, variable_list)
                                      
    def iniState(self):
        return np.zeros([1,self.cell.state_size])
  
class GRUPolicy(PolicyRecurrent):
    """ 
    recurrent policy gradient with GRU cell
    """
    
    def __init__(self,config,sampling,baseline):        

        self.sampling = sampling
        self.baseline = baseline
        self.config = config

        with tf.variable_scope("model",reuse=False):
            self.mTrain = Model(config,is_training=True)
        with tf.variable_scope("model",reuse=True):
            self.mTest = Model(config,is_training=False)
            
        self.reini()
        
    def get_action(self, obs):
        prob, state = self.session.run([self.mTest.predictions,self.mTest.state],feed_dict={
                      self.mTest.input_data:obs,self.mTest.initial_state:self.state})
        action = self.sampling.get_action(prob[0][0])
        self.state = state
        return action
        
    def reini(self):
        self.state = self.mTest.iniState()
    
    def learn(self, trajs):        
        rewards = trajs["reward"]
        if self.baseline:
            baseline = self.baseline.predict(trajs)
            advs= [r-b for r,b in zip(rewards,baseline)] 
        else:
            advs = rewards
            
        # olicy gradient update step
        self.session.run(self.mTrain.train_step,feed_dict={
                    self.mTrain.input_data:trajs["ob"],
                    self.mTrain.targets:trajs["action"],self.mTrain.adv_data:advs,
                    self.mTrain.seq_length:trajs["length"], 
                    self.mTrain.mask:tools.get_mask(self.config, trajs)})
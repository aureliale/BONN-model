# -*- coding: utf-8 -*-
"""
Created on Fri Jun 24 15:28:18 2016

@author: aurelia
"""
import tensorflow as tf
import numpy as np
from tensorflow.python.ops.rnn import dynamic_rnn
import tensorflow.contrib.slim as slim
import copy
from scipy.ndimage.interpolation import shift

from tensorflow.python.ops.rnn_cell import RNNCell
from tensorflow.python.ops.rnn_cell import _linear as linear

from policies.base import PolicyRecurrent
import tools

class BONNCell(RNNCell):
  """
  recurrent cell pour BONN model
  """

  def __init__(self, num_units,y_dim, x_dim,activation=tf.nn.tanh,is_training=True):
    self._num_units = num_units
    self.activation = activation
    self.y_dim = y_dim
    self.x_dim = x_dim

    self.is_training = is_training
    
  @property
  def state_size(self):
    return self._num_units

  @property
  def output_size(self):
      if self.is_training:
          return self._num_units+1+2
      else:
          return self._num_units+1

  def __call__(self, inputs, state, scope=None):
    with tf.variable_scope(scope or type(self).__name__):  
      #y = tf.slice(inputs, [0,0],[-1,self.y_dim])
      x = tf.slice(inputs, [0,self.y_dim],[-1,self.x_dim])
      if self.is_training: 
          sigma = tf.slice(inputs, [0,self.y_dim+self.x_dim],[-1,1])
          inputs = tf.slice(inputs, [0,0],[-1,self.y_dim+self.x_dim])
       
      with tf.variable_scope("Gates"):  
        u = tf.concat(1,[x,state])
        u_proba = slim.fully_connected(u, 2, activation_fn=None, scope="u_gate",biases_initializer=tf.constant_initializer(value=1.))#,trainable=False)
        if not self.is_training:
            u = tf.nn.softmax(u_proba)            
            u = tf.slice(u, [0,1],[-1,1])
            uniform = tf.random_uniform(tf.shape(u))
            sample = tf.less(uniform, u) 
            sample=tf.to_float(sample)
            u = sample
        else:
            u = sigma
        u = tf.stop_gradient(u)
        
      with tf.variable_scope("Candidate_with_y"):
          c = linear([inputs], self._num_units, True) 
          if self.activation:
              c = self.activation(c)              
             
      with tf.variable_scope("RecurrentGates"):  
        ra, ua = tf.split(1,2,linear([x, state],2*self._num_units, True, bias_start=1.0) )
        ra, ua = tf.nn.sigmoid(ra), tf.nn.sigmoid(ua)
      with tf.variable_scope("Candidate_with_x"):
          ca = linear([x,ra*state], self._num_units, True)  
          if self.activation:
              ca = self.activation(ca)
          ca = ua * state + (1-ua) * ca
          
      new_h = (1-u) * ca + u * c
    
      if self.is_training:
          return tf.concat(1,[new_h,u,u_proba]), new_h #output, new state    
      else:
          return tf.concat(1,[new_h,u]), new_h 


class BONNDiscretCell(RNNCell):
  """
      recurrent cell pour D-BONN model
  """

  def __init__(self, num_units,y_dim, x_dim, nH,activation=tf.nn.tanh,is_training=True):
    self._num_units = num_units
    self.activation = activation
    self.y_dim = y_dim
    self.x_dim = x_dim
    self.nH = nH
    
    self.is_training = is_training
    
  @property
  def state_size(self):
    return self._num_units

  @property
  def output_size(self):
      if self.is_training:
          return self._num_units+1+2+1+self.nH
      else:
          return self._num_units+1+1

  def __call__(self, inputs, state, scope=None):
    with tf.variable_scope(scope or type(self).__name__):  
      #y = tf.slice(inputs, [0,0],[-1,self.y_dim])
      x = tf.slice(inputs, [0,self.y_dim],[-1,self.x_dim])
      if self.is_training: 
          sigma = tf.slice(inputs, [0,self.y_dim+self.x_dim],[-1,1])
          alpha_n = tf.slice(inputs, [0,self.y_dim+self.x_dim+1],[-1,1])
          alpha_n = tf.to_int32(alpha_n)
          inputs = tf.slice(inputs, [0,0],[-1,self.y_dim+self.x_dim])

      with tf.variable_scope("Gates"):  
        u = tf.concat(1,[x,state])
        u_proba = slim.fully_connected(u, 2, activation_fn=None, scope="u_gate",biases_initializer=tf.constant_initializer(value=1.))#,trainable=False)
        if not self.is_training:
            u = tf.nn.softmax(u_proba)            
            u = tf.slice(u, [0,1],[-1,1])
            uniform = tf.random_uniform(tf.shape(u))
            sample = tf.less(uniform, u) 
            sample=tf.to_float(sample)
            u = sample
        else:
            u = sigma
        u = tf.stop_gradient(u)
           
      with tf.variable_scope("vecteurs_latents"):
          h = tf.get_variable("h",shape=[self.nH,self._num_units])
      with tf.variable_scope("alpha_gate"): #attention over discrete set of policies
        alpha_proba = slim.fully_connected(inputs, self.nH, activation_fn=None, scope="u_gate",biases_initializer=tf.constant_initializer(value=1.))#,trainable=False)
        if not self.is_training:
            alpha = alpha_proba
            alpha_n = tf.multinomial(alpha, 1)
        alpha_n = tf.reshape(alpha_n,[-1])
        alpha = tf.one_hot(alpha_n,self.nH,axis=-1)
        alpha = tf.stop_gradient(alpha)
      with tf.variable_scope("Candidate_with_y"): 
          c = tf.matmul(alpha,h)
          if self.activation:
              c = self.activation(c)
          c = linear([c,x],self._num_units, True)
                                 
      with tf.variable_scope("RecurrentGates"):  
        ra, ua = tf.split(1,2,linear([x, state],2*self._num_units, True, bias_start=1.0) )
        ra, ua = tf.nn.sigmoid(ra), tf.nn.sigmoid(ua)
      with tf.variable_scope("Candidate_with_x"):
          ca = linear([x,ra*state], self._num_units, True) 
          if self.activation:
              ca = self.activation(ca)
          ca = ua * state + (1-ua) * ca
              
      new_h = (1-u) * ca + u * c        
      
      alpha_n = tf.to_float(tf.reshape(alpha_n,[-1,1]))
      if self.is_training:
          return tf.concat(1,[new_h,u,u_proba,alpha_n,alpha_proba]), new_h #output, new state    
      else:
          return tf.concat(1,[new_h,u,alpha_n]), new_h

class Model(object):
    def __init__(self,config,is_training=True, training_gate=False,scope=None):
        with tf.variable_scope(scope or type(self).__name__) as vs:
            input_dim = config.input_dim #all observations
            x_dim=config.x_dim 
            y_dim = config.y_dim 
            output_dim=config.output_dim
            max_length=config.max_length
            if not is_training:
                max_length = 1               
            num_hidden=config.num_hidden
        
            lr=config.lr
            weight_regu = config.weight_regu
            activation = tools.get_activation(config.activation)
                
            # Placeholders
            self.input_data = tf.placeholder(tf.float32, [None, max_length,input_dim],"obs")
            self.actions_data = tf.placeholder(tf.int64, [None, max_length],"last_actions")   
            if is_training:
                self.targets = tf.placeholder(tf.int64, [None, max_length],"targets")
                self.adv_data = tf.placeholder(tf.float32, [None, max_length],"adv")
                self.seq_length = tf.placeholder(tf.int32, [None],"seq_length")                
                self.sigma_data = tf.placeholder(tf.int64, [None, max_length],"sigma")
                if config.model=="D-BONN":
                    self.alpha_data = tf.placeholder(tf.int64, [None, max_length],"alpha")                    
                self.mask = tf.placeholder(tf.float32, [None, max_length], "mask")
  
            #data representations 
            # x      
            if x_dim>0:
                x_data = tf.slice(self.input_data, [0,0,0],[-1,-1,x_dim])
                y_data = tf.slice(self.input_data, [0,0,x_dim],[-1,-1,y_dim])
            
                x_data = tf.reshape(x_data, [-1, x_dim])            
                x_data = tools.representation(x_data, config.representation, is_training, scope="input_representation")
                x_dim = int(x_data.get_shape()[1])
                x_data = tf.reshape(x_data, [-1, max_length, x_dim]) 
            else:
                y_data = self.input_data
            # y
            y_data = tf.reshape(y_data, [-1, y_dim])            
            y_data = tools.representation(y_data, config.representation_goal, is_training, scope="instr_representation")
            y_dim = int(y_data.get_shape()[1])
            y_data = tf.reshape(y_data, [-1, max_length, y_dim])                         
            # action
            actions_data = tf.one_hot(self.actions_data, output_dim, axis=-1, dtype=tf.float32)
            if x_dim>0:
                data = tf.concat(2,[y_data,x_data,actions_data])
            else:
                data = tf.concat(2,[y_data,actions_data])
            x_dim += output_dim
                
            # recurrent network
            if config.model=="BONN":
                self.cell = BONNCell(num_hidden, y_dim=y_dim, x_dim=x_dim,activation=activation, is_training=is_training)
            elif config.model=="D-BONN":
                self.cell = BONNDiscretCell(num_hidden, y_dim=y_dim, x_dim=x_dim, nH=config.nH,activation=activation, is_training=is_training)
              
            if is_training:
                sigma_data = tf.reshape(self.sigma_data,[-1,max_length,1])
                sigma_data = tf.to_float(sigma_data)
                if config.model=="D-BONN":
                    alpha_data = tf.reshape(self.alpha_data,[-1,max_length,1])
                    alpha_data = tf.to_float(alpha_data)
                    data = tf.concat(2,[data,sigma_data,alpha_data])
                else:
                    data =  tf.concat(2,[data,sigma_data])
                outputs, state = dynamic_rnn(self.cell, data, sequence_length=self.seq_length,dtype=tf.float32,scope="RNN")            
            else:
                self.initial_state = tf.placeholder(tf.float32,[None,self.cell.state_size],"ini_state")
                outputs, state = dynamic_rnn(self.cell, data, initial_state=self.initial_state ,scope="RNN")            
                self.state = state                                                             
            
            if config.model=="BONN":
                self.sigma = tf.slice(outputs,[0,0,num_hidden],[-1,-1,1])
                u_proba = tf.slice(outputs,[0,0,num_hidden+1],[-1,-1,2])
            elif config.model=="D-BONN":
                if is_training:
                    self.sigma = tf.slice(outputs,[0,0,num_hidden],[-1,-1,1])
                    u_proba = tf.slice(outputs,[0,0,num_hidden+1],[-1,-1,2])
                    self.alpha = tf.slice(outputs,[0,0,num_hidden+1+2],[-1,-1,1])
                    alpha_proba = tf.slice(outputs,[0,0,num_hidden+1+2+1],[-1,-1,config.nH])
                else:
                    self.sigma = tf.slice(outputs,[0,0,num_hidden],[-1,-1,1])
                    self.alpha = tf.slice(outputs,[0,0,num_hidden+1],[-1,-1,1])
                
            #probabilities of actions                
            data = tf.slice(outputs,[0,0,0],[-1,-1,num_hidden])                       
            data = tf.reshape(data, [-1, num_hidden])                
            data = slim.fully_connected(data, output_dim, activation_fn=None, scope="prediction")
            self.prediction = tf.reshape(tf.nn.softmax(data),[-1, max_length, output_dim])  
                  
            if is_training:
                variable_list = [v for v in tf.trainable_variables() if v.name.startswith(vs.name)]  

                u_proba = tf.reshape(u_proba,[-1,2])
                adv_data_u = self.adv_data
                
                loss = tools.loss_policy_gradient_recurrent(data, self.targets, adv_data_u, self.seq_length, self.mask, weight_regu, variable_list)
                loss += tools.loss_policy_gradient_recurrent(u_proba, self.sigma_data, adv_data_u, self.seq_length, self.mask)                                
                if config.model=="D-BONN":
                    alpha_proba = tf.reshape(alpha_proba,[-1,config.nH])
                    loss += tools.loss_policy_gradient_recurrent(alpha_proba, self.alpha_data, adv_data_u, self.seq_length, tf.to_float(self.sigma_data))   
                
                self.train_step = tools.training(loss, lr, variable_list)
                
    def iniState(self):
        return np.zeros([1,self.cell.state_size])

class BONNPolicy(PolicyRecurrent):
    
    def __init__(self,config,sampling,baseline, baseline_cost):      

        self.sampling = sampling
        self.baseline = baseline
        self.baseline_cost = baseline_cost
        self.config = config

        with tf.variable_scope("model",reuse=False):
            self.mTrain = Model(config,is_training=True)
        with tf.variable_scope("model",reuse=True):
            self.mTest = Model(config,is_training=False)         
            
        self.reini()        
        
    def get_action(self, obs):       
        data_obs = obs
        actions_data = self.last_action
        model = self.mTest
              
        if self.config.model=="BONN":
            prob, state, sigma = self.session.run([model.prediction,model.state, model.sigma],feed_dict={
                      model.actions_data:actions_data,model.input_data:data_obs,model.initial_state:self.state})
            alpha = None
        elif self.config.model=="D-BONN":
            prob, state, sigma, alpha = self.session.run([model.prediction,model.state, model.sigma, model.alpha],feed_dict={
                      model.actions_data:actions_data,model.input_data:data_obs,model.initial_state:self.state})
            alpha = alpha.flatten()[0]            
        sigma = sigma.flatten()[0]
        action = self.sampling.get_action(prob[0][0])
        self.state = state
        self.last_action = [[action]]
        return action, sigma, alpha
        
    def reini(self):        
        self.state = self.mTest.iniState()
        self.last_action = [[-1]]
            
    def learn(self, minibatch):
        trajs = minibatch       
        actions_data = shift(trajs["action"],[0,1],mode='constant',cval=-1.)
        data_obs = trajs["ob"]        
        rewards = trajs["reward"]

        if self.baseline:
            baseline = self.baseline.predict(trajs)
            advs= [r-b for r,b in zip(rewards,baseline)] 
        else:
            advs = rewards
            
        if self.config.lambd!=0:
            baseline = self.baseline_cost.predict(trajs)
            advs= [r-b for r,b in zip(advs,baseline)] 
            
        # policy gradient update step
        if self.config.model=="BONN":
            self.session.run(self.mTrain.train_step,feed_dict={
                        self.mTrain.actions_data:actions_data,
                        self.mTrain.input_data:data_obs,
                        self.mTrain.targets:trajs["action"],
                        self.mTrain.adv_data:advs,
                        self.mTrain.seq_length:trajs["length"], 
                        self.mTrain.sigma_data:trajs["sigma"], 
                        self.mTrain.mask:tools.get_mask(self.config, trajs)})
        elif self.config.model=="D-BONN":
            self.session.run(self.mTrain.train_step,feed_dict={
                        self.mTrain.actions_data:actions_data,
                        self.mTrain.input_data:data_obs,
                        self.mTrain.targets:trajs["action"],
                        self.mTrain.adv_data:advs,
                        self.mTrain.seq_length:trajs["length"], 
                        self.mTrain.sigma_data:trajs["sigma"], 
                        self.mTrain.alpha_data:trajs["alpha"], 
                        self.mTrain.mask:tools.get_mask(self.config, trajs)})           
                                
    def get_traj(self, env, episode_max_length, render=False):
        """
        run the policy for one trajectory and return it
        """
        
        ob = env.reset()
        obs = [copy.deepcopy(ob)]
        rews = []
        acts = []
        sigma = []
        alpha = []
        self.reini()
        for i in range(episode_max_length):       
            ob1 = np.reshape(ob,[1,1,-1])
            a, sigm, alph = self.get_action(ob1)
            (ob, rew, done, infos) = env.step(a)
            obs.append(copy.deepcopy(ob))
            rews.append(copy.deepcopy(rew))
            acts.append(copy.deepcopy(a))
            sigma.append(copy.deepcopy(sigm))
            alpha.append(copy.deepcopy(alph))
            
            if done: 
                break
            if render: 
                env.render()
        pad = episode_max_length - i -1
        if self.config.model=="D-BONN":
            alpha = np.pad(alpha,(0,pad),'constant',constant_values=(0))
        else:
            alpha = None
        traj = {"reward" : rews,
                "ob" : obs,
                "action" : np.pad(acts,(0,pad),'constant',constant_values=(0)),
                "length" : i+1,
                "sigma" : np.pad(sigma,(0,pad),'constant',constant_values=(0)),
                "alpha" : alpha,
                }
        return traj
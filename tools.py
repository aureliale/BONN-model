# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 11:40:57 2016

@author: aurelia
"""
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
    
def get_activation(activation_str):    
    if activation_str=="softsign":
        activation=tf.nn.softsign
    elif activation_str=="tanh":
        activation=tf.nn.tanh
    elif activation_str=="sigmoid":
        activation=tf.nn.sigmoid
    elif activation_str=="relu":
        activation=tf.nn.relu
    else:
        activation = None
    return activation
    
def representation(input_data, rep_string, is_training, scope="representation"):
    """ 
    return the output of the neural network corresponding to description "rep_string"
    "rep_string" is of the form linear:5:relu_linear:10:relu"
    """
    if rep_string == "None" or rep_string == None :
        return input_data
    else:
        with tf.variable_scope(scope):  
            net = input_data
            rep_list = rep_string.split("_")
            for r in rep_list:
                r_infos = r.split(":")
                # r_infos = linear:num_units_out:activation
                (net_type,num_outputs,activation) = r_infos
                num_outputs = int(num_outputs)
                activation = get_activation(activation)
                if net_type=="linear":
                    net = slim.fully_connected(net, num_outputs, activation)
                else:
                    raise NameError("net_type %s not known" %net_type)
            return net
        
def get_mask(config,trajs):
    assert np.size(np.shape(trajs["reward"]))>1, "dont need mask"
    mask = np.zeros((np.shape(trajs["reward"])[0], config.max_length))
    for i,le in enumerate(trajs["length"]):
        mask[i,:le]=1
    return mask
    
def loss_policy_gradient_recurrent(pred,targets,adv,seq_length,mask,weight_regu=None,var_list_regu=None):
    """ return the loss for recurrent policy gradient algorithm """
    flat_targets = tf.reshape(targets, [-1])
    flat_adv_data = tf.reshape(adv, [-1])
    flat_mask = tf.reshape(mask,[-1])
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(pred, flat_targets)
    loss = tf.reduce_sum(loss * flat_adv_data * flat_mask) / tf.to_float(tf.reduce_sum(seq_length))
    if weight_regu!=None and var_list_regu!=None:
        loss += weight_regu * tf.add_n([tf.nn.l2_loss(v) for v in var_list_regu])/len(var_list_regu)    
    return loss
    
def training(loss,lr,var_list):     
    """ return training_step """
    optimizer = tf.train.AdamOptimizer(lr)           
    grad = optimizer.compute_gradients(loss,var_list=var_list)  
    grad = [(tf.clip_by_value(g, -1., 1.), var) for g, var in grad]
    train_step = optimizer.apply_gradients(grad)
    return train_step
    
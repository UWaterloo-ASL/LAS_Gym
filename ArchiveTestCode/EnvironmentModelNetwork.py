#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 24 18:04:20 2018

@author: jack.lingheng.meng
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow import keras
import numpy as np
import tflearn
import os
import logging
# ===========================
#   Environment Model DNNs
# ===========================
class EnvironmentModelNetwork(object):
    """
    Environment Model is to learn the state trasition dynamics. Specifically,
    it learns the maping:
        from (observation_t, action_t) to (observation_t+1, reward_t)
    """
    def __init__(self, name, sess, observation_space, action_space,
                 learning_rate = 0.0001,
                 # Save Environment Model
                 env_model_save_path = '.',
                 # Restore Environment Model
                 env_restore_flag = False,
                 env_model_restore_path_and_name = 'results/models/env_model.ckpt'):
        """
        Initialize environment model.
        
        Parameters
        ----------
        name: str
        
        sess: tf.Session
        
        observation_space: gym.spaces.Box
            
        action_space: gym.spaces.Box
            
        learning_rate: float default = 0.0001
            learning rate
        env_model_save_path: str default = '.'
            the path to save environment model
                 # Restore Environment Model
        env_restore_flag: bool default = False
            idicate whether restore a previously trained environment model
        env_model_restore_path_and_name: str default = 'results/models/env_model.ckpt'
            the path and name of previously trained environment model that is 
            going to restore
        """
        self.name = name
        self.sess = sess
        self.observation_dim = observation_space.shape[0]
        self.action_dim = action_space.shape[0]
        self.learning_rate = learning_rate
        # Parameters for save environment model
        self.env_model_save_path = env_model_save_path
        # Parameters for restoring environment model
        self.env_restore_flag = env_restore_flag
        self.env_model_restore_path_and_name = env_model_restore_path_and_name
        
        # Create Environment Model
        with tf.variable_scope(self.name) as self.scope:
            self.obs_input, self.act_input,\
            self.obs_output, self.reward_output,\
            self.state_transition_model, self.reward_model = self.create_environment_model_network()
            # TODO:Load pre-trained model
            if self.env_restore_flag == True:
                self.state_transition_model.load()
                print('Restored actor model: {}'.format(temp_name))
        
        # Define loss and optimizer
        self.target_obs_output = tf.placeholder(tf.float32, [None, self.observation_dim])
        self.loss_state_transition_model = tflearn.objectives.mean_square(self.obs_output, self.target_obs_output)
        self.optimize_state_transition_model = tf.train.AdamOptimizer(self.learning_rate)(self.loss_state_transition_model)
        
        self.target_reward = tf.placeholder(tf.float32, [None, 1])
        self.loss_reward_model = tflearn.objectives.mean_square(self.reward_output, self.target_reward)
        self.optimize_reward_model = tf.train.AdamOptimizer(self.learning_rate)(self.loss_reward_model)
        
    def create_environment_model_network(self):
        """
        Create environment model network which includes:
            1. state_transition_model:
                (observation, action) -> next_observation
            2. reward_model:
                (observation, action) -> reward
        
        Returns
        -------
        observation_inputs: tf.Tensor
            tensor of observation input
        action_inputs: tf.Tensor
            tensor of action input
        observation_output: tf.Tensor
            tensor of observation output
        reward_output: tf.Tensor
            tensor of reward output
        state_transition_model: 
            
        reward_model:
        """
        observation_inputs = tflearn.input_data(shape=[None, self.observation_dim], name='EnvModel_observation_input')
        obs_h1 = tflearn.fully_connected(observation_inputs, 400, activation='relu')
        obs_h1 = tflearn.layers.normalization.batch_normalization(obs_h1)
        
        action_inputs = tflearn.input_data(shape = [None, self.action_dim], name='EnvModel_action_input')
        act_h1 = tflearn.fully_connected(action_inputs, 400, activation='relu')
        act_h1 = tflearn.layers.normalization.batch_normalization(act_h1)
        
        merged_inputs = tflearn.layers.merge_ops.merge([obs_h1, act_h1], mode='concat', name = 'EnvModel_merged_input')
        h2 = tflearn.fully_connected(merged_inputs, 300, activation='relu')
        h2 = tflearn.layers.normalization.batch_normalization(h2)
        
        observation_output = tflearn.fully_connected(h2, self.observation_dim, activation='sigmoid')
        reward_output = tflearn.fully_connected(h2, 1, activation='relu')
        
        state_transition_model = tflearn.models.dnn.DNN(observation_output, tensorboard_verbose = 3)
        reward_model = tflearn.models.dnn.DNN(reward_output, tensorboard_verbose = 3)
        
        return observation_inputs, action_inputs,\
               observation_output, reward_output,\
               state_transition_model, reward_model 
    
    def train(self, observation_inputs, action_inputs, observation_output, reward_output):
        """
        Train environment model which includes:
            1. state_transition_model
            2. reward_model
        
        Parameters
        ----------
        observation_inputs: ndarray
            observation at time step t
        action_inputs: ndarray
            action at time step t
        observation_output: ndarray
            observation at time step t+1
        reward_output: float
            reward at time step t
            
        """
        return self.sess.run([self.loss_state_transition_model, 
                              self.obs_output, 
                              self.optimize_state_transition_model,
                              self.loss_reward_model,
                              self.reward_output,
                              self.optimize_reward_model],
                             feed_dict={self.obs_input: observation_inputs,
                                        self.act_input: action_inputs,
                                        self.target_obs_output: observation_output,
                                        self.target_reward:reward_output})
    
    def evaluate(self, observation_inputs, action_inputs, observation_output, reward_output):
        """
        Evaluate environment model. The testing samples should be different from
        training samples.
        
        Parameters
        ----------
        observation_inputs: ndarray
            observation at time step t
        action_inputs: ndarray
            action at time step t
        observation_output: ndarray
            array combined of (observation,reward): 
                observation at time step t+1
                reward at time step t
        
        Returns
        -------
        loss_state_transition_model: float
            mean_square loss of state_transition_model
            
        loss_reward_model: float
            mean_square loss of reward_model
        """
        return self.sess.run(self.loss_state_transition_model,
                             self.loss_reward_model,
                             feed_dict = {self.obs_input: observation_inputs,
                                          self.act_input: action_inputs,
                                          self.target_obs_output: observation_output,
                                          self.target_reward:reward_output})
        
    def save_environment_model(self, version_num):
        """
        save environment model i.e. state_transition_model and reward_model
        Parameters
        ----------
        version_num: int or string
            the version number of the saved environment model
        """
        self.state_transition_model.save(os.path.join(self.env_model_save_path,
                                                      self.name + '_state_transition_model_' + str(version_num)+'.ckpt'))
        self.reward_model.save(os.path.join(self.env_model_save_path,
                                            self.name + '_reward_model_' + str(version_num)+'.ckpt'))
        logging.debug('Save environment model: {} done.'.format(self.name))
        
    def predict_target(self, observation, action):
        """
        Predict next step observation and reward based on current observation
        and action.
        
        Parameters
        ----------
        observation: ndarray
            observation at time step t
        action: ndarray
            action taken at time step t
        Returns
        -------
        obs_output: array
            prediction of next observation
        reward_output: float
            prediction of reward resulted from taking action in observation.
        """
        return self.sess.run(self.obs_output,
                             self.reward_output,
                             feed_dict={self.obs_input: observation,
                                        self.act_input: action})
#    def get_environment_model_weights(self):
#        """
#        
#        """
#        return self.state_transition_model.get_weights(), self.reward_model.get_weights()
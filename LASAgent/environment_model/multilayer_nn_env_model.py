#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 25 20:43:21 2018

@author: jack.lingheng.meng
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import tensorflow as tf
from tensorflow import layers
import numpy as np
import os
from datetime import datetime
# ===========================
#   Environment Model DNNs
# ===========================
class MultilayerNNEnvModel(object):
    """
    Environment Model is to learn the observation trasition model and reward
    model. Specifically:
        1. obs_transition_model: (observation, action) -> (next observation)
        2.         reward_model: (observation, action) -> (reward)
    """
    def __init__(self, name, sess, observation_space, action_space,
                 learning_rate = 0.0001,
                 # Save Environment Model
                 env_model_save_path = 'results/models/',
                 # Restore Environment Model
                 restore_model_flag = False):
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
        
        env_model_save_path: str default = 'results/models/'
            the path to save environment model
            
        env_load_flag: bool default = False
            idicate whether restore a previously trained environment model
        """
        self.name = name
        self.sess = sess
        self.observation_space = observation_space
        self.action_space = action_space
        self.observation_dim = observation_space.shape[0]
        self.action_dim = action_space.shape[0]
        self.learning_rate = learning_rate
        # Parameters for save environment model
        self.env_model_save_path = env_model_save_path
        # Parameters for restoring environment model
        self.restore_model_flag = restore_model_flag
        self.restore_model_version = self._find_the_most_recent_model_version()
        if self.restore_model_flag and self.restore_model_version == -1:
            logging.error('You do not have pretrained models.\nPlease set "load_pretrained_agent_flag = False".')
        
        with tf.name_scope(self.name): 
            # Create Environment Model
            with tf.variable_scope(self.name) as self.scope:
                self.obs_input, self.act_input,\
                self.obs_output, self.reward_output = self.create_env_model()
            self.env_model_params = tf.trainable_variables(self.name)
            self.env_model_saver = tf.train.Saver(self.env_model_params) # Saver to save and restore model variables
            
            # Optimize obs_transition_model
            self.next_obs_label = tf.placeholder(tf.float32, [None, self.observation_dim])
            self.obs_transition_model_loss = tf.losses.mean_squared_error(labels = self.next_obs_label, 
                                                                          predictions = self.obs_output)
            self.obs_transition_model_optimize = tf.train.AdamOptimizer(self.learning_rate).minimize(self.obs_transition_model_loss)
            
            # Optimize reward_model
            self.reward_label = tf.placeholder(tf.float32, [None, 1])
            self.reward_model_loss = tf.losses.mean_squared_error(labels = self.reward_label,
                                                                  predictions = self.reward_output)
            self.reward_model_optimize = tf.train.AdamOptimizer(self.learning_rate).minimize(self.reward_model_loss)
            
            # Initialize variables in variable_scope: self.name
            # Note: make sure initialize variables **after** defining all variable
            self.sess.run(tf.variables_initializer(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = self.name)))
            # Restore Environment Model
            if self.restore_model_flag == True:
                env_filepath = os.path.join(self.env_model_save_path,
                                            self.name + '_' + str(self.restore_model_version)+'.ckpt')
                self.restore_env_model(env_filepath)
    
    def create_env_model(self):
        """
        Create environment model network.
        
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
        observation_transition_model: 
            Keras model of observation transition model
        reward_model:
            Keras model of reward model
        """
        
        observation_inputs = tf.placeholder(tf.float32, shape=(None, self.observation_dim), name = 'EnvModel_observation_input')
        obs_h1 = layers.Dense(units = 100, activation = tf.nn.relu, 
                              kernel_initializer = tf.initializers.truncated_normal)(observation_inputs)
        obs_h1 = layers.BatchNormalization()(obs_h1)
        obs_h1 = layers.Dropout(0.5)(obs_h1)
        
        action_inputs = tf.placeholder(tf.float32, shape=(None, self.action_dim), name = 'EnvModel_action_input')
        act_h1 = layers.Dense(units = 100, activation = tf.nn.relu, 
                              kernel_initializer = tf.initializers.truncated_normal)(action_inputs)
        act_h1 = layers.BatchNormalization()(act_h1)
        act_h1 = layers.Dropout(0.5)(act_h1)
        
        merged = tf.concat([obs_h1, act_h1], axis=1, name = 'EnvModel_merged_input')
        
        merged_h1 = layers.Dense(units = 100, activation = tf.nn.relu,
                                 kernel_initializer = tf.initializers.truncated_normal)(merged)
        merged_h1 = layers.BatchNormalization()(merged_h1)
        merged_h1 = layers.Dropout(0.5)(merged_h1)
        
        observation_output = layers.Dense(units = self.observation_dim, 
                                          activation = tf.nn.relu,
                                          kernel_initializer = tf.initializers.truncated_normal)(merged_h1)
        
        reward_output = layers.Dense(units = 1,
                                     kernel_initializer = tf.initializers.truncated_normal)(merged_h1)
        
        return observation_inputs, action_inputs,\
               observation_output, reward_output      
    
    def train_env_model(self, observation_inputs, action_inputs, 
                                   observation_output, reward_output):
        """
        Train environemnt model which includes:
            1. observation transition model.
            2. reward model
        (These two share all weights except output layer.)
        
        Parameters
        ----------
        observation_inputs: ndarray
            observation at time step t
            
        action_inputs: ndarray
            action at time step t
            
        observation_output: ndarray
            observation at time step t+1
            
        reward_output: ndarray
                reward at time step t
        """
        return self.sess.run([self.obs_transition_model_optimize, self.reward_model_optimize],
                             feed_dict = {self.obs_input: observation_inputs, 
                                          self.act_input: action_inputs,
                                          self.next_obs_label: observation_output,
                                          self.reward_label: reward_output})
    
    def predict(self, observation, action):
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
        obs_output: 
            Numpy array of prediction of next observation.
        
        reward_output:
            Numpy array of prediction of reward.
        """
        return self.sess.run([self.obs_output, self.reward_output], 
                             feed_dict = {self.obs_input: observation, 
                                          self.act_input: action})
    
    def evaluate_env_model(self, observation_inputs, action_inputs, 
                                      observation_output, reward_output):
        """
        Evaluate observation transition model. The testing samples should be 
        different from training samples.
        
        Parameters
        ----------
        observation_inputs: ndarray
            observation at time step t
            
        action_inputs: ndarray
            action at time step t
            
        observation_output: ndarray
                observation at time step t+1
                
        reward_output: ndarray
            reward at time step t
        
        Returns
        -------
        obs_loss:
            square root of mean squared error of prediction of next observation 
            on the test set.
            
        reward_loss:
            square root of mean squared error of prediction of reward 
            on the test set.
        """
        obs_loss, reward_loss = self.sess.run([self.obs_transition_model_loss, self.reward_model_loss],
                                              feed_dict = {self.obs_input: observation_inputs, 
                                                           self.act_input: action_inputs,
                                                           self.next_obs_label: observation_output,
                                                           self.reward_label: reward_output})
        return np.sqrt(obs_loss), np.sqrt(reward_loss)
    
    def save_env_model(self, version_number):
        """
        Save observation transition model
        
        Parameters
        ----------
        version_number: int or string
            the version number of the saved observation transition model
        """
        env_filepath = os.path.join(self.env_model_save_path,
                                    self.name + '_' + str(version_number)+'.ckpt')
        self.env_model_saver.save(self.sess, env_filepath)
        logging.info('Environment model saved in path: {}.'.format(env_filepath))
        
    def restore_env_model(self, env_filepath):
        """ 
        The following code is to inspect variables in a checkpoint:
            from tensorflow.python.tools import inspect_checkpoint as chkp
            chkp.print_tensors_in_checkpoint_file(file_path, tensor_name='', all_tensors=True, all_tensor_names=True)
        """
        self.env_model_saver.restore(self.sess, env_filepath)
        logging.info('Restored environment model: {}'.format(env_filepath))
    
    def _find_the_most_recent_model_version(self):
        """
        Returns
        -------
        the_most_recent_model_version: int
            the most recent model version
        """
        # Find the most recent version
        model_version = []
        for file_name_temp in os.listdir(self.env_model_save_path):
            if self.name in file_name_temp:
                _, version_temp = file_name_temp.split('.')[0].split(self.name+'_')
                model_version.append(version_temp)
        if len(model_version) != 0:
            the_most_recent_model_version = max([int(i) for i in model_version])
        else:
            the_most_recent_model_version = -1
        return the_most_recent_model_version
    
    def get_env_model_weights(self):
        """
        Retrives the weights of obs_transition_model.
         
        Returns
        -------
        env_model_weights: A flat list of Numpy arrays
        """
        env_model_weights = []
        for param in self.env_model_params:
            env_model_weights.append(param.eval(session = self.sess))
        return env_model_weights
    
    def set_env_model_weights(self, env_model_weights):
        """
        Parameters:
        env_model_weights: A flat list of Numpy arrays
        """
        if len(env_model_weights) != len(self.env_model_params):
            logging.error('The environment model weights do not match.')
        
        for i, param in enumerate(self.env_model_params):
            param.assign(env_model_weights[i])
        
    def __deepcopy__(self, memo):
        """
        Deep Copy an environment model.
        """
        unique_name = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        environment_model_copy = MultilayerNNEnvModel('env_copy_'+unique_name,
                                                 self.sess,
                                                 self.observation_space,
                                                 self.action_space,
                                                 self.learning_rate,
                                                 self.env_model_save_path,
                                                 False)
        env_model_weights = []
        for param in self.env_model_params:
            env_model_weights.append(param.eval(session = self.sess))
        environment_model_copy.set_env_model_weights(env_model_weights)
        return environment_model_copy
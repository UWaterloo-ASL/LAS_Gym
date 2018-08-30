#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 26 08:57:33 2018

@author: jack.lingheng.meng
"""
import logging
from collections import deque
import numpy as np

import copy
import pickle

# ===========================
#  Intrinsic Motivation Components
# ===========================
class KnowledgeBasedIntrinsicMotivationComponent():
    def __init__(self, env_model, 
                 sliding_window_size):
        """
        Parameters
        ----------
        env_model: an object of env_model
            
        sliding_window_size: int
            the size of sliding window
        """
        if int(sliding_window_size) < 2:
            logging.error("sliding_window_size should be an int >= 2")
        self.sliding_window_size = sliding_window_size
       
        # Note: use copy.deepcopy()
        self.env_model_window = deque(maxlen = self.sliding_window_size)
        for w_i in range(self.sliding_window_size):
            self.env_model_window.append(copy.deepcopy(env_model))
        
    def update_env_model_window(self, env_model_weights):
        """
        Update the environment model window by:
            1. pop the odlest env model
            2. update the weights of oldeest env model to newest env model
            3. append the newest env model
        
        Parameters
        ----------
        env_model: an object of env_model
        """
#        # Note:
#        #   1. Since the window is full, newly appended environment model will cause
#        #      discarding oldest environment model.
#        #   2. Use copy.deepcopy() to append a deep-copy of environement model
#        self.env_model_window.append(copy.deepcopy(env_model))
        oldest_env_model = self.env_model_window.popleft()
        oldest_env_model.set_env_model_weights(env_model_weights)
        self.env_model_window.append(oldest_env_model)
        logging.info('Update Environment Model Window: done.')
        
    def knowledge_based_intrinsic_reward(self, obs_old, act_old,
                                         r_new, obs_new):
        """
        Calculate knowledge-based intrinsic motivation i.e. learning progress:
            learning_progress = |first_half_window_mean_error|-|second_half_window_mean_error|.
        
        Called every step.
        
        Parameters
        ----------
        obs_old: np.shape(observation) = (obs_dim,)
            observation at time step t
            
        act_old: np.shape(action) = (act_dim, )
            action at time step t
            
        r_new: float
            reward at time stem t
            
        obs_new: np.shape(observation) = (obs_dim,)
            observation at time step t+1
            
        Returns
        -------
        obs_learning_progress: float
            learning progress based on observation prediction error
            
        reward_learning_progress: float
            learning progress based on reward prediction error
        """
        obs_old = np.reshape(obs_old, [1, np.shape(obs_old)[0]])
        act_old = np.reshape(act_old, [1, np.shape(act_old)[0]])
        # Calculate Learning Progress
        window_split_index = int(self.sliding_window_size/2)
        first_half_obs_error_norm = 0
        first_half_reward_error_norm = 0
        second_half_obs_error_norm = 0
        second_half_reward_error_norm = 0
        for i in range(self.sliding_window_size):
            obs_prediction, reward_prediction = self.env_model_window[i].predict(obs_old, act_old)
            obs_error = obs_new - obs_prediction[0]
            reward_error = r_new - reward_prediction[0]
            
            obs_error_norm = np.linalg.norm(obs_error, ord = 2)
            reward_error_norm = np.linalg.norm(reward_error, ord = 2)
            
            if i <= window_split_index:
                first_half_obs_error_norm += obs_error_norm
                first_half_reward_error_norm += reward_error_norm
            else:
                second_half_obs_error_norm += obs_error_norm
                second_half_reward_error_norm += reward_error_norm
        average_first_half_obs_error_norm = first_half_obs_error_norm / window_split_index
        average_first_half_reward_error_norm = first_half_reward_error_norm / window_split_index
        
        average_second_half_obs_error_norm = second_half_obs_error_norm / (self.sliding_window_size-second_half_reward_error_norm)
        average_second_half_reward_error_norm = second_half_reward_error_norm / (self.sliding_window_size-second_half_reward_error_norm)
        
        obs_learning_progress = abs(average_first_half_obs_error_norm - average_second_half_obs_error_norm)
        reward_learning_progress = abs(average_first_half_reward_error_norm - average_second_half_reward_error_norm)
        return obs_learning_progress, reward_learning_progress
       
    def save_knowledge_based_intrinsic_motivation_component(self):
        """
        TODO: Save knowledge based intrinsic motivation component
              (Use pickle module)
        """
        
    
    def load_knowledge_based_intrinsic_motivation_component(self):
        """
        TODO: Load knowledge based intrinsic motivation component 
        """
    
    def visualize_knowledge_based_intrinsic_motivation(self, intrinsically_motivated_actor,
                                                       intrinsically_motivated_critic):
        """
        TODO: visualize intrinsic motivation using T-SNE.
        Our data has the form:
            action = Actor(observation)
            interest = Critic(observation, action)
        Parameters
        ----------
        intrinsically_motivated_actor:
            
        intrinsically_motivated_critic:
            
        """

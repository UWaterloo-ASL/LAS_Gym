#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  8 17:12:06 2018

@author: jack.lingheng.meng
"""
from gym import spaces
from datetime import datetime
import os

from LASAgent.LASAgent_Actor_Critic import LASAgent_Actor_Critic


class InternalEnvOfAgent(object):
    """
    This class provides Internal Environment for an agent.
    """
    def __init__(self, sess, agent_name, observation_space, action_space,
                 interaction_mode = 'real_interaction'):
        """
        Initialize internal environment for an agent
        Parameters
        ----------
        agent_name: string
            the name of the agent this internal environment serves for
        observation_space: gym.spaces.Box datatype
            observation space of "agent_name"
        action_space: gym.spaces.Box datatype
            action space of "agent_name"
        interaction_mode: string default = 'real_interaction'
            indicate interaction mode: 
                1) 'real_interaction': interact with real robot
                2) 'virtual_interaction': interact with virtual environment
        """
        self.tf_session = sess
        # Initialize agent
        self.agent_name = agent_name
        self.observation_space = observation_space
        self.action_space = action_space
        self.agent = LASAgent_Actor_Critic(self.tf_session,
                                           self.agent_name,
                                           self.observation_space,
                                           self.action_space,
                                           actor_lr = 0.0001, actor_tau = 0.001,
                                           critic_lr = 0.0001, critic_tau = 0.001, gamma = 0.99,
                                           minibatch_size = 64,
                                           max_episodes = 50000, max_episode_len = 1000,
                                           # Exploration Strategies
                                           exploration_action_noise_type = 'ou_0.2',
                                           exploration_epsilon_greedy_type = 'none',
                                           # Save Summaries
                                           save_dir = os.path.join('../ROM_Experiment_results',self.agent_name),
                                           experiment_runs = datetime.now().strftime("%Y%m%d-%H%M%S"),
                                           # Save and Restore Actor-Critic Model
                                           restore_actor_model_flag = False,
                                           restore_critic_model_flag = False)
        self.interaction_mode = interaction_mode
        
    def interact(self, observation, external_reward = 0, done = False):
        """
        The interface function interacts with external environment.
        
        Parameters
        ----------
        observation: ndarray
            the observation received from external environment
        external_reward: float (optional)
            this parameter is used only when external reward is provided by 
            simulating environment
            
        Returns
        -------
        action: ndarray
            the action chosen by intelligent agent
        """
        if self.interaction_mode == 'real_interaction':
            reward = self._extrinsic_reward_func(observation)
        elif self.interaction_mode == 'virtual_interaction':
            reward = external_reward
        else:
            raise Exception('Please choose right interaction mode!')
        
        done = False
        action = self.agent.perceive_and_act(observation, reward, done)
        return action
    
    def _extrinsic_reward_func(self, observation):
        """
        This function is used to provide extrinsic reward.
        """
        reward = 1
        return reward
    
    def start(self):
        """
        This interface function is to load pretrained models.
        """
        
        
    def stop(self):
        """
        This interface function is to save trained models.
        """
        
    def feed_observation(self, observation):
        """
        This interface function only receives observation from environment, but
        not return action.
        
        Parameters
        observation: ndarray
            the observation received from external environment
        """
        

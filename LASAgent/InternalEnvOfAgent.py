#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  8 17:12:06 2018

@author: jack.lingheng.meng
"""
import logging
logger = logging.getLogger('Learning.'+__name__)

from gym import spaces
from datetime import datetime
import os
import numpy as np
from collections import deque
import csv
import tensorflow as tf
import math

from LASAgent.LASAgent_Actor_Critic import LASAgent_Actor_Critic


class InternalEnvOfAgent(object):
    """
    This class provides Internal Environment for an agent.
    """
    def __init__(self, agent_name, 
                 observation_space, action_space,
                 observation_space_name, action_space_name,
                 x_order_sensor_reading = 20,
                 x_order_sensor_reading_sliding_window = 5,
                 x_order_sensor_reading_preprocess_type = 'concatenate_sensory_readings',
                 occupancy_reward_type = 'IR_distance',
                 interaction_mode = 'real_interaction',
                 load_pretrained_agent_flag = False,
                 baseline_agent_flag = False):
        """
        Initialize internal environment for an agent
        
        Args:
            agent_name (string): the name of the agent this internal environment serves for
            observation_space (gym.spaces.Box): observation space of "agent_name".
            action_space (gym.spaces.Box): action space of "agent_name"
            observation_space_name (string list): each entry corresponds to the 
                name of sensor in observation space
            action_space_name (string list): each entry corresponds to the name
                of actuator in action space 
        
        Kwargs:
            x_order_sensor_reading (int): the # of sensory readings after which
                an action will be produced
            x_order_sensor_reading_sliding_window (int): size of sliding window
                for preprocessing sensory readings
            x_order_sensor_reading_preprocess_type (string): the way to combine
                sensory readings to form an observation:
                    1. concatenate_sensory_readings
                    2. average_pool_sensory_readings
                    3. max_pool_sensory_readings
            occupancy_reward_type (string): the way to calculate reward:
                    1. 'IR_distance': based on IR distance from detected object to IR
                    2. 'IR_state_ratio': the ratio of # of detected objects and all # 
                            of IR sensors 
                    3. 'IR_state_number': the number of detected objects
            interaction_mode (string): indicates interaction mode: 
                    1. 'real_interaction': interact with real robot
                    2. 'virtual_interaction': interact with virtual environment
            load_pretrained_agent_flag (bool): whether load pretrained agent
                    if True: load pretrained agent. Otherwise randomly initialize.
            baseline_agent_flag (bool): whether choose baseline_agent
                    if True, only collect data and not produce action.
        """
        # self.tf_session is released in self.stop()
        self.tf_session = tf.Session()
        
        self.x_order_sensor_reading = x_order_sensor_reading
        self.x_order_sensor_reading_sliding_window = 5
        self.x_order_sensor_reading_sequence = deque(maxlen = self.x_order_sensor_reading)
        self.x_order_sensor_reading_preprocess_type = x_order_sensor_reading_preprocess_type
        
        self.occupancy_reward_type = occupancy_reward_type
        self.interaction_mode = interaction_mode
        #####################################################################
        #                       Initialize agent                            #
        #####################################################################
        self.name = agent_name
        self.agent_name = agent_name
        
        self.original_observation_space = observation_space
        self.original_observation_space_name = observation_space_name
        self.original_action_space = action_space
        self.original_action_space_name = action_space_name
        
        self.actual_observation_space,\
        self.actual_observation_space_name = self._initialize_actual_observation_space_and_name(self.x_order_sensor_reading,
                                                                                                self.x_order_sensor_reading_sliding_window,
                                                                                                self.x_order_sensor_reading_preprocess_type)
        self.actual_action_space = action_space
        self.actual_action_space_name = action_space_name
        
        self.baseline_agent_flag = baseline_agent_flag
        
        # Model saving directory
        self.model_version_number = 0
        self.agent_model_save_dir = os.path.join(os.path.abspath('../..'),'ROM_Experiment_results',self.agent_name)
        
        if load_pretrained_agent_flag == False:
            self.agent = LASAgent_Actor_Critic(self.tf_session,
                                               self.agent_name,
                                               self.actual_observation_space,
                                               self.actual_action_space,
                                               actor_lr = 0.0001, actor_tau = 0.001,
                                               critic_lr = 0.0001, critic_tau = 0.001, gamma = 0.99,
                                               minibatch_size = 64,
                                               max_episodes = 50000, max_episode_len = 500,
                                               # Exploration Strategies
                                               exploration_action_noise_type = 'none', #'ou_0.2',
                                               exploration_epsilon_greedy_type = 'epsilon-greedy-max_1_min_0.05_decay_0.9',#'none',#
                                               # Save Summaries
                                               save_dir = self.agent_model_save_dir,
                                               experiment_runs = datetime.now().strftime("%Y%m%d-%H%M%S"),
                                               # Save and Restore Actor-Critic Model
                                               restore_actor_model_flag = False,
                                               restore_critic_model_flag = False)
        elif load_pretrained_agent_flag == True:
            self._initialize_pretrained_agent()
        else:
            raise Exception('Please set load_pretrained_agent parameter!')
        #####################################################################
        #                 Interaction data saving directory                 #
        #####################################################################
        self.interaction_data_dir = os.path.join(os.path.abspath('../..'),
                                                 'ROM_Experiment_results',
                                                 self.agent_name,
                                                 'interaction_data')
        if not os.path.exists(self.interaction_data_dir):
            os.makedirs(self.interaction_data_dir)
        self.interaction_data_file = os.path.join(self.interaction_data_dir,
                                                  datetime.now().strftime("%Y%m%d-%H%M%S")+'.csv')
        with open(self.interaction_data_file, 'a') as csv_datafile:
            fieldnames = ['Time', 'Observation', 'Reward', 'Action']
            writer = csv.DictWriter(csv_datafile, fieldnames = fieldnames)
            writer.writeheader()
        #####################################################################
        #                   Logging Experiment Setting                      #
        #####################################################################
        # TODO: add function to log experiment setting to ensure not messing
        #   up experiment results.
        #self._logging_experiment_setting
        
    def interact(self, actual_observation, external_reward = 0, done = False):
        """
        The interface function interacts with external environment.
        
        Args:
            actual_observation (list): the actual observation after preprocessing
            external_reward (float): this parameter is used only when external 
                reward is provided by simulating environment
            
        Returns:
            action (list): the action chosen by intelligent agent
        """
        if self.interaction_mode == 'real_interaction':
            reward = self._reward_occupancy(actual_observation,
                                            self.x_order_sensor_reading,
                                            self.x_order_sensor_reading_sliding_window,
                                            self.occupancy_reward_type)
        elif self.interaction_mode == 'virtual_interaction':
            reward = external_reward
        else:
            raise Exception('Please choose right interaction mode!')
        logger.info('Reward of {} is: {}'.format(self.agent_name, reward))
        # If running baseline only collect data, no need to calculate action
        if self.baseline_agent_flag == False:
            #logger.info('Not use baseline_agent')
            action = self.agent.perceive_and_act(actual_observation, reward, done)
        else:
            # TODO: The reward of baseline is not summarized in tensorflow summary.
            #logger.info('Use baseline_agent!')
            action = self.agent.perceive_and_act(actual_observation, reward, done)
            action = []
        # Logging interaction data
        self._logging_interaction_data(actual_observation,
                                       reward,
                                       action)
        return action
    
    def _initialize_actual_observation_space_and_name(self, x_order_sensor_reading,
                                                      x_order_sensor_reading_sliding_window,
                                                      x_order_sensor_reading_preprocess_type):
        """
        Initialize actual observation space and name according to arguments.
        
        Args:
            x_order_sensor_reading (int): the # of sensory readings after which
                an action will be produced
            x_order_sensor_reading_sliding_window (int): size of sliding window
                for preprocessing sensory readings
            x_order_sensor_reading_preprocess_type (string): the way to combine
                sensory readings to form an observation:
                    1. concatenate_sensory_readings
                    2. average_pool_sensory_readings
                    3. max_pool_sensory_readings
            
        Returns:
            actual_observation_space (gym.spaces.Box):
            actual_observation_space_name (string list):
        """
        if x_order_sensor_reading_preprocess_type == 'concatenate_sensory_readings':
            tile_number = x_order_sensor_reading
        else:
            tile_number = math.ceil(x_order_sensor_reading / x_order_sensor_reading_sliding_window)
        actual_observation_space = spaces.Box(low = np.tile(self.original_observation_space.low,tile_number),
                                              high = np.tile(self.original_observation_space.high,tile_number), 
                                              dtype = np.float32)
        actual_observation_space_name = np.tile(self.original_observation_space_name, tile_number)
        return actual_observation_space, actual_observation_space_name
    
    def _generate_actual_observation(self, x_order_sensor_reading,
                                     x_order_sensor_reading_sequence,
                                     x_order_sensor_reading_sliding_window,
                                     x_order_sensor_reading_preprocess_type):
        """
        Generate actual observation accroding to arguments and a queue of sensory
        readings.
        Args:
            x_order_sensor_reading (int): the # of sensory readings after which
                an action will be produced
            x_order_sensor_reading_sequence (deque object): a queue contains
                #x_order_sensor_reading sensory readings
            
        x_order_sensor_reading_preprocess_type (string): ways to generate actual 
            observation:
                1. concatenate_sensory_readings: concatenate all sensor readings
                      in x_order_sensor_reading_sequence
                2. average_pool_sensory_readings: average sensor readings within
                      x_order_sensor_reading_sliding_window, then concatenate
                3. max_pool_sensory_readings: take maximun sensor reading whithin
                      x_order_sensor_reading_sliding_window, then concatenate
        
        Returns:
            actual_observation (numpy array):
        """
        # TODO: prepare at least three ways
        #   1. concatenate_sensory_readings
        #   2. average_pool_sensory_readings
        #   3. max_pool_sensory_readings
        actual_observation = []
        if x_order_sensor_reading_preprocess_type == 'concatenate_sensory_readings':
            while x_order_sensor_reading_sequence:
                obs_temp = x_order_sensor_reading_sequence.popleft()
                actual_observation = np.append(actual_observation, obs_temp)
        elif x_order_sensor_reading_preprocess_type == 'average_pool_sensory_readings':
            while x_order_sensor_reading_sequence:
                temp_count = 0
                temp_data_matrix = []
                while temp_count <= x_order_sensor_reading_sliding_window and x_order_sensor_reading_sequence:
                    data = x_order_sensor_reading_sequence.popleft()
                    temp_data_matrix = np.append(temp_data_matrix, data)
                    temp_count += 1
                temp_data_matrix = temp_data_matrix.reshape(temp_count,self.original_observation_space.shape[0])
                actual_observation = np.append(actual_observation, np.mean(temp_data_matrix, axis = 0))
        elif x_order_sensor_reading_preprocess_type == 'max_pool_sensory_readings':
            while x_order_sensor_reading_sequence:
                temp_count = 0
                temp_data_matrix = []
                while temp_count <= x_order_sensor_reading_sliding_window and x_order_sensor_reading_sequence:
                    data = x_order_sensor_reading_sequence.popleft()
                    temp_data_matrix = np.append(temp_data_matrix, data)
                    temp_count += 1
                temp_data_matrix = temp_data_matrix.reshape(temp_count,self.original_observation_space.shape[0])
                actual_observation = np.append(actual_observation, np.max(temp_data_matrix, axis = 0))
        else:
            raise Exception('Please choose a proper x_order_sensor_reading_preprocess_type!')
        
        return actual_observation
    
    def _reward_occupancy(self, observation,
                          x_order_sensor_reading,
                          x_order_sensor_reading_sliding_window,
                          reward_type = 'IR_distance'):
        """
        Calculate reward based on occupancy i.e. the IRs data
        
        Args:
            observation (numpy array): observation array
            x_order_sensor_reading (int): the # of sensory readings after which
                an action will be produced
            x_order_sensor_reading_sliding_window (int): size of sliding window
                for preprocessing sensory readings
        Kwargs:
            reward_type (string): reward type
                1. 'IR_distance': based on IR distance from detected object to IR
                2. 'IR_state_ratio': the ratio of # of detected objects and all # 
                                     of IR sensors 
                3. 'IR_state_number': the number of detected objects
        
        Returns:
            reward (float): the value of reward
        """
        prox_distances = observation
        # Make here insistent with IR data
        #   1. 'IR_distance': based on IR distance from detected object to IR
        #   2. 'IR_state_ratio': the ratio of # of detected objects and all # 
        #                        of IR sensors 
        #   3. 'IR_state_number': the number of detected objects
        reward_temp = 0.0
        if reward_type == 'IR_distance':
            for distance in prox_distances:
                if distance != 0:
                    reward_temp += distance
        elif reward_type == 'IR_state_ratio':
            for distance in prox_distances:
                if distance != 0:
                    reward_temp += 1
            reward_temp = reward_temp / len(prox_distances)
        elif reward_type == 'IR_state_number':
            for distance in prox_distances:
                if distance != 0:
                    reward_temp += 1
        else:
            raise Exception('Please choose a proper reward type!')
        # Average ovser windows 
        tile_number = math.ceil(x_order_sensor_reading / x_order_sensor_reading_sliding_window)
        self.reward = reward_temp / tile_number
        return self.reward
    
    def _initialize_pretrained_agent(self):
        """
        This function is to load pretrained models. This function 
        is actually to reinitialize an agent with pretrained models.
        
        (Call this function only if you want to start learning with most recently 
        trained agent.)
        """
        # Search for most recent model in model directory
        model_directory = os.path.join(self.agent_model_save_dir,'models')
        model_created_date = []
        for directory_temp in os.listdir(model_directory):
            # Only compare date
            if '2018' in directory_temp:
                model_created_date.append(directory_temp)
        directory_of_most_recent_models = max(model_created_date)
        # Instantiate Agent With the Pretrained Model
        self.agent = LASAgent_Actor_Critic(self.tf_session,
                                           self.agent_name,
                                           self.actual_observation_space,
                                           self.actual_action_space,
                                           actor_lr = 0.0001, actor_tau = 0.001,
                                           critic_lr = 0.0001, critic_tau = 0.001, gamma = 0.99,
                                           minibatch_size = 64,
                                           max_episodes = 50000, max_episode_len = 500,
                                           # Exploration Strategies
                                           exploration_action_noise_type = 'none',
                                           exploration_epsilon_greedy_type = 'epsilon-greedy-max_0.5_min_0.05_decay_0.9',
                                           # Save Summaries
                                           save_dir = self.agent_model_save_dir,
                                           experiment_runs = directory_of_most_recent_models,
                                           # Save and Restore Actor-Critic Model
                                           restore_actor_model_flag = True,
                                           restore_critic_model_flag = True,
                                           restore_env_model_flag = True)
        
        
    def stop(self):
        """
        This interface function is to:
            1. save trained models for the agent:
               * actor-critic model
               * environment model
            2. release tensorflow.Session resources
        
        (Try to call this function before shut down learning to maintain most
        recently trained agent, although there is a periodic saving which cannot
        ensure saving the most recent trained agent.)
        """
        print('{}: Stoping and Saving ...'.format(self.name))
        # TODO: the version number is better to be an unique time.
        self.model_version_number += 1
        # Save Actor-Critic model
        self.agent.extrinsic_actor_model.save_actor_network(self.model_version_number)
        self.agent.extrinsic_critic_model.save_critic_network(self.model_version_number)
        # Save Environment Model
        
        self.agent.environment_model.save_env_model(self.model_version_number)
        # Save Replay Buffer ?? (not really necessary)
        
        # Release TensorFlow.Session resources
        self.tf_session.close()
        print('{}: Saved model.'.format(self.name))
        
    def feed_observation(self, observation, external_reward = 0, done = False):
        """
        This interface function receives observation from environment, but
        produce an action with a different frequency.
        
        If take_action_flag == Ture, there is a valid action can be taken.
        
        (Training could also be done when feeding observation.)
        
        Args:
            observation (list): the sensory reading received from external environment
            external_reward (float): only provied when using virtual environment 
                (ignored when interact with real system)
            done (bool): only provied when using virtual environment
                (ignored when interact with real system)
        
        Returns:
            take_action_flag (bool): indicate whether to take an action
            action (numpy array): the action value
        """
        # If x_order_sensor_reading_sequence is not filled, keep filling.
        # After filled, take an action.
        if len(self.x_order_sensor_reading_sequence) != self.x_order_sensor_reading:
            self.x_order_sensor_reading_sequence.append(observation)
            action = []
            take_action_flag = False
            # Train all agent when feeding observation (Optional)
            self.agent._train()
        else:
            # Generate observation for x_order_MDP
            #   Note: 
            #       1. use shallow copy:
            #             self.x_order_sensor_reading_sequence.copy(),
            #         otherwise the self.x_order_sensor_reading_sequence is reset to empty,
            #         after call this function.
            #       2. clear the observation queue after taking an action
            actual_observation = self._generate_actual_observation(self.x_order_sensor_reading,
                                                                   self.x_order_sensor_reading_sequence.copy(),
                                                                   self.x_order_sensor_reading_sliding_window,
                                                                   self.x_order_sensor_reading_preprocess_type)
            
            action = self.interact(actual_observation, external_reward, done)
            take_action_flag = True
            # Clear the observation queue
            self.x_order_sensor_reading_sequence.clear()
        return take_action_flag, action

    def _logging_interaction_data(self, actual_observation,
                                  reward,
                                  action):
        """
        Saving interaction data
        
        Args:
            actual_observation (numpy array):
            reward (float):
            action (numpy array):
        """
        with open(self.interaction_data_file, 'a') as csv_datafile:
            fieldnames = ['Time', 'Actual_Observation', 'Reward', 'Action']
            writer = csv.DictWriter(csv_datafile, fieldnames = fieldnames)
            writer.writerow({'Time':datetime.now().strftime("%Y%m%d-%H%M%S"),
                             'Actual_Observation': actual_observation, 
                             'Reward': reward, 
                             'Action':action})
        
        
        
       

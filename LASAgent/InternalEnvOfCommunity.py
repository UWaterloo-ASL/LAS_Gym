#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  8 21:47:50 2018

@author: jack.lingheng.meng
"""
import logging
logger = logging.getLogger('Learning.'+__name__)

import os
from datetime import datetime
import numpy as np
import csv
from gym import spaces
from LASAgent.InternalEnvOfAgent import InternalEnvOfAgent
from LASAgent.RandomLASAgent import RandomLASAgent

class InternalEnvOfCommunity(object):
    """
    This class provides an internal environment for a community of agents to 
    interact with external environment.
    Note:
        InternalEnvOfCommunity is only used to partition observation and action
        space.
    """
    def __init__(self, community_name, community_size,
                 observation_space, action_space, 
                 observation_space_name, action_space_name,
                 x_order_sensor_reading = 20,
                 x_order_sensor_reading_sliding_window = 5,
                 x_order_sensor_reading_preprocess_type = 'concatenate_sensory_readings',
                 occupancy_reward_type = 'IR_distance',
                 interaction_mode = 'real_interaction',
                 load_pretrained_agent_flag = False):
        """
        Initialize internal environment for an agent community in where multiple
        agents live in.
        
        Args:
            community_name (string): the name of the community
            community_size (int): the # of agents living in the community 
            observation_space (gym.spaces.Box): the sensory space of the agent 
                community
            action_space (gym.spaces.Box): the actuator space of the whole agent
                community
            observation_space_name (list of string): each entry corresponds to
                the name of sensor in observation space
            action_space_name (list of string): each entry corresponds to the 
                name of actuator in action space
        
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
        """
        self.x_order_sensor_reading = x_order_sensor_reading
        self.x_order_sensor_reading_sliding_window = x_order_sensor_reading_sliding_window
        self.x_order_sensor_reading_preprocess_type = x_order_sensor_reading_preprocess_type
        
        self.occupancy_reward_type = occupancy_reward_type
        self.interaction_mode = interaction_mode
        
        self.load_pretrained_agent_flag = load_pretrained_agent_flag
        
        # Initialize community
        self.name = community_name
        self.community_name = community_name
        self.community_size = community_size
        ####################################################################
        #                          Configuration
        ####################################################################
        # Config 1: with shared sensor
        self.community_config_obs = {'agent_1':['node','node#22','node#21','node#20','node#19','node#18','node#17','node#16','node#15'],
                                     'agent_2':['node#16','node#15','node#14','node#13','node#12','node#11','node#10','node#9','node#8'],
                                     'agent_3':['node#9','node#8','node#7','node#6','node#5','node#4','node#3','node#2','node#1','node#0']
                                     }
        self.community_config_act = {'agent_1':['node','node#22','node#21','node#20','node#19','node#18','node#17'],
                                     'agent_2':['node#16','node#15','node#14','node#13','node#12','node#11','node#10','node#9','node#8'],
                                     'agent_3':['node#7','node#6','node#5','node#4','node#3','node#2','node#1','node#0']
                                     }
#        # Config 2: no shared sensor
#        self.community_config_obs = {'agent_1':['node','node#22','node#21','node#20','node#19','node#18','node#17'],
#                                     'agent_2':['node#16','node#15','node#14','node#13','node#12','node#11','node#10','node#9','node#8'],
#                                     'agent_3':['node#7','node#6','node#5','node#4','node#3','node#2','node#1','node#0']
#                                     }
#        self.community_config_act = {'agent_1':['node','node#22','node#21','node#20','node#19','node#18','node#17'],
#                                     'agent_2':['node#16','node#15','node#14','node#13','node#12','node#11','node#10','node#9','node#8'],
#                                     'agent_3':['node#7','node#6','node#5','node#4','node#3','node#2','node#1','node#0']
#                                     }
        # Config 3: share all sensor
#        self.community_config_obs = {'agent_1':['node','node#22','node#21','node#20','node#19','node#18','node#17','node#16','node#15',\
#                                                'node#14','node#13','node#12','node#11','node#10','node#9','node#8','node#7','node#6',\
#                                                'node#5','node#4','node#3','node#2','node#1','node#0'],
#                                     'agent_2':['node','node#22','node#21','node#20','node#19','node#18','node#17','node#16','node#15',\
#                                                'node#14','node#13','node#12','node#11','node#10','node#9','node#8','node#7','node#6',\
#                                                'node#5','node#4','node#3','node#2','node#1','node#0'],
#                                     'agent_3':['node','node#22','node#21','node#20','node#19','node#18','node#17','node#16','node#15',\
#                                                'node#14','node#13','node#12','node#11','node#10','node#9','node#8','node#7','node#6',\
#                                                'node#5','node#4','node#3','node#2','node#1','node#0']
#                                     }
#        self.community_config_act = {'agent_1':['node','node#22','node#21','node#20','node#19','node#18','node#17'],
#                                     'agent_2':['node#16','node#15','node#14','node#13','node#12','node#11','node#10','node#9','node#8'],
#                                     'agent_3':['node#7','node#6','node#5','node#4','node#3','node#2','node#1','node#0']
#                                     }
        ####################################################################
        
        self.observation_space = observation_space
        self.observation_space_name = observation_space_name
        self.action_space = action_space
        self.action_space_name = action_space_name
        # Information on Partition Config of observation and action space
        self.agent_community_partition_config = \
                self._create_community_partition_from_config(self.community_name,
                                                             self.community_size,
                                                             self.observation_space,
                                                             self.observation_space_name,
                                                             self.action_space,
                                                             self.action_space_name,
                                                             self.community_config_obs,
                                                             self.community_config_act)
        # Creat a community of agents
        #   1. 'random_agent'
        #   2. 'actor_critic_agent'
        self.agent_community = self._create_agent_community(self.agent_community_partition_config,
                                                            self.x_order_sensor_reading,
                                                            self.x_order_sensor_reading_sliding_window,
                                                            self.x_order_sensor_reading_preprocess_type,
                                                            self.occupancy_reward_type,
                                                            self.interaction_mode,
                                                            self.load_pretrained_agent_flag,
                                                            agent_config = 'actor_critic_agent')  
        ####################################################################
        #                          Initialize Summary
        ####################################################################
        self.total_step_counter = 0
        
        #####################################################################
        #                 Interaction data saving directory                 #
        #                                                                   #
        # Note: each agent living in this community has its own Interaction #
        #       data saving directory. Here is a saving of interaction from #
        #       the perspective of Agent-Community.
        #####################################################################
        self.interaction_data_dir = os.path.join(os.path.abspath('../..'), 'ROM_Experiment_results',
                                                 self.community_name, 'interaction_data')
        if not os.path.exists(self.interaction_data_dir):
            os.makedirs(self.interaction_data_dir)
        self.interaction_data_file = os.path.join(self.interaction_data_dir,
                                                  datetime.now().strftime("%Y%m%d-%H%M%S")+'.csv')
        with open(self.interaction_data_file, 'a') as csv_datafile:
            fieldnames = ['time', 'observation', 'observation_partition', 'reward_partition',
                          'action_partition', 'take_action_flag_partition',
                          'take_action_flag', 'action']
            writer = csv.DictWriter(csv_datafile, fieldnames = fieldnames)
            writer.writeheader()
    
    def _create_community_partition_from_config(self, community_name,
                                                community_size, 
                                                observation_space,
                                                observation_space_name,
                                                action_space,
                                                action_space_name,
                                                community_config_obs,
                                                community_config_act):
        """
        Partition a community consisting of #community_size agents according to
        configurations of sensors and actuators.
        
        Args:
            community_name (string): the name of the agent community
            community_size (int): the # of agents living in the community 
            observation_space (gym.spaces.Box): the sensory space of the agent 
                community
            action_space (gym.spaces.Box): the actuator space of the whole agent
                community
            observation_space_name (list of string): each entry corresponds to
                the name of sensor in observation space
            action_space_name (list of string): each entry corresponds to the 
                name of actuator in action space
            community_config_obs (dict): give the information on how to configurate 
                sensors for the community
            community_config_act (dict): give the information on how to configurate 
                actuators for the community
        
        Returns:
            agent_community_partition (dict): a dictionary of community partition 
                configuration in where:
                agent_community_partition = {'agent_name_1': {'obs_mask':agent_obs_mask,
                                                              'act_mask':agent_act_mask,
                                                              'observation_space':observation_space,
                                                              'action_space':action_space
                                                              }}
        """
        agent_community_partition = {}
        # Find partition mask for observation and action
        for agent_conf_index in self.community_config_obs.keys():
            agent_name = community_name + '_' + agent_conf_index
            agent_obs_mask = np.zeros(len(observation_space_name))
            agent_act_mask = np.zeros(len(action_space_name))
            for obs_node in self.community_config_obs[agent_conf_index]:
                for i in range(len(observation_space_name)):
                    if observation_space_name[i].endswith(obs_node):
                        agent_obs_mask[i] = 1
            for act_node in self.community_config_act[agent_conf_index]:
                for j in range(len(action_space_name)):
                    if action_space_name[j].endswith(act_node):
                        agent_act_mask[j] = 1
            agent_community_partition[agent_name] = {}
            agent_community_partition[agent_name]['obs_mask'] = agent_obs_mask
            agent_community_partition[agent_name]['act_mask'] = agent_act_mask
        # Create observation and action space and their corresponding name 
        # for each agent
        for agent_name in agent_community_partition.keys():
            # observation
            obs_dim = int(np.sum(agent_community_partition[agent_name]['obs_mask']))
            obs_low = np.zeros(obs_dim)
            obs_high = np.zeros(obs_dim)
            obs_name = [] # name for observation entry
            obs_temp_i = 0
            for obs_i in range(len(agent_community_partition[agent_name]['obs_mask'])):
                if agent_community_partition[agent_name]['obs_mask'][obs_i] == 1:
                    obs_low[obs_temp_i] = observation_space.low[obs_i]
                    obs_high[obs_temp_i] = observation_space.high[obs_i]
                    obs_name.append(observation_space_name[obs_i])
                    obs_temp_i += 1
            # action
            act_dim = int(np.sum(agent_community_partition[agent_name]['act_mask']))
            act_low = np.zeros(act_dim)
            act_high = np.zeros(act_dim)
            act_name = [] # name for action entry
            act_temp_i = 0
            for act_i in range(len(agent_community_partition[agent_name]['act_mask'])):
                if agent_community_partition[agent_name]['act_mask'][act_i] == 1:
                    act_low[act_temp_i] = action_space.low[act_i]
                    act_high[act_temp_i] = action_space.high[act_i]
                    act_name.append(action_space_name[act_i])
                    act_temp_i += 1
            # Generate observation_space accroding to partition configuration
            agent_community_partition[agent_name]['observation_space'] = spaces.Box(low=obs_low,high=obs_high, dtype = np.float32)
            agent_community_partition[agent_name]['observation_space_name'] = obs_name
            agent_community_partition[agent_name]['action_space'] = spaces.Box(low=act_low, high=act_high, dtype = np.float32)
            agent_community_partition[agent_name]['action_space_name'] = act_name
        return agent_community_partition
        
    def _create_agent_community(self, agent_community_partition_config,
                                x_order_sensor_reading = 20,
                                x_order_sensor_reading_sliding_window = 5,
                                x_order_sensor_reading_preprocess_type = 'concatenate_sensory_readings',
                                occupancy_reward_type = 'IR_distance',
                                interaction_mode = 'real_interaction',
                                load_pretrained_agent_flag = False,
                                agent_config = 'random_agent'):
        """
        Create agent community according to community partition configuration
        and agent configuration.
        
        Args:
            agent_community_partition_config (dict of dict): contains information 
                on how to partition observation and action space
        
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
            agent_config (string): information on how to configurate each agent:
                    1. 'random_agent': random agent
                    2. 'actor_critic_agent': actor_critic agent
        
        Returns:
            agent_community (dict): a dict of agent living in the community:
                agent_community = {'agent_name': agent}
        """
        agent_community = {}
        if agent_config == 'random_agent':
            for agent_name in agent_community_partition_config.keys():
                observation_space = agent_community_partition_config[agent_name]['observation_space']
                action_space = agent_community_partition_config[agent_name]['action_space']
                # Instantiate learning agent
                agent_community[agent_name] = RandomLASAgent(observation_space, action_space)
                print('Create random_agent community done!')
        elif agent_config == 'actor_critic_agent':
            for agent_name in agent_community_partition_config.keys():
                observation_space = agent_community_partition_config[agent_name]['observation_space']
                action_space = agent_community_partition_config[agent_name]['action_space']
                observation_space_name = agent_community_partition_config[agent_name]['observation_space_name']
                action_space_name = agent_community_partition_config[agent_name]['action_space_name']
                
                agent_community[agent_name] = InternalEnvOfAgent(agent_name, 
                                                                 observation_space, 
                                                                 action_space,
                                                                 observation_space_name, 
                                                                 action_space_name,
                                                                 x_order_sensor_reading,
                                                                 x_order_sensor_reading_sliding_window,
                                                                 x_order_sensor_reading_preprocess_type,
                                                                 occupancy_reward_type,
                                                                 interaction_mode,
                                                                 load_pretrained_agent_flag)
            logger.info('Create actor_critic_agent community done!')
        else:
            raise Exception('Please choose a right agent type!')
        return agent_community
        
    def _partition_observation(self, observation, agent_community_partition_config):
        """
        Partition whole observation into each agent's observation field according
        to community partition configuration.
        
        Args:
            observation (list): observation of whole external environment.
            agent_community_partition_config (dict of dict): info on how to 
                partition whole observation and action.
        
        Returns:
            observation_partition (dict): a dist of partitioned observation 
                where each value corresponds to the observation of one agent:
                    observation_partition = {'agent_name': observation}
        """
        observation_partition = {}
        for agent_name in agent_community_partition_config.keys():
            obs_index = []
            obs_index = np.where(agent_community_partition_config[agent_name]['obs_mask'] == 1)
            observation_temp = []
            observation_temp = observation[obs_index]
            observation_partition[agent_name] = observation_temp
        return observation_partition
    
    # TODO: _partition_reward is useless in real interaction.
    def _partition_reward(self, observation_partition, 
                          agent_community_partition_config,
                          reward_type = 'IR_distance'):
        """
        Partition reward based on observation_partition and agent_community_partition_config
        
        Args:
            observation_partition (dict):
                observation_partition = {'agent_name': observation}
            agent_community_partition_config (dict of dict):
                agent_community_partition_config = {'agent_name_1': {'obs_mask':agent_obs_mask,
                                                                     'act_mask':agent_act_mask,
                                                                     'observation_space':observation_space,
                                                                     'action_space':action_space
                                                                     }}
            reward_type (string):
                1. 'IR_distance': based on IR distance from detected object to IR
                2. 'IR_state_ratio': the ratio of # of detected objects and all # 
                        of IR sensors 
                3. 'IR_state_number': the number of detected objects
        
        Returns:
            reward_partition (dict): a dict of reward:
                    reward_partition = {'agent_name': reward}
        
        """
        reward_partition = {}
        for agent_name in observation_partition.keys():
            IR_data = []
            for i, name in enumerate(agent_community_partition_config[agent_name]['observation_space_name']):
                # Get IR sensor info
                if 'ir_node' in name:
                    IR_data.append(observation_partition[agent_name][i])
            # Make here insistent with IR data
            # 1. 'IR_distance': sum of reciprocal of distance from detected 
            #                   object to IR.
            # 2. 'IR_state': ratio of # of detected objects and # of IR
            reward_temp = 0.0
            if reward_type == 'IR_distance':
                for distance in IR_data:
                    if distance != 0:
                        reward_temp += 1/distance
            elif reward_type == 'IR_state_ratio':
                for distance in IR_data:
                    if distance != 0:
                        reward_temp += 1
                reward_temp = reward_temp / len(IR_data)
            elif reward_type == 'IR_state_number':
                for distance in IR_data:
                    if distance != 0:
                        reward_temp += 1
            else:
                raise Exception('Please choose a proper reward type!')
            # Average occupancy
            reward_partition[agent_name] = reward_temp
        return reward_partition
    
    def _combine_action(self, action_partition, agent_community_partition_config):
        """
        Combine each agent's action into a whole action.
        
        Args:
            action_partition (dict): a dict of actions:
                    action_partition = {'agent_name': action}
            agent_community_partition_config (dict of dict): contains info on 
                how to partition whole observation and action.
        
        Returns:
            action (list): an array of action on the whole action space
        """
        action = np.zeros(self.action_space.shape)
        for agent_name in agent_community_partition_config.keys():
            act_index = []
            act_index = np.where(agent_community_partition_config[agent_name]['act_mask']==1)
            action[act_index] = action_partition[agent_name]
        return action        
        
    def stop(self):
        """
        This interface function is to save trained models.
        """
        # Safely stop each agent living in the agent_community
        for agent_name in self.agent_community.keys():
            self.agent_community[agent_name].stop()
        
    def feed_observation(self, observation, external_reward = 0, done = False):
        """
        This interface function receives observation from environment, but
        produce an action with a different frequency.
        
        If take_action_flag == Ture, there is a valid action can be taken.
        
        (Training could also be done when feeding observation.)
        
        Args:
            observation (list): the observation received from external environment
            external_reward (float): only provied when using virtual environment 
                (ignored when interact with real system)
            done (bool): only provied when using virtual environment
                (ignored when interact with real system)
        
        Returns:
            take_action_flag (bool): indicate whether to take an action
            action (list): the action value
        """
        action_partition = {}
        take_action_flag_partition = {}
        # Partition observation
        observation_partition = self._partition_observation(observation, self.agent_community_partition_config)
        # TODO: For virtual env, let's see how to particition external reward
        reward_partition = self._partition_reward(observation_partition, self.agent_community_partition_config,
                                                  self.occupancy_reward_type)
        # Collect actions 
        for agent_name_temp in observation_partition.keys():
            take_action_flag_partition[agent_name_temp], action_partition[agent_name_temp] = self.agent_community[agent_name_temp].feed_observation(observation_partition[agent_name_temp], reward_partition[agent_name_temp], done = False)
        # TODO: here take action synchronously among all agents. In future, we
        #   can make it  synchroneously.
        take_action_flag = take_action_flag_partition[agent_name_temp]
        # Combine actions
        if take_action_flag == True:
            action = self._combine_action(action_partition, self.agent_community_partition_config)
        else:
            action = []
        
        self._logging_interaction_data(observation,
                                       observation_partition,
                                       reward_partition,
                                       action_partition,
                                       take_action_flag_partition,
                                       take_action_flag,
                                       action)
        
        return take_action_flag, action
        
    
    def _logging_interaction_data(self, observation,
                                  observation_partition,
                                  reward_partition,
                                  action_partition,
                                  take_action_flag_partition,
                                  take_action_flag,
                                  action):
        """
        Saving interaction data
        
        Args:
            observation:
            observation_partition:
            reward_partition:
            action_partition:
            take_action_flag_partition:
            take_action_flag:
            action:
        """
        with open(self.interaction_data_file, 'a') as csv_datafile:
            fieldnames = ['time', 'observation', 'observation_partition', 'reward_partition',
                          'action_partition', 'take_action_flag_partition',
                          'take_action_flag', 'action']
            writer = csv.DictWriter(csv_datafile, fieldnames = fieldnames)
            writer.writerow({'time':datetime.now().strftime("%Y%m%d-%H%M%S"),
                             'observation': observation,
                             'observation_partition': observation_partition,
                             'reward_partition': reward_partition,
                             'action_partition': action_partition,
                             'take_action_flag_partition': take_action_flag_partition,
                             'take_action_flag': take_action_flag,
                             'action':action})
# =================================================================== #
#                   Initialization Summary Functions                  #
# =================================================================== #     










    

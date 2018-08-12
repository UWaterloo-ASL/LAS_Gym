#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  8 21:47:50 2018

@author: jack.lingheng.meng
"""
import numpy as np
from gym import spaces
from LASAgent.InternalEnvOfAgent import InternalEnvOfAgent
from LASAgent.RandomLASAgent import RandomLASAgent

class InternalEnvOfCommunity(object):
    """
    This class provides an internal environment for a community of agents to 
    interact with external environment.
    """
    def __init__(self, sess, community_name, community_size,
                 observation_space, observation_space_name,
                 action_space, action_space_name,
                 interaction_mode = 'real_interaction'):
        """
        Initialize internal environment for an agent
        Parameters
        ----------
        community_name: string
            the name of the community this internal environment serves for
        community_size: int
            the # of agents living in the community 
        observation_space: gym.spaces.Box datatype
            observation space of "agent_name"
        observation_space_name: list of string
            gives the name of each entry in observation space
        action_space: gym.spaces.Box datatype
            action space of "agent_name"
        action_space_name: list of strings
            gives the name of each entry in action space
        interaction_mode: string default = 'real_interaction'
            indicate interaction mode: 
                1) 'real_interaction': interact with real robot
                2) 'virtual_interaction': interact with virtual environment
        """
        self.tf_session = sess
        self.interaction_mode = interaction_mode
        # Initialize community
        self.community_name = community_name
        self.community_size = community_size
        self.community_config_obs = {'agent_1':['node','node#22','node#21','node#20','node#19','node#18','node#17','node#16','node#15'],
                                     'agent_2':['node#16','node#15','node#14','node#13','node#12','node#11','node#10','node#9','node#8'],
                                     'agent_3':['node#9','node#8','node#7','node#6','node#5','node#4','node#3','node#2','node#1','node#0']
                                     }
        self.community_config_act = {'agent_1':['node','node#22','node#21','node#20','node#19','node#18','node#17'],
                                     'agent_2':['node#16','node#15','node#14','node#13','node#12','node#11','node#10','node#9','node#8'],
                                     'agent_3':['node#7','node#6','node#5','node#4','node#3','node#2','node#1','node#0']
                                     }
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
        self.agent_community = self._create_agent_community(self.agent_community_partition_config,
                                                            agent_config = 'actor_critic_agent')  # agent_config = 'random_agent'
        
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
        # Partition observation
        observation_partition = self._partition_observation(observation,
                                                            self.agent_community_partition_config)
        # Partition reward
        reward_partition = self._partition_reward(observation_partition,
                                                  self.agent_community_partition_config)
        for agent_name in reward_partition.keys():
            print('Reward of {} is: {}'.format(agent_name,reward_partition[agent_name]))
        # Collect actions from each agent
        action_partition = self._collect_action(observation_partition,
                                                reward_partition,
                                                self.agent_community)
        # Combine actions from agents
        action = self._combine_action(action_partition, self.agent_community_partition_config)
        return action
    
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
        configuration.
        
        Parameters
        ----------
        community_name: string
            the name of the community this internal environment serves for
        community_size: int
            size of community i.e. # of agents in the community
        observation_space: gym.spaces.Box datatype
            observation space of "agent_name"
        observation_space_name: list of string
            gives the name of each entry in observation space
        action_space: gym.spaces.Box datatype
            action space of "agent_name"
        action_space_name: list of strings
            gives the name of each entry in action space
        community_config_obs: 
            give the information on how to configurate observation for the community
        community_config_act:
            give the information on how to configurate action for the community
        
        Returns
        -------
        agent_community_partition: dictionary
            a dictionary of community partition configuration in where:
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
            agent_community_partition[agent_name]['observation_space'] = spaces.Box(low=obs_low,high=obs_high, dtype = np.float32)
            agent_community_partition[agent_name]['observation_space_name'] = obs_name
            agent_community_partition[agent_name]['action_space'] = spaces.Box(low=act_low, high=act_high, dtype = np.float32)
            agent_community_partition[agent_name]['action_space_name'] = act_name
        return agent_community_partition
        
    def _create_agent_community(self, agent_community_partition_config, 
                                agent_config = 'random_agent'):
        """
        Create agent community according to community partition configuration
        and agent configuration.
        
        Parameters
        ----------
        agent_community_partition_config: dict of dict
            contains information on how to partition observation and action space
            
        agent_config: (not determined)
            contains information on how to configurate each agent:
                'random_agent': random agent
                'actor_critic_agent': actor_critic agent
        
        Returns
        -------
        agent_community: dict
            a dict of agent living in the community:
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
                # Instantiate LAS-agent
                agent_community[agent_name] = InternalEnvOfAgent(self.tf_session,\
                                                                 agent_name,
                                                                 observation_space,
                                                                 action_space,
                                                                 interaction_mode = 'virtual_interaction')
            print('Create actor_critic_agent community done!')
        else:
            raise Exception('Please choose a right agent type!')
        return agent_community
        
    def _partition_observation(self, observation, agent_community_partition_config):
        """
        Partition whole observation into each agent's observation field according
        to community partition configuration.
        
        Parameters
        ----------
        observation: ndarray
            observation of whole external environment.
            
        agent_community_partition_config: dict of dict
            contains info on how to partition whole observation and action.
        
        Returns
        -------
        observation_partition: dict
            a dist of observation where each value corresponds to the observation
            of one agent:
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
    
    def _partition_reward(self, observation_partition, 
                          agent_community_partition_config):
        """
        Partition reward based on observation_partition and agent_community_partition_config
        
        Parameters
        ----------
        observation_partition: dict
            observation_partition = {'agent_name': observation}
            
        agent_community_partition_config: dict of dict
            agent_community_partition_config = {'agent_name_1': {'obs_mask':agent_obs_mask,
                                                                 'act_mask':agent_act_mask,
                                                                 'observation_space':observation_space,
                                                                 'action_space':action_space
                                                                 }}
        Returns
        -------
        reward_partition: dict
            a dict of reward:
                reward_partition = {'agent_name': reward}
        
        """
        reward_partition = {}
        for agent_name in observation_partition.keys():
            obs_interest = np.zeros(agent_community_partition_config[agent_name]['observation_space'].shape)
            for i, name in enumerate(agent_community_partition_config[agent_name]['observation_space_name']):
                # Get IR sensor info
                if 'ir_node' in name:
                    obs_interest[i] = observation_partition[agent_name][i]
            # Make here insistent with IR data
            reward_temp = 0.0
            for distance in obs_interest:
                if distance != 0:
                    reward_temp += 1/distance
            reward_partition[agent_name] = reward_temp
        return reward_partition
    
    def _collect_action(self, observation_partition, reward_partition, agent_community):
        """
        Collect actions from each agent into a dict.
        
        Parameters
        ----------
        observation_partition: dict
            a dict of observation partitions:
                observation_partition = {'agent_name': observation}
        
        reward_partition: dict
            a dict of reward partitions:
                reward_partition = {'agent_name': reward}
        
        agent_community: dict
            a dict of agents:
                agent_community = {'agent_name': agent_object}
        
        Returns
        -------
        action_partition: dict
            a dict of actions:
                action_partition = {'agent_name': action}
        
        """
        done = False
        action_partition = {}
        for agent_name in agent_community.keys():
            action_partition[agent_name] = agent_community[agent_name].interact(observation_partition[agent_name],\
                            reward_partition[agent_name],done)
        return action_partition
    
    def _combine_action(self, action_partition, agent_community_partition_config):
        """
        Combine each agent's action into a whole action.
        
        Parameters
        ----------
        action_partition: dict
            a dict of actions:
                action_partition = {'agent_name': action}
        
        agent_community_partition_config: dict of dict
            contains info on how to partition whole observation and action.
        
        Returns
        -------
        action: ndarray
            an array of action on the whole action space
        """
        action = np.zeros(self.action_space.shape)
        for agent_name in agent_community_partition_config.keys():
            act_index = []
            act_index = np.where(agent_community_partition_config[agent_name]['act_mask']==1)
            action[act_index] = action_partition[agent_name]
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
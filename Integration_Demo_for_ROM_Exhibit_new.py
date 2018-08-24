#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 23 00:37:13 2018

@author: jack.lingheng.meng
"""
import logging
import tensorflow as tf
import numpy as np
import time

from datetime import datetime, date
from threading import Timer

import os



from Environment.LASEnv import LASEnv
from LASAgent.InternalEnvOfAgent import InternalEnvOfAgent
from LASAgent.InternalEnvOfCommunity import InternalEnvOfCommunity

# Logging
logging.basicConfig(filename = '../ROM_Experiment_results/ROM_experiment_'+datetime.now().strftime("%Y%m%d-%H%M%S")+'.log', 
                    level = logging.DEBUG,
                    format='%(asctime)s:%(levelname)s: %(message)s')

#######################################################################
#                 Instatiate LAS virtual environment                  #
#######################################################################
# Instantiate LAS environment object
envLAS = LASEnv('127.0.0.1', 19997, reward_function_type = 'occupancy')
observation = envLAS.reset()

#######################################################################
#                          Instatiate LAS-Agent                       #
#######################################################################
# Note: 1. Set load_pretrained_agent_flag to "True" only when you have 
#           and want to load pretrained agent.
#       2. Keep observation unchanged if using pretrained agent.
agent_name = 'LAS_Single_Agent'
observation_space = envLAS.observation_space
action_space = envLAS.action_space
observation_space_name = [], 
action_space_name = []
x_order_MDP = 5
x_order_MDP_observation_type = 'concatenate_observation'
occupancy_reward_type = 'IR_distance'
interaction_mode = 'real_interaction'
load_pretrained_agent_flag = False

single_agent = InternalEnvOfAgent(agent_name, 
                           observation_space, 
                           action_space,
                           observation_space_name, 
                           action_space_name,
                           x_order_MDP,
                           x_order_MDP_observation_type,
                           occupancy_reward_type,
                           interaction_mode,
                           load_pretrained_agent_flag)
logging.info('Instantiate LAS-Agent done!')
#######################################################################
#                    Instatiate LAS-Agent-Community                   #
#######################################################################
# Note: 1. Set load_pretrained_agent_flag to "True" only when you have and want 
#          to load pretrained agent.
#       2. Keep observation unchanged if using pretrained agent.
community_name = 'LAS_Agent_Community'
community_size = 3
x_order_MDP = 5
x_order_MDP_observation_type = 'concatenate_observation'
occupancy_reward_type = 'IR_distance'
interaction_mode = 'real_interaction'
load_pretrained_agent_flag = False

LAS_agent_community = InternalEnvOfCommunity(community_name, 
                                             community_size,
                                             envLAS.observation_space,
                                             envLAS.action_space, 
                                             envLAS.observation_space_name,
                                             envLAS.action_space_name,
                                             x_order_MDP,
                                             x_order_MDP_observation_type,
                                             occupancy_reward_type,
                                             interaction_mode,
                                             load_pretrained_agent_flag)
logging.info('Instantiate LAS-Agent-Community done!')


#######################################################################
#                      Schedual two experiments                       #
# Note:
#   1. Initialize Single_Agent and Agent_Community will take about 10 minutes.
#      Thus, the master script should be run before 9:45am
#   2. Single_Agent.stop() will take about 3 minutes. Thus, if first experiment
#      is stopped at 2:30pm, the second experiment should start at 2:35pm
#   3. Agent_Community.stop() will take about 10 minutes. Thus, if the second 
#      experiment is stopped at 4:00pm, the baseline bahavior should start at 
#      4:15pm.
# Solution: to get rid of time-gap when switching behavior modes, use multiple
#      threads to do Single_Agent.stop() and Agent_Community.stop().
#######################################################################

def interact_with_learning_agent(agent, env, end_time = '143000'):
    """
    
    """
    logging.info('{}: Start interaction. Default End_time: {}'.format(agent.name, end_time))
    # Interact untill end_time or interrupted by 'Ctrl+c'
    while not datetime.now().strftime("%H%M%S") > end_time:
        observation = env._self_observe()
        take_action_flag, action = agent.feed_observation(observation)
        if take_action_flag == True:
            observation, _, _, _ = env.step(action)
    # Save learned model
    logging.info('{}: Interaction is done. Saving learned models...'.format(agent.name))
    agent.stop()
    logging.info('{}: Saving learned models done.'.format(agent.name))
    

def interact_with_prescribed_behavior():
    """
    TODO: Please put prescribe behavior in this function.
    """
    pass


open_time = datetime.now()

# Schedule first experiment
first_experiment_start_time = '225500'  # format: %H%M%S e.g. 1:00pm is 130000
first_experiment_end_time = '225800'    # format: %H%M%S e.g. 2:30pm is 143000

first_experiment_start_delay = (datetime.strptime(date.today().strftime("%Y%m%d")+'-'+first_experiment_start_time, '%Y%m%d-%H%M%S') - open_time).total_seconds()
if first_experiment_start_delay < 0:
    logging.error('First Experiment starts earlier than the open-time of ROM!')

first_experiment_thread = Timer(interval = first_experiment_start_delay,
                                function = interact_with_learning_agent,
                                kwargs={'agent': single_agent,
                                        'env': envLAS,
                                        'end_time': first_experiment_end_time})

# Schedule second experiment
second_experiment_start_time = '225801' # format: %H%M%S e.g. 2:30pm is 143000
second_experiment_end_time = '230500'   # format: %H%M%S e.g. 4:00pm is 160000

second_experiment_start_delay = (datetime.strptime(date.today().strftime("%Y%m%d")+'-'+second_experiment_start_time, '%Y%m%d-%H%M%S') - open_time).total_seconds()
if second_experiment_start_delay < 0:
    logging.error('Second Experiment starts earlier than the end of First Experiment!')

second_experiment_thread = Timer(interval = second_experiment_start_delay,
                                 function = interact_with_learning_agent,
                                 kwargs={'agent': LAS_agent_community,
                                         'env': envLAS,
                                         'end_time': second_experiment_end_time})
    
if __name__ == '__main__':
    
    # Run two experiments
    first_experiment_thread.start()
    logging.info('first_experiment_thread started...')
    second_experiment_thread.start()
    logging.info('second_experiment_thread started...')

        
  
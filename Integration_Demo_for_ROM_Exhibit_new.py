#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 23 00:37:13 2018

@author: jack.lingheng.meng
"""
import logging
from datetime import datetime, date
from threading import Timer
import os

from gym import spaces
import numpy as np

from Environment.LASEnv import LASEnv
from LASAgent.InternalEnvOfAgent import InternalEnvOfAgent
from LASAgent.InternalEnvOfCommunity import InternalEnvOfCommunity

# Logging
experiment_results_dir = os.path.join(os.path.abspath('..'), 'ROM_Experiment_results')
if not os.path.exists(experiment_results_dir):
    os.makedirs(experiment_results_dir)
logging.basicConfig(filename = os.path.join(experiment_results_dir,'ROM_experiment_'+datetime.now().strftime("%Y%m%d_%H%M%S")+'.log'), 
                    level = logging.DEBUG,
                    format='%(asctime)s:%(levelname)s: %(message)s')

#######################################################################
#                 Instatiate LAS virtual environment                  #
#  TODO: This part is not needed when interacting with real exhibit   #
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
# TODO: Adam provides information and we will define these variables.
#   In realy system, there is no virtual environment. We need to define:
#       1. observation_space: gym.spaces.Box object
#       2. observation_space_name: each entry is the name of sensor
#       3. action_space: gym.spaces.Box object
#       4. action_space_name: ench entry is the name of actuator
agent_name = 'LAS_Single_Agent'
observation_space = envLAS.observation_space
action_space = envLAS.action_space
observation_space_name = [], 
action_space_name = []

x_order_sensor_reading = 20
x_order_sensor_reading_sliding_window = 5
x_order_sensor_reading_preprocess_type = 'max_pool_sensory_readings'#'average_pool_sensory_readings'#'concatenate_sensory_readings'
occupancy_reward_type = 'IR_distance'
interaction_mode = 'real_interaction'
load_pretrained_agent_flag = True

single_agent = InternalEnvOfAgent(agent_name,
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
logging.info('Instantiate LAS-Agent done!')
#######################################################################
#                    Instatiate LAS-Agent-Community                   #
#######################################################################
# Note: 1. Set load_pretrained_agent_flag to "True" only when you have and want 
#          to load pretrained agent.
#       2. Keep observation unchanged if using pretrained agent.
community_name = 'LAS_Agent_Community'
community_size = 3
observation_space = envLAS.observation_space
action_space = envLAS.action_space
observation_space_name = envLAS.observation_space_name
action_space_name = envLAS.action_space_name
x_order_sensor_reading = 20
x_order_sensor_reading_sliding_window = 5
x_order_sensor_reading_preprocess_type = 'max_pool_sensory_readings'#'average_pool_sensory_readings'#'concatenate_sensory_readings'
occupancy_reward_type = 'IR_distance'
interaction_mode = 'real_interaction'
load_pretrained_agent_flag = True

LAS_agent_community = InternalEnvOfCommunity(community_name, 
                                             community_size,
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
logging.info('Instantiate LAS-Agent-Community done!')
#######################################################################
#                      Schedual two experiments                       #
# Note:
#   1. Initialize Single_Agent and Agent_Community will take about 10 minutes.
#      Thus, the master script should be run, at least, before 9:45am everyday.
#   2. Single_Agent.stop() will take about 3 minutes to save learned models.
#   3. Agent_Community.stop() will take about 10 minutes to save learned models.
# Problem:
#   We don't want visitors to feel the pause when saving learned models
# Solution: 
#   To get rid of time-gap when switching behavior modes, use multiple threads
#   to interact with visitors with different interaction modes i.e. when one 
#   interaction mode is done, another interaction mode starts interacting 
#   immediately at the same time the previous thread keeps saving learned models.
#######################################################################

def interact_with_learning_agent(agent, end_time = '143000'):
    """
    Note:
        When integrate with master_script, replease:
            1. envLAS._self_observe() -> get_obseravtion()
            2. envLAS.step(action) -> take_action(action)
    Parameters
    ----------
    agent: learning agent object
    
    end_time: str (in format %HH%MM%SS)
        the end time of interaction
    """
    logging.info('{}: Start interaction. Default End_time: {}'.format(agent.name, end_time))
    # Interact untill end_time
    while not datetime.now().strftime("%H%M%S") > end_time:
        observation = envLAS._self_observe()    # TODO: replace with get_observation()
        take_action_flag, action = agent.feed_observation(observation)
        if take_action_flag == True:
            observation, _, _, _ = envLAS.step(action)  # TODO: replace with take_action(action)
    # Save learned model
    logging.info('{}: Interaction is done. Saving learned models...'.format(agent.name))
    agent.stop()
    logging.info('{}: Saving learned models done.'.format(agent.name))
    

def interact_with_prescribed_behavior(agent = 'prescribed_behavior', end_time = '130000'):
    """
    TODO: Please put prescribed behavior in this function.
    
    Parameters
    ----------
    agent: str
        not important paramter just for keeping the same format with interact_with_learning_agent
    
    end_time: str (in format %HH%MM%SS)
        the end time of interaction
    """
    logging.info('{}: Start interaction. Default End_time: {}'.format(agent, end_time))
    # Interact untill end_time
    while not datetime.now().strftime("%H%M%S") > end_time:
        observation = envLAS._self_observe()    # TODO: replace with get_observation()
        action = envLAS.action_space.sample()   # TODO: replace with prescribed-behavior
        observation, _, _, _ = envLAS.step(action) # TODO: replace with take_action(action)
    logging.info('{}: Interaction is done.'.format(agent))

def interaction_mode_scheduler(interaction_mode, agent, 
                                start_time, end_time, schedule_start_time):
    """
    Parameters
    ----------
    interaction_mode: func
        function
    
    agent: depends on interaction mode
        1. agent object: for learning agent
        2. 'priscribed_behavior': for priscribed behavior
        
    start_time: str (with format'hhmmss')
        
    end_time: str (with format'hhmmss')
    
    schedule_start_time: datetime object
        
    Returns
    -------
    interaction_thread
        a delayed thread for an interaction mode which will start at a given time.
    """
    start_delay = (datetime.strptime(date.today().strftime("%Y%m%d")+'-'+start_time, '%Y%m%d-%H%M%S') - schedule_start_time).total_seconds()
    if start_delay < 0:
        logging.error('{} starts earlier than schedualing time!'.format(interaction_mode.__name__))

    interaction_thread = Timer(interval = start_delay,
                               function = interaction_mode,
                                    kwargs={'agent': agent,
                                            'end_time': end_time})
    return interaction_thread

# Get current time to calculate interaction start-time-delay
schedule_start_time = datetime.now()

# Schedule first experiment
# TODO: set start and end times to '130002' and '143000'
first_experiment_start_time = '175501'  # format: %H%M%S e.g. 1:00pm is 130000
first_experiment_end_time = '180500'    # format: %H%M%S e.g. 2:30pm is 143000
first_experiment_thread = interaction_mode_scheduler(interact_with_learning_agent,
                                                     single_agent,
                                                     first_experiment_start_time, 
                                                     first_experiment_end_time, 
                                                     schedule_start_time)
# Schedule second experiment
# TODO: set start and end times to '143002' and '160000'
second_experiment_start_time = '180501' # format: %H%M%S e.g. 2:30pm is 143000
second_experiment_end_time = '182500'   # format: %H%M%S e.g. 4:00pm is 160000
second_experiment_thread = interaction_mode_scheduler(interact_with_learning_agent, 
                                                      LAS_agent_community,
                                                      second_experiment_start_time, 
                                                      second_experiment_end_time, 
                                                      schedule_start_time)
# Schedule prescribed-behavior 1
# Note: 
#   Make sure to leave an, at least 10 minuts, time-gap between the time you 
#   start thsi script and the start time for the first interaction. 
#   (This is because instantiating learning agent takes around 8 minutes.)
# TODO: set start and end times to '093000' and '130000'
prescribed_behavior_start_time_1 = '175301' # format: %H%M%S e.g. 10:00am is 100000
prescribed_behavior_end_time_1 = '175500'   # format: %H%M%S e.g. 1:00pm is 130000
prescribed_behavior_thread_1 = interaction_mode_scheduler(interact_with_prescribed_behavior,
                                                          'prescribed_behavior',
                                                          prescribed_behavior_start_time_1,
                                                          prescribed_behavior_end_time_1, 
                                                          schedule_start_time)
# Schedule prescribed-behavior 2
# TODO: set start and end times to '160002' and '173000'
prescribed_behavior_start_time_2 = '182501' # format: %H%M%S e.g. 4:00pm is 160000
prescribed_behavior_end_time_2 = '182800'   # format: %H%M%S e.g. 5:30pm is 173000
prescribed_behavior_thread_2 = interaction_mode_scheduler(interact_with_prescribed_behavior,
                                                          'prescribed_behavior',
                                                          prescribed_behavior_start_time_2, 
                                                          prescribed_behavior_end_time_2, 
                                                          schedule_start_time)
if __name__ == '__main__':
    
    # TODO: put initialization work for master script in here
    
    # Schedule interaction with learning agent
    first_experiment_thread.start()
    logging.info('first_experiment_thread scheduled: {}-{}'.format(first_experiment_start_time, first_experiment_end_time))
    second_experiment_thread.start()
    logging.info('second_experiment_thread scheduled: {}-{}'.format(second_experiment_start_time, second_experiment_end_time))
    # Schedule interaction with presribed-behavior
    prescribed_behavior_thread_1.start()
    logging.info('prescribed_behavior_thread_1 scheduled: {}-{}'.format(prescribed_behavior_start_time_1, prescribed_behavior_end_time_1))
    prescribed_behavior_thread_2.start()
    logging.info('prescribed_behavior_thread_2 scheduled: {}-{}'.format(prescribed_behavior_start_time_2, prescribed_behavior_end_time_2))
    
    # Check if all interactions are done.
    while True:
        if not first_experiment_thread.is_alive()\
                and not second_experiment_thread.is_alive()\
                and not prescribed_behavior_thread_1.is_alive()\
                and not prescribed_behavior_thread_2.is_alive():
           logging.info('All interactions are done.')
           break
  
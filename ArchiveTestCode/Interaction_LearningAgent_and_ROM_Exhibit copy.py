#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 23 00:37:13 2018

@author: jack.lingheng.meng
"""

import tensorflow as tf
import numpy as np
import time
from datetime import datetime, timedelta
import os

import sched, time

from Environment.LASEnv import LASEnv
from LASAgent.InternalEnvOfAgent import InternalEnvOfAgent
from LASAgent.InternalEnvOfCommunity import InternalEnvOfCommunity

###############################################################################
#                          Only Used For Summary                              #
###############################################################################
def calculate_total_bright_light_number(light_intensity,
                                        bright_light_threshold = 0.95):
    """
    calculate the # of bright light.
    
    Parameters
    ----------
    light_intensity: array
        
    bright_light_threshold: float default = 0.95
        
    """
    bright_light_number = 0
    for intensity in light_intensity:
        if intensity >= bright_light_threshold:
            bright_light_number += 1
    return bright_light_number

def _init_summarize_total_bright_light_number():
        """
        Summarize the # of bright light.
        """
        bright_light_number = tf.placeholder(dtype = tf.float32)
        bright_light_number_sum = tf.summary.scalar('total_bright_light_number', bright_light_number)
        bright_light_number_sum_op = tf.summary.merge([bright_light_number_sum])
        return bright_light_number_sum_op, bright_light_number
# Summary directory
save_dir = os.path.join(os.path.abspath('..'),'ROM_Experiment_results', 'Overall_Summary_Single_Agent')
summary_dir = os.path.join(save_dir,datetime.now().strftime("%Y%m%d-%H%M%S"))
if not os.path.isdir(summary_dir):
    os.makedirs(summary_dir)

tf_writer = tf.summary.FileWriter(summary_dir)
# Summarize # of bright light
total_bright_light_number_sum_op,\
total_bright_light_number_sum = _init_summarize_total_bright_light_number()
###############################################################################

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
interaction_mode = 'virtual_interaction'
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
interaction_mode = 'virtual_interaction'
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
#######################################################################
#                      Schedual two experiments                       #
#######################################################################

def interact_with_learning_agent(agent, env, end_time = '143000'):
    try:
        # Interact untill end_time or interrupted by 'Ctrl+c'
        while not datetime.now().strftime("%H%M%S") > end_time:
            observation = env._self_observe()
            take_action_flag, action = agent.feed_observation(observation)
            if take_action_flag == True:
                observation, _, _, _ = env.step(action)
    except KeyboardInterrupt:
        # Save learned model
        agent.stop()

scheduler = sched.scheduler(time.time, time.sleep)
# Assume ROM open at 10am and this script is run at 10am
open_time = 10
# Schedule first experiment
first_experiment_start_time = 13
first_experiment_delay = timedelta(hours = first_experiment_start_time-open_time).total_seconds()
scheduler.enter(first_experiment_delay, 1, interact_with_learning_agent,
                kwargs={'agent': single_agent,
                        'env': envLAS,
                        'end_time': '143000'})
# Schedule second experiment
second_experiment_start_time = 14.5
second_experiment_delay = timedelta(hours = second_experiment_start_time-open_time).total_seconds()
scheduler.enter(second_experiment_delay, 1, interact_with_learning_agent,
                kwargs={'agent': LAS_agent_community,
                        'env': envLAS,
                        'end_time': '160000'})

if __name__ == '__main__':
    sess = tf.Session()
    
        
    
    
    # Step counter
    i = 1
    done = False
    reward = 0
    try:
        while True:
            
#            current_time = datetime.now().strftime("%H%M%S")
#            if '130000' < current_time and current_time < '143000':
#                take_action_flag, action = single_agent.feed_observation(observation, reward, done)
#                if take_action_flag == True:
#                    observation, reward, done, info = envLAS.step(action)
#            elif '143000' < current_time and current_time < '160000':
#                take_action_flag, action = LAS_agent_community.feed_observation(observation, reward, done)
#                if take_action_flag == True:
#                    observation, reward, done, info = envLAS.step(action)
#            else:
#                action = envLAS.action_space.sample()
#                observation, reward, done, info = envLAS.step(action)
#            
#            ###################################################################
#            #                        First Experiment                         #
#            ###################################################################
#            while datetime.now().strftime("%H%M%S") > '130000' and datetime.now().strftime("%H%M%S") < '143000':
#                take_action_flag, action = single_agent.feed_observation(observation, reward, done)
#                if take_action_flag == True:
#                    observation, reward, done, info = envLAS.step(action)
#            # Save trained models
#            single_agent.stop()
#            ###################################################################
#            #                        Second Experiment                        #
#            ###################################################################
#            while datetime.now().strftime("%H%M%S") > '143000' and datetime.now().strftime("%H%M%S") < '160000':
#                take_action_flag, action = LAS_agent_community.feed_observation(observation, reward, done)
#                if take_action_flag == True:
#                    observation, reward, done, info = envLAS.step(action)
#            # Save trained models
#            LAS_agent_community.stop()
            
            ###################################################################
            #                          Summary                                #
            ###################################################################
            # This is a cheating function and only for analysis, because light
            # intensity is not perceived by any sensor.
            light_intensity = envLAS._get_all_light_intensity()
            bright_light_number = calculate_total_bright_light_number(light_intensity)
            # Summary total bright light number
            summary_str_bright_light_number = sess.run(total_bright_light_number_sum_op, 
                                                       feed_dict={total_bright_light_number_sum:bright_light_number})
            tf_writer.add_summary(summary_str_bright_light_number, i)
            ###################################################################
            
            i += 1
    except KeyboardInterrupt:
        agent.stop()
        
        sess.close()
        envLAS.destroy()
    
    
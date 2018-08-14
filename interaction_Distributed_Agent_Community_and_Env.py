#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  8 21:15:57 2018

@author: jack.lingheng.meng
"""



import tensorflow as tf
import numpy as np
import time
import os
from datetime import datetime

from Environment.LASEnv import LASEnv
from LASAgent.InternalEnvOfAgent import InternalEnvOfAgent
from LASAgent.InternalEnvOfCommunity import InternalEnvOfCommunity

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
save_dir = os.path.join(os.path.abspath('..'),'ROM_Experiment_results', 'Overall_Summary_Agent_Community')
summary_dir = os.path.join(save_dir,datetime.now().strftime("%Y%m%d-%H%M%S"))
if not os.path.isdir(summary_dir):
    os.makedirs(summary_dir)

tf_writer = tf.summary.FileWriter(summary_dir)
# Summarize # of bright light
total_bright_light_number_sum_op,\
total_bright_light_number_sum = _init_summarize_total_bright_light_number()


if __name__ == '__main__':
    
    with tf.Session() as sess:
        # Instantiate LAS environment object
        envLAS = LASEnv('127.0.0.1', 19997, reward_function_type = 'occupancy')
        observation_For_LAS= envLAS.reset()
        
        # Instatiate LAS-community
        community_name = 'LAS_Agent_Community'
        community_size = 3
        LAS_agent_community = InternalEnvOfCommunity(sess, 
                                                     community_name, 
                                                     community_size,
                                                     envLAS.observation_space,
                                                     envLAS.action_space, 
                                                     envLAS.observation_space_name,
                                                     envLAS.action_space_name,
                                                     occupancy_reward_type = 'IR_distance',
                                                     interaction_mode = 'virtual_interaction')
        
        
        # Step counter
        i = 1
        done = False
        reward_for_LAS = 0
        while not done:
            # LAS_community interacts with environment.
            actionLAS = LAS_agent_community.interact(observation_For_LAS, reward_for_LAS, done)
            # delay the observing of consequence of LASAgent's action
            observation_For_LAS, reward_for_LAS, done, info = envLAS.step(actionLAS)
            
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
        
        envLAS.destroy()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  8 19:35:51 2018

@author: jack.lingheng.meng
"""
import tensorflow as tf
from datetime import datetime
import os

from Environment.LASEnv import LASEnv
from LASAgent.InternalEnvOfAgent import InternalEnvOfAgent

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

if __name__ == '__main__':
    sess = tf.Session()
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
    x_order_sensor_reading = 20
    x_order_sensor_reading_sliding_window = 5
    x_order_sensor_reading_preprocess_type = 'max_pool_sensory_readings'#'average_pool_sensory_readings'#'concatenate_sensory_readings'
    occupancy_reward_type = 'IR_distance'
    interaction_mode = 'real_interaction'
    load_pretrained_agent_flag = False
    
    agent = InternalEnvOfAgent(agent_name, 
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
    #######################################################################
    
    # Step counter
    i = 1
    done = False
    reward = 0
    try:
        while True:
            take_action_flag, action = agent.feed_observation(observation, reward, done)
            if take_action_flag == True:
                observation, reward, done, info = envLAS.step(action)
            
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
    
    
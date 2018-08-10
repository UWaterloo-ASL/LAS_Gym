#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  8 21:15:57 2018

@author: jack.lingheng.meng
"""



import tensorflow as tf
import numpy as np
import time

from Environment.LASEnv import LASEnv
from LASAgent.InternalEnvOfAgent import InternalEnvOfAgent
from LASAgent.InternalEnvOfCommunity import InternalEnvOfCommunity



if __name__ == '__main__':
    
    with tf.Session() as sess:
        # Instantiate LAS environment object
        envLAS = LASEnv('127.0.0.1', 19997, reward_function_type = 'occupancy')
        observation_For_LAS= envLAS.reset()
        
        # Instatiate LAS-community
        community_name = 'LAS_agent_community'
        community_size = 3
        LAS_agent_community = InternalEnvOfCommunity(sess, 
                                                     community_name, 
                                                     community_size,
                                                     envLAS.observation_space, 
                                                     envLAS.observation_space_name,
                                                     envLAS.action_space, 
                                                     envLAS.action_space_name,
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
            #print("LAS Step: {}, reward: {}".format(i, reward_for_LAS))
            
            i += 1
        
        envLAS.destroy()
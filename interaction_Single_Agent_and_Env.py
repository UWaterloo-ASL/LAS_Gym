#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  8 19:35:51 2018

@author: jack.lingheng.meng
"""



import tensorflow as tf
import numpy as np
import time

from Environment.LASEnv import LASEnv
from LASAgent.InternalEnvOfAgent import InternalEnvOfAgent


if __name__ == '__main__':
    
    with tf.Session() as sess:
        # Instantiate LAS environment object
        envLAS = LASEnv('127.0.0.1', 19997, reward_function_type = 'occupancy')
        observation_For_LAS= envLAS.reset()
        
        # Iinstantiate LAS-agent
        agent_name = 'single_agent'
        observation_space = envLAS.observation_space
        action_space = envLAS.action_space
        agent = InternalEnvOfAgent(sess, agent_name, observation_space, action_space,
                                   interaction_mode = 'virtual_interaction')
        
        # Step counter
        i = 1
        done = False
        reward_for_LAS = 0
        while not done:
            # LAS interacts with environment.
            actionLAS = agent.interact(observation_For_LAS, reward_for_LAS, done)
            # delay the observing of consequence of LASAgent's action
            observation_For_LAS, reward_for_LAS, done, info = envLAS.step(actionLAS)
            print("LAS Step: {}, reward: {}".format(i, reward_for_LAS))
            
            i += 1
        
        envLAS.destroy()
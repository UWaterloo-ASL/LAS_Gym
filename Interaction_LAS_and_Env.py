#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  9 08:52:36 2018

@author: jack.lingheng.meng
"""

from Environment.LASEnv import LASEnv
from LASAgent.RandomLASAgent import RandomLASAgent

if __name__ == '__main__':
    
    # Iinstantiate LAS-agent
    LASAgent1 = RandomLASAgent()

    # Instantiate environment object
    envLAS = LASEnv('127.0.0.1', 19997)
    
    observationForLAS, rewardLAS, done, info = envLAS.reset()
    
    # Step counter
    i = 1
    while not done:

        actionLAS = LASAgent1.perceive_and_act(observationForLAS, rewardLAS, done)
        observationForLAS, rewardLAS, done, info = envLAS.step_LAS(actionLAS)
        print("LAS Step: {}, reward: {}".format(i, rewardLAS))
        i += 1
    
    envLAS.destroy()

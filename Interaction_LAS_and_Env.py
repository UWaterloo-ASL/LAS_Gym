#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  9 08:52:36 2018

@author: jack.lingheng.meng
"""

import time
import numpy as np
import multiprocessing as mp

from RedLightExcitedVisitorAgent import RedLightExcitedVisitorAgent
from RedLightExcitedVisitor_LAS_Env import LivingArchitectureEnv
from LASAgent import LASAgent



if __name__ == '__main__':
    
    # Iinstantiate LAS-agent
    LASAgent1 = LASAgent()

    
    # Instantiate environment object
    envLAS = LivingArchitectureEnv('127.0.0.1', 19997)
    
    observationForLAS, rewardLAS, rewardVisitor, done = envLAS.reset()
    
    # Step counter
    i = 1
    while not done:

        actionLAS = LASAgent1.perceive_and_act(observationForLAS, rewardLAS, done)
        observationForLAS, rewardLAS, done, info = envLAS.step_LAS(actionLAS)
        print("LAS Step: {}, reward: {}".format(i, rewardLAS))
        i += 1
            
        
    envLAS.destroy()

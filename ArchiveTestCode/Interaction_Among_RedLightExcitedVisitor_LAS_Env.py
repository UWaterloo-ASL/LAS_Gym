#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  2 17:46:02 2018

@author: jack.lingheng.meng
"""

import time
import multiprocessing as mp

from RedLightExcitedVisitorAgent import RedLightExcitedVisitorAgent
from RedLightExcitedVisitor_LAS_Env import LivingArchitectureEnv
from LASAgent import LASAgent



if __name__ == '__main__':
    
    # Iinstantiate LAS-agent
    LASAgent1 = LASAgent()

    # Instantiate a red light excited visitor
    visitor = RedLightExcitedVisitorAgent("Visitor#0")
    
    # Instantiate environment object
    envLAS = LivingArchitectureEnv('127.0.0.1', 19997)
    envVisitor = LivingArchitectureEnv('127.0.0.1', 19999)
    
    observationForLAS, observationForVisitor, rewardLAS, rewardVisitor, done, [] = envLAS.reset_env_for_LAS_red_light_excited_visitor(visitor._bodayName)
    
    # Step counter
    i = 1
    last_time = time.time()
    while not done:
        try:
            action_LAS = LASAgent1.perceive_and_act(observationForLAS, rewardLAS, done)
            observationForLAS, rewardLAS, done, info = envLAS.step_LAS(action_LAS)
            #print("Step: {}, reward: {}".format(i, rewardLAS))
            
            targetPositionName, bodyName, action_visitor = visitor.perceive_and_act(observationForVisitor,rewardVisitor,done)
            observationForVisitor, reward, done, [] = envVisitor.step_red_light_excited_visitor(targetPositionName, bodyName, action_visitor)
        except KeyboardInterrupt: # Ctrl-C stop interaction
            pass
        
    envVisitor.destroy()
    envLAS.destroy()

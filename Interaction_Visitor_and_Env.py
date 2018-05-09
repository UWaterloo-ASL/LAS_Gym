#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  9 08:46:49 2018

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
    #envLAS = LivingArchitectureEnv('127.0.0.1', 19997)
    envVisitor = LivingArchitectureEnv('127.0.0.1', 19999)
    
    observationForVisitor = envVisitor._self_observe_for_red_excited_visitor(visitor._bodayName)
    done = False
    rewardVisitor = 0
    
    # Step counter
    i = 1
    last_time = time.time()
    while not done:
            
        targetPositionName, bodyName, action_visitor = visitor.perceive_and_act(observationForVisitor,rewardVisitor,done)
        observationForVisitor, reward, done, [] = envVisitor.step_red_light_excited_visitor(targetPositionName, bodyName, action_visitor)
        print("Visitor Step: {}".format(i))
        i += 1
        
    envVisitor.close_connection()

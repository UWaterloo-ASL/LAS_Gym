#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  9 08:46:49 2018

@author: jack.lingheng.meng
"""

import time

from VisitorAgent.RedLightExcitedVisitorAgent import RedLightExcitedVisitorAgent
from Environment.VisitorEnv import VisitorEnv

if __name__ == '__main__':
    
    # Instantiate a red light excited visitor
    visitor0 = RedLightExcitedVisitorAgent("Visitor#0")
    
    # Instantiate visitor environment object
    envVisitor = VisitorEnv('127.0.0.1', 19997)
    
    observationForVisitor0 = envVisitor._self_observe(visitor0._visitorName)
    done = False
    rewardVisitor = 0
    
    # Step counter
    i = 1
    last_time = time.time()
    while not done:
        
        visitorName, action = visitor0.perceive_and_act(observationForVisitor0,rewardVisitor,done)
        observationForVisitor, reward, done, [] = envVisitor.step(visitorName, action)
        print("Visitor Step: {}, Red light number: {}".format(i, visitor0.red_light_num))
        i += 1

    envVisitor.close_connection()

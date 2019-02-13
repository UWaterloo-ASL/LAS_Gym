#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""

import time

from VisitorAgent.RedLightExcitedVisitorAgent import RedLightExcitedVisitorAgent
from Environment.VisitorEnv import VisitorEnv

if __name__ == '__main__':
    
    # Instantiate a red light excited visitor
    visitor = RedLightExcitedVisitorAgent("Visitor#0")
    
    # Instantiate visitor environment object
    envVisitor = VisitorEnv('127.0.0.1', 19999)
    
    observationForVisitor = envVisitor._self_observe(visitor._bodayName)
    done = False
    rewardVisitor = 0
    
    # Step counter
    i = 1
    last_time = time.time()
    while not done:
            
        targetPositionName, bodyName, action = visitor.perceive_and_act(observationForVisitor,rewardVisitor,done)
        observationForVisitor, reward, done, [] = envVisitor.step(targetPositionName, bodyName, action)
        print("Visitor Step: {}, Red light number: {}".format(i, visitor.red_light_num))
        i += 1

    envVisitor.close_connection()

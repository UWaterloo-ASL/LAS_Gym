# -*- coding: utf-8 -*-
"""
Created on Tue Aug  7 23:09:52 2018

@author: lingheng
"""

import tensorflow as tf
import numpy as np
import time

from Environment.LASEnv import LASEnv
from LASAgent.RandomLASAgent import RandomLASAgent
from LASAgent.LASAgent_Actor_Critic import LASAgent_Actor_Critic

from Environment.VisitorEnv import VisitorEnv
from VisitorAgent.RedLightExcitedVisitorAgent import RedLightExcitedVisitorAgent


if __name__ == '__main__':
 
    # Instantiate visitor environment object
    envVisitor = VisitorEnv('127.0.0.1', 19999)
    
    # Instantiate a red light excited visitor0
    visitor = RedLightExcitedVisitorAgent("Visitor")
    # Instantiate a red light excited visitor0
    visitor0 = RedLightExcitedVisitorAgent("Visitor#0")
    # Instantiate a red light excited visitor1
    visitor1 = RedLightExcitedVisitorAgent("Visitor#1")
    # Instantiate a red light excited visitor2
    visitor2 = RedLightExcitedVisitorAgent("Visitor#2")
    # Instantiate a red light excited visitor2
    visitor3 = RedLightExcitedVisitorAgent("Visitor#3")
    
    observation_For_Visitor = envVisitor._self_observe(visitor._visitorName)
    observation_For_Visitor0 = envVisitor._self_observe(visitor0._visitorName)
    observation_For_Visitor1 = envVisitor._self_observe(visitor1._visitorName)
    observation_For_Visitor2 = envVisitor._self_observe(visitor2._visitorName)
    observation_For_Visitor3 = envVisitor._self_observe(visitor3._visitorName)
    
    # Step counter
    i = 1
    done = False
    reward_for_Visitor = 0
    reward_for_Visitor0 = 0
    reward_for_Visitor1 = 0
    reward_for_Visitor2 = 0
    reward_for_Visitor3 = 0
    while not done:
        # Visitor interacts with environment.
        visitorName, action = visitor.perceive_and_act(observation_For_Visitor,reward_for_Visitor,done)
        observation_For_Visitor, reward_for_Visitor, done, [] = envVisitor.step(visitorName, action)
        print("Visitor Step: {}, Red light number: {}".format(i, visitor.red_light_num))
        
        # Visitor0 interacts with environment.
        visitorName0, action = visitor0.perceive_and_act(observation_For_Visitor0,reward_for_Visitor0,done)
        observation_For_Visitor0, reward_for_Visitor0, done, [] = envVisitor.step(visitorName0, action)
        print("Visitor Step: {}, Red light number: {}".format(i, visitor0.red_light_num))
        
        # Visitor1 interacts with environment.
        visitorName1, action = visitor1.perceive_and_act(observation_For_Visitor1,reward_for_Visitor1,done)
        observation_For_Visitor1, reward_for_Visitor1, done, [] = envVisitor.step(visitorName1, action)
        print("Visitor Step: {}, Red light number: {}".format(i, visitor1.red_light_num))
        
        # Visitor2 interacts with environment.
        visitorName2, action = visitor2.perceive_and_act(observation_For_Visitor2,reward_for_Visitor2,done)
        observation_For_Visitor2, reward_for_Visitor2, done, [] = envVisitor.step(visitorName2, action)
        print("Visitor Step: {}, Red light number: {}".format(i, visitor2.red_light_num))
        
        # Visitor3 interacts with environment.
        visitorName3, action = visitor3.perceive_and_act(observation_For_Visitor3,reward_for_Visitor3,done)
        observation_For_Visitor3, reward_for_Visitor3, done, [] = envVisitor.step(visitorName3, action)
        print("Visitor Step: {}, Red light number: {}".format(i, visitor3.red_light_num))
        
        time.sleep(0.1)
        
        
        # reset all visitors out of the range of LAS
        move = 1
        action = [move, -6, np.random.randint(-3,0), 0]
        envVisitor.step(visitorName0, action)
        envVisitor.step(visitorName1, action)
        envVisitor.step(visitorName2, action)
        envVisitor.step(visitorName3, action)
        
        #time.sleep(0.01)
        
        i += 1
    
    envVisitor.destroy()
    

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  8 16:15:42 2018

@author: jack.lingheng.meng
"""
import numpy as np
from collections import deque

class RandomLASAgent():
    """
    Single LAS agent contorl all actuators i.e. non-distributed
    
    """
    def __init__(self, env):
        self._smas_num = 3*13   # 13 nodes, each has 3 smas
        self._light_num = 3*13  # 13 nodes, each has 3 lights
        
        self.env = env
        self.actionSpace = env.actionSpace             # gym.spaces.Box object
        self.observationSpace = env.observationSpace   # gym.spaces.Box object
        
        # ========================================================================= #
        #                 Initialize Temprary Memory                                #
        # ========================================================================= # 
        # Temporary hard memory: storing every experience
        self._memory = deque(maxlen = 10000)
        # Temporary memory: variables about last single experience
        self._firstExperience = True
        self._observationOld = []   # observation at time t
        self._observationNew = []   # observation at time t+1
        self._actionOld = []        # action at time t
        self._actionNew = []        # action at time t+1
        # Cumulative reward
        self._cumulativeReward = 0
        self._cumulativeRewardMemory = deque(maxlen = 10000)
        
    def perceive_and_act(self, observation, reward, done):
        self._observation = observation
        self._reward = reward
        self._done = done
        
        self._cumulativeReward += reward
        self._cumulativeRewardMemory.append([self._cumulativeReward])
        
        self._actionNew = self._act()
        return self._actionNew
    
    def _act(self):
#        smas = np.random.randn(self._smas_num)
#        lights_state = np.random.randint(2,size = 39)
#        #lights_state = np.ones(self._light_num)
#        lights_color = np.random.uniform(0,1,self._light_num*3)
#        #lights_color = np.array([1,0,0]*self._light_num)
#        action = np.concatenate((smas, lights_state, lights_color))
        return self.actionSpace.sample()

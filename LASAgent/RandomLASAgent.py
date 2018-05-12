#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  8 16:15:42 2018

@author: jack.lingheng.meng
"""
import numpy as np
from collections import deque

import matplotlib.pyplot as plt
def plot_cumulative_reward(cumulativeReward):
    line, = plt.plot(cumulativeReward, '-+')
    plt.ion()
    #plt.ylim([0,10])
    plt.show()
    plt.pause(0.0001)

class RandomLASAgent():
    """
    Single LAS agent contorl all actuators i.e. non-distributed
    
    """
    def __init__(self, env):
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
        # plot in real time
        if len(self._memory) %200 == 0:
            plot_cumulative_reward(self._cumulativeRewardMemory)
        
        self._actionNew = self._act()
        return self._actionNew
    
    def _act(self):
        return self.actionSpace.sample()

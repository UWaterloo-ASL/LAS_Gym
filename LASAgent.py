#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  8 16:15:42 2018

@author: jack.lingheng.meng
"""
import numpy as np

class LASAgent():
    """
    Single LAS agent contorl all actuators i.e. non-distributed
    
    """
    def __init__(self):
        self._smas_num = 3*13   # 13 nodes, each has 3 smas
        self._light_num = 3*13  # 13 nodes, each has 3 lights
        
    def perceive_and_act(self, observation, reward, done):
        self._observation = observation
        self._reward = reward
        self._done = done
        
        self._actionNew = self._act()
        return self._actionNew
    
    def _act(self):
        smas = np.random.randn(self._smas_num)
        lights_state = np.random.randint(2,size = 39)
        #lights_state = np.ones(self._light_num)
        lights_color = np.random.uniform(0,1,self._light_num*3)
        #lights_color = np.array([1,0,0]*self._light_num)
        action = np.concatenate((smas, lights_state, lights_color))
        return action
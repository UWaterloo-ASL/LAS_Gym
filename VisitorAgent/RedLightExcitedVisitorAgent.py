#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  8 14:35:52 2018

@author: jack.lingheng.meng
"""
import numpy as np

class RedLightExcitedVisitorAgent():
    """
    Visitor who is only excited about red light.
        Return:
            name: visitor name
            action: [move, x, y, z] if move = 0, don't move, else move.
    """
    def __init__(self, name):
        """
        
        """
        self._visitorName = name
        self.red_light_num = 0
        # threshold distance between last destination and current location
        self._distanceThreshold = np.sqrt((0.5**2) + (0.5**2))
        self._lastTargetPositionMaintainThreshold = 1000 # at least maintain 25 steps
        
        self._lastTargetPositionMaintainCounter = 0 # count how may step have elapsed for last target position
        
        self._lastDestination = []
        self._currLocation = []
        self._firstStep = True
    def perceive_and_act(self, observation, reward, done):
        """
        Interaction interface with environment.
        
        Parameters
        ----------
        observation: observation of environment
        reward: external reward from environment
        done: whether simulation is done or not
        
        Returns
        -------
        targetPositionName: string
            The name of visitor's target object.
        bodayName: string
            The name of visitor's body object
        action: ndarray [moveFlag, x, y, z]
            The action based on current observation. 
            moveFlag indicates whether need to move visitor's target position.
                if moveFlag == 0, then no move
                if moveFlag == 1, then move to (x, y, z)
        
        Examples
        --------
            
        """
        self._observation = observation
        self._currLocation = observation[-3:-1] # ignore z coordinate
        self._reward = reward
        self._done = done
        
        self._actionNew = self._act()
        #print("_actionNew = {}".format(self._actionNew))
        return self._visitorName, self._actionNew

    def _act(self):
        """
        Calculate visitor's action. The action depends on three conditions:
            1. red light num > 0
            2. visitor is very close to his/her previous target position
            3. the number of steps having no movement has exceeded a threshold.
        If these three conditions are satisfied, the visitor will randomly pick
        a red light, and set the red light's position as next target position.
        
        Returns
        -------
        action: ndarray [moveFlag, x, y, z]
                
        """
        red_light_positions = self._red_light_position()
        
        # only when there is red light and close to target position and maintain a maximum number to approach to target
        if self.red_light_num > 0:
            #print("Red light number: {}".format(self.red_light_num))
            random_red_light = np.random.randint(0,self.red_light_num)
            position = red_light_positions[random_red_light,:]
            move = 1
            action = np.concatenate(([move], position.flatten()))    
        else:
            move = 1
            action = [move, -6, np.random.randint(-3,0), 0]
    
        return action
    
    def _red_light_position(self):
        """
        Abstract red lights' position from current observation.
        
        If the RGB color of a light within the following thresholds, we regard
        it as a red light.
        
        Threshould for red lights:
            Red:    0.70 - 1.00
            Green:  0.00 - 0.30
            Blue:   0.00 - 0.30
        """
        light_num = int((len(self._observation) - 3) / 7) # (length - 3visitorPosition ) / (1State + 3Color + 3Position)
        #print("len(self._observation): {}".format(len(self._observation)))
        #print("Light number: {}".format(light_num))
        light_color_start_index = light_num
        light_position_start_index = light_num + light_num * 3 # start after state & color
        red_light_index = []
        for i in range(light_num):
            R = self._observation[light_color_start_index + i*3]
            G = self._observation[light_color_start_index + i*3 + 1]
            B = self._observation[light_color_start_index + i*3 + 2]
            #print("Light: {}, R={}, G={}, B={}".format(i, R,G,B))
            if 0.7<= R <=1 and 0<=G<=0.3 and 0<=B<=0.3:
                #print("Find one red light!!")
                red_light_index.append(i)
        
        self.red_light_num = len(red_light_index)
        
        red_light_positions = np.zeros([self.red_light_num,3])
        
        for i in range(self.red_light_num):
            index = red_light_index[i]
            red_light_positions[i,:] = self._observation[light_position_start_index + index*3:light_position_start_index + (index+1)*3]
            #print("Red light index:{}, Position:{}".format(index, red_light_positions[i,:]))
        return red_light_positions  
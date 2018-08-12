#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  8 14:35:52 2018

@author: jack.lingheng.meng
"""
import numpy as np

class BrightLightExcitedVisitorAgent():
    """
    Visitor who is only excited about brightest light.
        Return:
            name: visitor name
            action: [move, x, y, z] if move = 0, don't move, else move.
    """
    def __init__(self, visitor_name, light_num=24):
        """
        Parameters
        ----------
        visitor_name: string
            the name of visitor
            
        light_num: 
            the number of light
        """
        self._visitorName = visitor_name
        self.light_num = light_num
        self.bright_light_num = 0
        
    def perceive_and_act(self, observation, reward, done):
        """
        Interaction interface with environment.
        
        Parameters
        ----------
        observation: 
            observation of environment
        reward: 
            external reward from environment
        done: 
            whether simulation is done or not
        
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
        self.observation = observation
        self._currLocation = observation[-3:-1] # ignore z coordinate
        self._reward = reward
        self._done = done
        
        self._actionNew = self._act(self.observation)
        return self._visitorName, self._actionNew  
    
    def _act(self, observation):
        """
        Calculate visitor's action. If there are multiple brightest lights, randomly
        pick one as target position, while if all lights are off, put visitors
        out of the scope of Living Architecture.
        
        Parameters
        ----------
        observation:
        
        Returns
        -------
        action: ndarray [moveFlag, x, y, z]
                
        """
        bright_light_position = self._find_bright_light_position(observation)
        self.bright_light_num = len(bright_light_position)
        if self.bright_light_num == 0:
            # All lights are off
            move = 1
            action = [move, -6, np.random.randint(-3,0), 0]
        else:
            if self.bright_light_num == 1:
                random_light = 0
            else:
                # If more than one, randomly pick one
                random_light = np.random.randint(0,int(self.bright_light_num))
            position = bright_light_position[random_light]
            move = 1
            action = np.concatenate(([move], position.flatten()))
    
        return action
    
    def _find_bright_light_position(self, observation):
        """
        find the position of lights with intensity >= 0.8.
        
        Parameters
        ----------
        observation
        
        Returns
        -------
        bright_light_position: list
            if not empty: each entry is (3,) position
            if empty: 
        """
        bright_threthold = 0.8
        light_intensity, light_position, _= self._extract_observation(observation)
        bright_light_position = []
        for i_temp, intensity in enumerate(light_intensity):
            if intensity >= bright_threthold:
                bright_light_position.append(light_position[i_temp*3:(i_temp+1)*3])
        return bright_light_position
    
    def _extract_observation(self, observation):
        """
        Extract separate information from observation.
        
        Returns
        -------
        light_intensity:
            
        light_position:
            
        visitor_position:
            
        """
        light_intensity = observation[:self.light_num]
        light_position = observation[self.light_num:(self.light_num+self.light_num*3)]
        visitor_position = observation[-3:]
        return light_intensity, light_position, visitor_position
    
    # Don't use this function. Use "_find_bright_light_position(self, observation)"
    def _find_brightest_light_position(self, observation):
        """
        find the positions of brightest lights (if all lights are off, return
        empty list).
        
        Parameters
        ----------
        observation:
            
        Returns
        -------
        brightest_light_position: list
            if not empty: each entry is (3,) position
            if empty: 
            
        """
        light_intensity, light_position, _= self._extract_observation(observation)
        # Get descending order of intensity
        light_intensity_order_ascending = np.argsort(light_intensity) #ascending sort
        light_intensity_order_descending = light_intensity_order_ascending[::-1]
        # Find lights with maximum intensity
        brightest_light_position = []
        max_intensity = light_intensity[light_intensity_order_descending[0]]
        # This ensures not include any light when all lights are turned off
        if max_intensity != 0:  
            for i in light_intensity_order_descending:
                index_temp = light_intensity_order_descending[i]
                if light_intensity[index_temp] >= max_intensity:
                    brightest_light_position.append(light_position[index_temp*3:(index_temp+1)*3])
        else:
            brightest_light_position = []
        return brightest_light_position
    
    
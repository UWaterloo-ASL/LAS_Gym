#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  9 19:55:34 2018

@author: jack.lingheng.meng
"""

try:
    from .VrepRemoteApiBindings import vrep
except:
    print ('--------------------------------------------------------------')
    print ('"vrep.py" could not be imported. This means it is very likely')
    print ('that either "vrep.py" or the remoteApi library could not be found.')
    print ('Make sure both are in the same folder as this file,')
    print ('or appropriately adjust the file "vrep.py"')
    print ('--------------------------------------------------------------')
    print ('')

import gym
from gym import spaces
import numpy as np
import warnings

from .UtilitiesForEnv import get_all_object_name_and_handle

class LASEnv(gym.Env):
    def __init__(self, IP = '127.0.0.1',Port = 19997):
        """
        Instantiate LASEnv. LASEnv is the interface between LAS and Environment. Thus, LASEnv is the internal environment of LAS.
        
        Parameters
        ----------
        IP: string default = '127.0.0.1'
            IP address to connect V-REP server.
         
        Port: int default = 19997
            Port to communicate with V-REP server.
        """
        print ('Initialize LASEnv ...')
        # ========================================================================= #
        #                      Initialize V-REP related work                        #
        # ========================================================================= # 
        # Connect to V-REP server
        vrep.simxFinish(-1) # just in case, close all opened connections
        self.clientID = vrep.simxStart(IP,Port,True,True,5000,5) # Connect to V-REP
        if self.clientID!=-1:
            print ('LASEnv connected to remote V-REP API server')
        else:
            print ('LASEnv failed connecting to remote V-REP API server')
        
        # Initialize operation mode of communicated command in V-REP
        #   To get sensor data
        #     vrep.simx_opmode_buffer:   does not work, don't know why?
        #     vrep.simx_opmode_blocking: too slow
        #     vrep.simx_opmode_oneshot:  works pretty good
        self._def_op_mode = vrep.simx_opmode_blocking
        
        self._set_joint_op_mode = vrep.simx_opmode_oneshot
        self._set_light_op_mode = vrep.simx_opmode_oneshot
        self._set_visitor_op_mode = vrep.simx_opmode_oneshot

        self._get_prox_op_mode = vrep.simx_opmode_oneshot 
        self._get_light_op_mode = vrep.simx_opmode_oneshot
        
        # Start simulating in V-REP
        vrep.simxStartSimulation(self.clientID, self._def_op_mode)
        # ========================================================================= #
        #           Call utility function to get object names and handles           #
        # ========================================================================= #        
        self.proxSensorHandles, self.proxSensorNames, \
        self.lightHandles, self.lightNames, \
        self.jointHandles, self.jointNames, \
        self.visitorTargetNames, self.visitorTargetHandles, \
        self.visitorBodyNames, self.visitorBodyHandles = get_all_object_name_and_handle(self.clientID, self._def_op_mode, vrep)
        # ========================================================================= #
        #               Initialize LAS action and observation space                 #
        # ========================================================================= # 
        print("Initialize LAS action and observation space...")
        self.prox_sensor_num = len(self.proxSensorHandles)
        self.smas_num = len(self.jointHandles)
        self.lights_num = len(self.lightHandles)
        
        self.sensors_dim = self.prox_sensor_num + self.lights_num * (1+3)
        self.actuators_dim = self.smas_num + self.lights_num * (1+3) # light state & color
        # Sensor range:
        #   prox sensor: 0 or 1 
        #   light state: 0 or 1
        #   light color: [0, 1] * 3
        self.obs_max = np.array([1.]*self.sensors_dim)      
        self.obs_min = np.array([0.]*self.sensors_dim)
        # Actuator range:
        #   sma: not sure ??
        #   light state: 0 or 1
        #   light color: [0, 1] * 3
        self.act_max = np.array([1]*self.actuators_dim)
        self.act_min = np.array([0]*self.actuators_dim)
        # Agent should be informed about observationSpace and actionSpace to initialize
        # agent's observation and action dimension and value limitation.
        self.observationSpace = spaces.Box(self.obs_min, self.obs_max)
        self.actionSpace = spaces.Box(self.act_min, self.act_max)
        print("Initialization of LAS done!")
        # ========================================================================= #
        #                       Initialize other variables                          #
        # ========================================================================= #
        self.time_reward = np.array([0]*self.prox_sensor_num) # Reward for each proximity sensor
        self.reward = 0
        self.done = False
        self.info = []
        self.observation = []
    
    def step_LAS(self, action):
        """
        Take one step of interaction.
        
        Parameters
        ----------
        action: ndarray
        
        Returns
        -------
        observation: ndarray
            obervation of environment after taking an action
        reward: float
            reward after taking an action
        done: bool
            whether simulation is done or not.
        info:
            some information for debugging
        """
        # Clip action to avoid improer action command
        action = np.clip(action, self.act_min, self.act_max)
        # Split action for light and sma
        action_smas = action[:self.smas_num]
        action_lights_state = action[self.smas_num:self.smas_num+self.lights_num]
        action_lights_state = action_lights_state.astype(int)
        action_lights_color = action[self.smas_num+self.lights_num:]
        # Taking action
        vrep.simxPauseCommunication(self.clientID,True)     #temporarily halting the communication thread 
        self._set_all_joint_position(action_smas)
        self._set_all_light_state_and_color(action_lights_state,action_lights_color)
        vrep.simxPauseCommunication(self.clientID,False)    #and evaluated at the same time

        # Observe current state
        self.observation = self._self_observe()

        # Caculate reward
        self.reward = self._reward()
        
        done = False
        info = []
        return self.observation, self.reward, done, info

    def _self_observe(self):
        """
        This observe function is for LAS:
            proximity sensors
            light state
            light color
            
        Returns
        -------
        observation: ndarray (proxStates, lightStates, lightDiffsePart.flatten())
        """
        # Currently we only use proxStates, maby in the future we will need proxPosition
        proxStates, proxPosition = self._get_all_prox_data()
        lightStates, lightDiffsePart, lightSpecularPart = self._get_all_light_data()
        observation = np.concatenate((proxStates, lightStates, lightDiffsePart.flatten()))
        return observation

    def _reward(self):
        """
        Calculate reward based on observation of proximity sensor.
        The longer the proximity sensor is triggered, the higher the reward.
        """
        prox_obs = self.observation[:self.prox_sensor_num]
        self.time_reward = (prox_obs + self.time_reward)*prox_obs
        # Use adjusted sigmoid function f to map the time reward from R to [0,1]
        # f(t) = 2*sigmoid(t/ratio) -1
        # Then average all f(t) to obtain final reward
        ratio = 1000
        self.reward = np.mean(-1 + 2/(1+np.exp(-self.time_reward/ratio)))

        return self.reward

    def reset(self):
        """
        Reset environment.
        
        Returns
        -------
        observation:
        reward:
        done:
        info:
        """
        #vrep.simxStopSimulation(self.clientID, self._def_op_mode)
        vrep.simxStartSimulation(self.clientID, self._def_op_mode)
        
        self.observation = self._self_observe()
        self.reward = self._reward()
        
        done = False
        info =[]
        return self.observation, self.reward, done, info
        
    def destroy(self):
        """
        Stop simulation on server and release connection.
        """
        vrep.simxStopSimulation(self.clientID, self._def_op_mode)
        vrep.simxFinish(self.clientID)
        
    def close_connection(self):
        """
        Release connnection, but not stop simulation on server.
        """
        vrep.simxFinish(self.clientID)
    
    def _set_all_joint_position(self, targetPosition):
        """
        Set all joint position.
        
        Parameters
        ----------
        targetPosition: ndarray
        target position of each joint
        """
        jointNum = len(self.jointHandles)
        for i in range(jointNum):
            vrep.simxSetJointTargetPosition(self.clientID, self.jointHandles[i], targetPosition[i], self._set_joint_op_mode)

    def _set_all_light_state_and_color(self, targetState, targetColor):
        
        """
        Set all light sate and color
        
        Parameters
        ----------
        targetState: ndarray
            target state of each light
        targetColor ndarray
            target color of each light
        """
        lightNum = self.lights_num
        if len(targetState) != lightNum:
            print("len(targetState) != lightNum")
        
        # Inner function: remote function call to set light state
        def _set_light_state_and_color(clientID, name, handle, targetState, targetColor, opMode):

            emptyBuff = bytearray()
            res,retInts,retFloats,retStrings,retBuffer = vrep.simxCallScriptFunction(clientID,
                                                                           name,
                                                                           vrep.sim_scripttype_childscript,
                                                                           'setLightStateAndColor',
                                                                           [handle, targetState],targetColor,[],emptyBuff,
                                                                           opMode)
            if res != vrep.simx_return_ok:
                warnings.warn("Remote function call: setLightStateAndColor fail in Class AnyLight.")
        # inner function end
        for i in range(lightNum):
           _set_light_state_and_color(self.clientID, str(self.lightNames[i]), self.lightHandles[i], targetState[i], targetColor[i*3:(i+1)*3], self._set_light_op_mode)

    def _get_all_prox_data(self):
        """
        Get all proximity sensory data
        """
        proxSensorNum = len(self.proxSensorHandles)
        proxStates = np.zeros(proxSensorNum)
        proxPosition = np.zeros([proxSensorNum, 3])
        for i in range(proxSensorNum):
            code, proxStates[i], proxPosition[i,:], handle, snv = vrep.simxReadProximitySensor(self.clientID, self.proxSensorHandles[i], self._get_prox_op_mode)
        return proxStates, proxPosition

    def _get_all_light_data(self):
        """
        Get all light data.
        
        Returns
        -------
        lightStates: ndarray
            light states
        lightDiffsePart: ndarray
            light color
        lightSpecularPart: ndarray
            also light color, currently not used
        """
        lightNum = len(self.lightHandles)
        #print("lightNum:{}".format(lightNum))
        lightStates = np.zeros(lightNum)
        lightDiffsePart = np.zeros([lightNum,3])
        lightSpecularPart = np.zeros([lightNum,3])
        
        # inner function to get light state and color
        def _get_light_state_and_color(clientID, name , handle, op_mode):
            emptyBuff = bytearray()
            res,retInts,retFloats,retStrings,retBuffer=vrep.simxCallScriptFunction(clientID,
                                                                                   name,
                                                                                   vrep.sim_scripttype_childscript,
                                                                                   'getLightStateAndColor',
                                                                                   [handle],[],[],emptyBuff,
                                                                                   op_mode)
            if res==vrep.simx_return_ok:
                #print ('getLightStateAndColor works! ',retStrings[0]) # display the reply from V-REP (in this case, just a string)
                lightState = retInts[0]
                diffusePart = [retFloats[0],retFloats[1],retFloats[2]]
                specularPart = retFloats[3],retFloats[4],retFloats[5]
                return lightState, diffusePart, specularPart
            else:
                warnings.warn("Remote function call: getLightStateAndColor fail in Class AnyLight.")
                return -1, [0,0,0], [0,0,0]
        # inner function end
        
        for i in range(lightNum):
           lightStates[i], lightDiffsePart[i,:], lightSpecularPart[i,:] = _get_light_state_and_color(self.clientID, str(self.lightNames[i]), self.lightHandles[i], self._get_light_op_mode)
        
        return lightStates, lightDiffsePart, lightSpecularPart    
    

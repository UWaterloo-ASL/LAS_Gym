#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on July 9 19:55:34 2018

@author: daiwei.lin
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

import time

from collections import deque
import re

from .UtilitiesForEnv import get_all_object_name_and_handle
from .Behaviour import Behaviour
from IPython.core.debugger import Tracer

class LASROMEnv(gym.Env):
    def __init__(self, IP = '127.0.0.1', Port = 19997,
                 reward_function_type = 'actuator_intensity'):
        """
        Instantiate LASEnv. LASEnv is the interface between LAS and Environment. Thus, LASEnv is the internal environment of LAS.
        
        Parameters
        ----------
        IP: string default = '127.0.0.1'
            IP address to connect V-REP server.
         
        Port: int default = 19997
            Port to communicate with V-REP server.
        
        reward_function_type: str default = 'red_light_dense'
            Choose reward function type.
            Options:
                1. 'red_light_dense'
                2. 'red_light_sparse'
        """
        print ('Initialize LASROMEnv ...')
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
        
#        self._get_prox_op_mode = vrep.simx_opmode_streaming 
#        self._get_light_op_mode = vrep.simx_opmode_streaming
        
        # Start simulating in V-REP
        vrep.simxStartSimulation(self.clientID, self._def_op_mode)
        # ========================================================================= #
        #           Call utility function to get object names and handles           #
        # ========================================================================= #        
        self.proxSensorHandles, self.proxSensorNames, \
        self.lightHandles, self.lightNames, \
        self.jointHandles, self.jointNames, \
        self.visitorTargetNames, self.visitorTargetHandles, \
        self.visitorBodyNames, self.visitorBodyHandles, \
        self.excitorHandles, self.excitorNames = get_all_object_name_and_handle(self.clientID, self._def_op_mode, vrep)
        # ========================================================================= #
        #               Initialize LAS action and observation space                 #
        # ========================================================================= # 
        print("Initialize LAS action and observation space...")
        self.prox_sensor_num = len(self.proxSensorHandles)
        self.smas_num = len(self.jointHandles)
        self.lights_num = len(self.lightHandles)
        # Sensor range:
        #   prox sensor: 0 or 1 
        #   light color: [0, 1] * 3
        self.sensors_dim = self.prox_sensor_num  # + self.lights_num * (3)
        self.obs_max = np.array([1.]*self.sensors_dim)      
        self.obs_min = np.array([0.]*self.sensors_dim)
        # Actuator range:
        #   sma: not sure ??
        #   light color: [0, 1] * 3
        # self.actuators_dim = self.smas_num + self.lights_num * (3) # light state & color
        self.actuators_dim = 6
        # Set action range to [-1, 1], when send command to V-REP useing 
        # (action + 1) / 2 to transform action to range [0,1]
        self.act_max = np.array([1]*self.actuators_dim)
        self.act_min = np.array([-1]*self.actuators_dim)
        # Agent should be informed about observation_space and action_space to initialize
        # agent's observation and action dimension and value limitation.
        self.observation_space = spaces.Box(self.obs_min, self.obs_max, dtype = np.float32)
        self.action_space = spaces.Box(self.act_min, self.act_max, dtype = np.float32)
        print("Initialization of LAS done!")
        # ========================================================================= #
        #                       Initialize other variables                          #
        # ========================================================================= #
        self.time_reward = np.array([0]*self.prox_sensor_num) # Reward for each proximity sensor
        self.action_history = deque(maxlen=500)
        self.group_id, self.group_num = self._create_group()
        self._create_group()
        self.reward = 0
        self.done = False
        self.info = []
        self.observation = []
        self.prev_excitor_size = 1.0
        # ========================================================================= #
        #                    Initialize Reward Function Type                        #
        # ========================================================================= #        
        self.reward_function_type = reward_function_type

        self.behaviour = Behaviour(self.clientID, self._get_light_op_mode, self._set_joint_op_mode,self.jointHandles, self.jointNames, self.lightHandles, self.lightNames)

    def step(self, action):
        """
        Take one step of interaction.

        Parameters
        ----------
        action:  x,y,z,radius,threshold,intensity
        range:
        x [-7.5 7.5]
        y [-5, 5]
        z   [0 2]
        radius = [0 2]
        threshold = [0,5]
        intensity = [0,1]

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

        # Action is performed through Behaviour
        x = action[0]*7.5
        y = action[1]*5.0
        z = action[2] + 1.0
        radius = action[3] + 1.0
        threshold = action[4]*2.5 + 2.5
        intensity = action[5]*0.5 + 0.5

        self.behaviour.act([x,y,z,radius,threshold,max(intensity,radius)]) # taking max() here is to make it consistant with the Behaviour class

        # Visualize excitor
        # self._set_excitor_pos_and_size("Excitor", threshold, [x,y,z])

        # move the visitor based on observations one step ago
        # self._move_single_to_excitor_location(self.observation, isBody=True)
        self._move_to_excitor_location(self.observation)

        time.sleep(0.01)
        # Observe current state
        try:
            self.observation = self._self_observe()
        except ValueError:
            self._nanObervationFlag = 1
            print("Observation has NAN value.")
       # Caculate reward
       # self.reward = self._reward(self.observation)

        # Therefore, it's only used for tunning hyper-parameters of LASAgent
        if self.reward_function_type == 'ir':
            self.reward = self._reward_ir(self.observation)
        elif self.reward_function_type == 'actuator_intensity':
            self.reward = self._reward_actuator_intensity(self.observation)
        else:
            raise ValueError('No reward function: {}'.format(self.reward_function_type))


        done = False
        info = []
        observation = self.observation[0:self.prox_sensor_num] # return only proximity sensor
        return observation, self.reward, done, info

    def _self_observe(self):
        """
        This observe function is for LAS:
            proximity sensors
            light color
            
        Returns
        -------
        observation: ndarray (proxStates, lightStates, lightDiffsePart.flatten())
        """
        # Currently we only use proxStates, maybe in the future we will need proxPosition

        proxStates, proxPosition = self._get_all_prox_data()
        lightStates, lightDiffsePart, lightSpecularPart = self._get_all_light_data()

        observation = np.concatenate((proxStates, lightDiffsePart.flatten()))
        return observation

    def _reward(self, observation):
        """
        Calculate reward based on observation of proximity sensor.

        All sensors have accumulated time rewards. The longer it has been triggered, the greater
        the reward will be. The reward function also uses memories of actions. For sensors with
        zero accumulated rewards, if it detects a new trigger but an action belonging to its own
        group CANNOT be found in the history, then this signal is ignored.

        Use adjusted sigmoid function to calculate the individual reward r(t)_i based on accumulated
        time reward (t)

        r(t)_i = 2*sigmoid(t/ratio) -1

        Final reward = avg(all sensors reward r(t)_i)
        """
        individual_action_summary = np.array([0]*self.lights_num)
        if len(self.action_history) > 0:
            for step_action in self.action_history:
                individual_action_summary = individual_action_summary | step_action

        group_action_summary = np.array([0]*self.group_num)
        for j in range(0, self.lights_num):
            # Notice here the group id (node number) starts from 1
            group_action_summary[self.group_id[j]-1] = group_action_summary[self.group_id[j]-1] | \
                                                       individual_action_summary[self.group_id[j]-1]

        prox_obs = observation[:self.prox_sensor_num]
        is_newly_triggered = prox_obs - self.time_reward > 0

        for i in range(0, self.lights_num):
            if prox_obs[i] == 1:
                obs = 1
                if is_newly_triggered[i] and group_action_summary[self.group_id[j]-1] == 0:
                    obs = 0
                self.time_reward[i] = self.time_reward[i] + obs
            else:
                self.time_reward[i] = 0

        # prox_obs = observation[:self.prox_sensor_num]
        # self.time_reward = (prox_obs + self.time_reward) * prox_obs

        ratio = 1000
        self.reward = np.mean(-1 + 2/(1+np.exp(-self.time_reward/ratio)))

        return self.reward

    def _create_group(self):
        """
        Create a list that maps actuator/sensor to its corresponding node number
        Group number starts from 1
        Return
        ------
        group_id: a group id list. e.x, a list = [1,4,3,2,1,2] means the first sensor belongs to node#1,
        the second belongs to node#4, etc.

        group_num: the total number of groups
        """
        group_id = np.array([0] * self.prox_sensor_num)
        for i in range(0, self.prox_sensor_num):
            node_num = re.search(r'\d+', self.proxSensorNames[i]).group()
            group_id[i] = int(node_num)

        group_num = group_id.max()

        return group_id, group_num

    def _reward_ir(self, observation):
        """
        Reward function based on proximity sensor reading
        """
        length = self.prox_sensor_num
        reward = 0
        for i in range(length):
            reward += observation[i]

        return reward

    def _reward_actuator_intensity(self, observation):
        """
        This reward function is used in non-interactive model and only for testing and tuning hyper-parameters of LASAgent.
        The reward is sum of the intensity of light.
        Maybe we will add intensity of SMA into calculating the rewards later.

        """
        reward = 0.0
        light_color_start_index = self.prox_sensor_num
        red_light_index = []
        for i in range(self.lights_num):
            R = observation[light_color_start_index + i * 3]
            G = observation[light_color_start_index + i * 3 + 1]
            B = observation[light_color_start_index + i * 3 + 2]
            reward += R

        return reward

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
        # self.reward = self._reward(self.observation)
        
        observation = self.observation[:self.prox_sensor_num]

        self._set_excitor_pos_and_size("Excitor", 1, [0, 0, 0])

        return observation#, self.reward, done
        
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
    # ========================================================================= #
    #                                                                           # 
    #                           Set Sma and Light Color                         #
    #                                                                           # 
    # ========================================================================= #     
    def _set_all_joint_position(self, targetPosition):
        """
        Set all joint position.
        
        Parameters
        ----------
        targetPosition: ndarray
        target position of each joint
        """
        jointNum = self.smas_num
        if jointNum != len(targetPosition):
            raise ValueError('Joint targetPosition is error!!!')

        for i in range(jointNum):
            res = vrep.simxSetJointTargetPosition(self.clientID, self.jointHandles[i], targetPosition[i], self._set_joint_op_mode)

    def _set_all_light_state_and_color(self, targetColor):
        
        """
        Set all light sate and color
        
        Parameters
        ----------
        targetColor ndarray
            target color of each light
        """
        lightNum = self.lights_num
        if len(targetColor) != (lightNum*3):
            raise ValueError('len(targetColor) != lightNum*3')
        
        # Inner function: remote function call to set light state
        targetState = np.ones(self.lights_num, dtype = int)
        def _set_light_state_and_color(clientID, name, handle, targetState, targetColor, opMode):

            emptyBuff = bytearray()
            res,retInts,retFloats,retStrings,retBuffer = vrep.simxCallScriptFunction(clientID,
                                                                           name,
                                                                           vrep.sim_scripttype_childscript,
                                                                           'setLightStateAndColor',
                                                                           [handle, targetState],targetColor,[],emptyBuff,
                                                                           opMode)
            return res
        # inner function end
        for i in range(lightNum):
           res = _set_light_state_and_color(self.clientID, str(self.lightNames[i]), self.lightHandles[i], targetState[i], targetColor[i*3:(i+1)*3], self._set_light_op_mode)
    # ========================================================================= #
    #                                                                           #         
    #                    Get Proximity Sensor and Light Data                    #
    #                                                                           #
    # ========================================================================= # 
    def _get_all_prox_data(self):
        """
        Get all proximity sensory data
        """
        proxSensorNum = len(self.proxSensorHandles)
        proxStates = np.zeros(proxSensorNum)
        proxPosition = np.zeros([proxSensorNum, 3])
        for i in range(proxSensorNum):
            code, proxStates[i], proxPosition[i,:], handle, snv = vrep.simxReadProximitySensor(self.clientID, self.proxSensorHandles[i], self._get_prox_op_mode)
            if np.sum(np.isnan(proxStates[i])) != 0:
                raise ValueError("Find nan value in proximity sensor data!")
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
                return 0, [0,0,0], [0,0,0]
        # inner function end
        
        for i in range(lightNum):
            lightStates[i], lightDiffsePart[i,:], lightSpecularPart[i,:] = _get_light_state_and_color(self.clientID, str(self.lightNames[i]), self.lightHandles[i], self._get_light_op_mode)            
            if np.sum(np.isnan(lightDiffsePart[i,:])) != 0:
                raise ValueError("Find nan value in light color!")
        return lightStates, lightDiffsePart, lightSpecularPart    
    

    #======================================================#
    #                                                      #
    #                 Visitor control                      #
    #                                                      #
    #======================================================#

    def _set_single_visitor_position(self, visitorBodyName, position):
        visitorIndex = np.where(self.visitorBodyNames == visitorBodyName)
        if len(visitorIndex[0]) == 0:
            print("Not found visitor body: {}".format(visitorBodyName))
        else:
            vrep.simxSetObjectPosition(self.clientID, self.visitorBodyHandles[visitorIndex], -1, position, self._set_visitor_op_mode)

    def _set_single_target_position(self, visitorTargetName, position):
        visitorIndex = np.where(self.visitorTargetNames == visitorTargetName)
        if len(visitorIndex[0]) == 0:
            print("Not found visitor target: {}".format(visitorTargetName))
        else:
            vrep.simxSetObjectPosition(self.clientID, self.visitorTargetHandles[visitorIndex], -1, position, self._set_visitor_op_mode)

    def _move_to_excitor_location(self, observation):
        """
        find lights with maximum intensities and move the visitors to that location

        """
        light_color_start_index = self.prox_sensor_num

        light_intensity = np.zeros(self.lights_num)
        for i in range(self.lights_num):
            light_intensity[i] = observation[light_color_start_index + i * 3]

        # sort intensity by ascending order
        light_inten_order = np.argsort(light_intensity)

        num_visitors = len(self.visitorBodyHandles)
        visitor_idx = 0
        for light_idx in reversed(light_inten_order):
            if light_intensity[light_idx] > 0 and visitor_idx < num_visitors:
                position = self.behaviour.single_light_position(light_idx)
                position[2] = 0.9154
                self._set_single_visitor_position("Body_Visitor#"+str(visitor_idx), position)
                visitor_idx += 1
            else:
                break

        # Put the rest of the visitors at origin
        if visitor_idx < num_visitors:
            for i in range(visitor_idx,num_visitors):
                self._set_single_visitor_position("Body_Visitor#" + str(i), [0,0,0.9154])

    def _move_single_to_excitor_location(self, observation, isBody=True):
        """
        find the light with maximum intensity and move Visitor#1 to that location
        Testing target movement

        """
        light_color_start_index = self.prox_sensor_num

        target_light_index = -1
        max_intensity = 0

        for i in range(self.lights_num):
            light_intensity = observation[light_color_start_index + i * 3]
            if light_intensity > max_intensity:
                target_light_index = i

        if target_light_index >= 0:
            position = self.behaviour.single_light_position(target_light_index)
            position[2] = 0.9154 # fixed height of human model
        else:
            position = [0,0,0.9154] # if no light is turned on, then visitor is placed at origin.

        if isBody:
            # visitor body
            self._set_single_visitor_position("Body_Visitor#1", position)
        else:
            # visitor target
            position[2] = 0
            self._set_single_target_position("Target_Visitor#1",position)

    #==================================#
    #                                  #
    #      Excitor in the env          #
    #                                  #
    #==================================#

    # This function is for visualization of the excitor

    def _set_excitor_pos_and_size(self, excitor_name, excitor_size, excitor_position):
        excitorIndex = np.where(self.excitorNames == excitor_name)
        if len(excitorIndex[0]) == 0:
            print("Not found excitor: {}".format(excitor_name))
        else:
            ex_handle = self.excitorHandles[excitorIndex]

            # Set excitor position
            vrep.simxSetObjectPosition(self.clientID, ex_handle, -1, excitor_position, self._set_visitor_op_mode)

            # Set excitor size
            ratio = excitor_size/self.prev_excitor_size
            self.prev_excitor_size = excitor_size
            emptyBuff = bytearray()
            res, retInts, retFloats, retStrings, retBuffer = vrep.simxCallScriptFunction(self.clientID,
                                                                                         excitor_name,
                                                                                         vrep.sim_scripttype_childscript,
                                                                                         'setExcitorSize',
                                                                                         [ex_handle],
                                                                                         [ratio], [],
                                                                                         emptyBuff,
                                                                                         vrep.simx_opmode_oneshot)
            if res != vrep.simx_return_ok:
                warnings.warn("Remote function call: setExcitorSize failed.")

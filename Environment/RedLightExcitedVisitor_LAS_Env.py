#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  2 17:46:02 2018

@author: jack.lingheng.meng
"""

try:
    from .VrepRemoteApiBindings import vrep
except:
    print ('--------------------------------------------------------------')
    print ('"vrep.py" could not be imported. This means very probably that')
    print ('either "vrep.py" or the remoteApi library could not be found.')
    print ('Make sure both are in the same folder as this file,')
    print ('or appropriately adjust the file "vrep.py"')
    print ('--------------------------------------------------------------')
    print ('')
import gym
from gym import spaces
import sys
import ctypes
import time
import numpy as np
import warnings
import multiprocessing as mp

class LivingArchitectureEnv(gym.Env):
    """
    Currently this environment class is pretty versatile i.e. both visitor and
    LAS interact with VREP through this class. Therefore, many customized functions
    and interfaces only for visitor mixe with customized functions and interfaces
    for LAS. 
    
    This might not a good idea, because this could bottleneck communication if 
    both visitor and LAS interact with VREP through the same port.
    
    Therefore, I believe we should separate environment class for visitor and 
    LAS. But we still can keep some basic common functions in a parant environment
    class to make our code easier to understand and reuse.
    """
    def __init__(self, IP = '127.0.0.1', Port = 19997):
        """
        Initialize environment
        
        Parameters
        ----------
        IP: string default = '127.0.0.1'
        
        Port: int  default = 19997
        
        """
        print ('Program started')
        # connect to V-REP server
        vrep.simxFinish(-1) # just in case, close all opened connections
        self.clientID = vrep.simxStart(IP,Port,True,True,5000,5) # Connect to V-REP
        if self.clientID!=-1:
            print ('Connected to remote API server')
        else:
            print ('Failed connecting to remote API server')
        # start simulate
        self._def_op_mode = vrep.simx_opmode_blocking
        self._set_joint_op_mode = vrep.simx_opmode_oneshot
        self._set_light_op_mode = vrep.simx_opmode_oneshot
        self._set_visitor_op_mode = vrep.simx_opmode_oneshot
        
        # To get sensor data
        #   vrep.simx_opmode_buffer: does not work, don't know why?
        #   vrep.simx_opmode_blocking: too slow
        #   vrep.simx_opmode_oneshot: works pretty good
        self._get_prox_op_mode = vrep.simx_opmode_oneshot 
        self._get_light_op_mode = vrep.simx_opmode_oneshot
        
        
        
        vrep.simxStartSimulation(self.clientID, self._def_op_mode)
        
        # get object names and handles
        self._get_object_name_and_handle()
        
        # initialize action and observation space
        print("Initialize LAS action and observation space...")
        self.prox_sensor_num = len(self.proxSensorHandles)
        self.smas_num = len(self.jointHandles)
        self.lights_num = len(self.lightHandles)
        self.sensors_dim = self.prox_sensor_num + self.lights_num * (1+3)
        self.actuators_dim = self.smas_num + self.lights_num * (1+3) # light state & color
        
        self.act_max = np.array([np.inf]*self.actuators_dim)
        self.act_min = - np.array([np.inf]*self.actuators_dim)
        self.obs_max = np.array([1.]*self.sensors_dim)
        self.obs_min = - np.array([1.]*self.sensors_dim)
        
        self.observation_space = spaces.Box(self.obs_min, self.obs_max)
        self.action_space = spaces.Box(self.act_min, self.act_max)
        print("Initialization of LAS done!")
        
        # initialize Visitor action and observation space
        print("Initialize Visitor action and observation space...")
        self.visitor_num = len(self.visitorHandles)
        self.visitor_action_dim = self.visitor_num * 2 # visitor's position (x,y,0)
        self.visitor_action_max = np.array([7,9]*self.visitor_num) # later we should find a way to automatic get this limit
        self.visitor_action_min = np.array([-7,-9]*self.visitor_num)
        self.visitor_action_space = spaces.Box(self.visitor_action_min, self.visitor_action_max)
        
        # initialize Single Visitor action and observation space
        print("Initialize Visitor action and observation space...")
        self.single_visitor_action_dim = self.visitor_num * 2 # visitor's position (x,y,0)
        self.single_visitor_action_max = np.array([7,9]) # later we should find a way to automatic get this limit
        self.single_visitor_action_min = np.array([-7,-9])
        self.single_visitor_action_space = spaces.Box(self.single_visitor_action_min, self.single_visitor_action_max)
        
        print("Initialization of visitor done!")
        
        self.reward = 0
        
    
    def _get_object_name_and_handle(self):
        """
        # objectType:
            #       joint: sim_object_joint_type
            #       proximity sensor: sim_object_proximitysensor_type
            #       light: sim_object_light_type
        """
        dataType = 0    # 0: retrieves the object names (in stringData.)
        print("Get objects' names and handles ...")
        proxSensorIndex = []
        lightIndex = []
        jointIndex = []
        visitorIndex = []
        visitorBodyIndex = []
        # proximity sensor
        rc = vrep.simx_return_initialize_error_flag
        while rc != vrep.simx_return_ok:
            rc, proxSensorHandles, intData, floatData, proxSensorNames = vrep.simxGetObjectGroupData(self.clientID,vrep.sim_object_proximitysensor_type, dataType, self._def_op_mode)
            if rc==vrep.simx_return_ok:
                print ('Get Prox Sensor Success!!!!!') # display the reply from V-REP (in this case, just a string)
                for i, name in enumerate(proxSensorNames):
                    if "_node#" in name:
                        print("Proximity Sensor: {}, and handle: {}".format(name, proxSensorHandles[i]))
                        proxSensorIndex.append(i)
                break
            else:
                print ('Fail to get proximity sensors!!!')
        # light 
        rc = vrep.simx_return_initialize_error_flag
        while rc != vrep.simx_return_ok:
            rc, lightHandles, intData, floatData, lightNames = vrep.simxGetObjectGroupData(self.clientID,vrep.sim_object_light_type, dataType, self._def_op_mode)
            if rc==vrep.simx_return_ok:
                print ('Get Lihgt Success!!!!!') # display the reply from V-REP (in this case, just a string)
                for i, name in enumerate(lightNames):
                    if "_node#" in name:
                        print("Light: {}, and handle: {}".format(name, lightHandles[i]))
                        lightIndex.append(i)
                break
            else:
                print ('Fail to get lights!!!')
        # joint
        rc = vrep.simx_return_initialize_error_flag
        while rc != vrep.simx_return_ok:
            rc, jointHandles, intData, floatData, jointNames = vrep.simxGetObjectGroupData(self.clientID,vrep.sim_object_joint_type, dataType, self._def_op_mode)
            if rc==vrep.simx_return_ok:
                print ('Get Joint Success!!!!!') # display the reply from V-REP (in this case, just a string)
                for i, name in enumerate(jointNames):
                    if "_node#" in name:
                        print("Joint: {}, and handle: {}".format(name, jointHandles[i]))
                        jointIndex.append(i)
                break
            else:
                print ('Fail to get joints!!!')
        
        # visitor targetPosition
        rc = vrep.simx_return_initialize_error_flag
        while rc != vrep.simx_return_ok:
            rc, visitorHandles, intData, floatData, visitorNames = vrep.simxGetObjectGroupData(self.clientID,vrep.sim_object_dummy_type, dataType, self._def_op_mode)
            if rc==vrep.simx_return_ok:
                print ('Get Visitor Success!!!!!') # display the reply from V-REP (in this case, just a string)
                for i, name in enumerate(visitorNames):
                    if "TargetPosition_Visitor#" in name:
                        print("Visitor: {}, and handle: {}".format(name, visitorHandles[i]))
                        visitorIndex.append(i)
                break
            else:
                print ('Fail to get visitors!!!')
        # visitor body
        rc = vrep.simx_return_initialize_error_flag
        while rc != vrep.simx_return_ok:
            rc, visitorBodyHandles, intData, floatData, visitorBodyNames = vrep.simxGetObjectGroupData(self.clientID,vrep.sim_object_shape_type, dataType, self._def_op_mode)
            if rc==vrep.simx_return_ok:
                print ('Get Visitor Body Success!!!!!') # display the reply from V-REP (in this case, just a string)
                for i, name in enumerate(visitorBodyNames):
                    if "Body_Visitor#" in name:
                        print("Visitor body: {}, and handle: {}".format(name, visitorBodyHandles[i]))
                        visitorBodyIndex.append(i)
                break
            else:
                print ('Fail to get visitors body!!!')
        
        proxSensorHandles = np.array(proxSensorHandles)
        proxSensorNames = np.array(proxSensorNames)
        lightHandles = np.array(lightHandles)
        lightNames = np.array(lightNames)
        jointHandles = np.array(jointHandles)
        jointNames = np.array(jointNames)
        visitorHandles = np.array(visitorHandles)
        visitorNames = np.array(visitorNames)
        visitorBodyHandles = np.array(visitorBodyHandles)
        visitorBodyNames = np.array(visitorBodyNames)
        # All objects handels and names
        self.proxSensorHandles = proxSensorHandles[proxSensorIndex]
        self.proxSensorNames = proxSensorNames[proxSensorIndex]
        self.lightHandles = lightHandles[lightIndex]
        self.lightNames = lightNames[lightIndex]
        self.jointHandles = jointHandles[jointIndex]
        self.jointNames = jointNames[jointIndex]
        self.visitorNames = visitorNames[visitorIndex]
        self.visitorHandles = visitorHandles[visitorIndex]
        self.visitorBodyNames = visitorBodyNames[visitorBodyIndex]
        self.visitorBodyHandles = visitorBodyHandles[visitorBodyIndex]

    def step_LAS(self, action):
        """
        Take one step of action
        Input: action
        Output: observation, reward, done, info
        """
        #
        action = np.clip(action, self.act_min, self.act_max)
        # split action for light and sma
        action_smas = action[:self.smas_num]
        action_lights_state = action[self.smas_num:self.smas_num+self.lights_num]
        action_lights_state = action_lights_state.astype(int)
        action_lights_color = action[self.smas_num+self.lights_num:]
        # taking action
        #start = time.time()
        vrep.simxPauseCommunication(self.clientID,True)     #temporarily halting the communication thread 
        self._set_all_joint_position(action_smas)
        self._set_all_light_state(action_lights_state,action_lights_color)
        vrep.simxPauseCommunication(self.clientID,False)    #and evaluated at the same time
        #print("Action running time: {}".format(time.time()-start))
        
        # observe
        #start = time.time()
        self._self_observe()
        #print("Observation running time: {}".format(time.time()-start))
        # caculate reward
        self._reward()
        
        done = False
        
        return self.observation, self.reward, done, []
    
    def step_visitor(self, position):
        """
        This interface is for change visitor's position.
        Input: position
        Output: observation, reward, done, info
        """
        #
        position = np.clip(position,self.visitor_action_min, self.visitor_action_max)
        vrep.simxPauseCommunication(self.clientID,True)
        self._set_all_visitor_position(position)
        vrep.simxPauseCommunication(self.clientID,False)
        
        self._self_observe()
        self._reward_visitor()
        done = False
        return self.observation, self.reward_visitor, done, [] 

    def step_single_visitor(self, name, position):
        """
        This interface is for change visitor's position.
        Input: position
        Output: observation, reward, done, info
        """
        #
        position = np.clip(position,self.single_visitor_action_min, self.single_visitor_action_max)
        #vrep.simxPauseCommunication(self.clientID,True)
        self._set_single_visitor_position(name, position)
        #vrep.simxPauseCommunication(self.clientID,False)
        
        self._self_observe()
        self._reward_visitor()
        done = False
        return self.observation, self.reward_visitor, done, []   
    
    def step_red_light_excited_visitor(self, targetPositionName, bodyName, action):
        """
        A specific interface for red excited visitor:
            return observation:
                light state: observation[:lightNum]
                light color: observation[lightNum:lightNum * 4]
                light position: observation[lightNum * 4:lightNum * 5]
                visitor position: observation[lightNum*5:]
        """
        move = action[0]
        position = action[1:3] # we can leave z coordinate
        #print("Set position:{}".format(position))
        position = np.clip(position,self.single_visitor_action_min, self.single_visitor_action_max)
        # if move == 1, move; otherwise don't move.
        if move == 1:
            #vrep.simxPauseCommunication(self.clientID,True)
            #print("Set Position in Vrep: {}".format(position))
            self._set_single_visitor_position(targetPositionName, position)
            #vrep.simxPauseCommunication(self.clientID,False)
        
        observation = self._self_observe_for_red_excited_visitor(bodyName)
        #print("len(observation):{}".format(len(observation)))
        reward = 0
        done = False
        return observation, reward, done, []
    
    def _set_single_visitor_position(self, targetPositionName, position):
        visitorIndex = np.where(self.visitorNames == targetPositionName)
        if len(visitorIndex[0]) == 0:
            print("Not found visitor: {}".format(targetPositionName))
        else:
            vrep.simxSetObjectPosition(self.clientID, self.visitorHandles[visitorIndex], -1, [position[0],position[1],0], self._set_visitor_op_mode)
    def _get_single_visitor_body_position(self, bodyName):
        """
        Give bodyName, return bodyPosition
        """
        bodyPosition = np.zeros(3)
        visitorBodyIndex = np.where(self.visitorBodyNames == bodyName)
        if len(visitorBodyIndex[0]) == 0:
            print("Not found visitor: {}".format(bodyName))
        else:
            res, bodyPosition = vrep.simxGetObjectPosition(self.clientID, self.visitorBodyHandles[visitorBodyIndex], -1, self._get_light_op_mode)
        #print("Visitor position: {}".format(position))
        return np.array(bodyPosition)
    
    def _set_all_visitor_position(self, position):
        visitorNum = len(self.visitorHandles)
        for i in range(visitorNum):
            vrep.simxSetObjectPosition(self.clientID, self.visitorHandles[i], -1, [position[i*2],position[i*2+1],0], self._set_visitor_op_mode)
    
    def _set_all_joint_position(self, targetPosition):
        jointNum = len(self.jointHandles)
        for i in range(jointNum):
            vrep.simxSetJointTargetPosition(self.clientID, self.jointHandles[i], targetPosition[i], self._set_joint_op_mode)
        
    def _set_all_light_state(self, targetState, targetColor):
        lightNum = len(self.lightHandles)
        if len(targetState) != lightNum:
            print("len(targetState) != lightNum")
        
        # inner function: remote function call to set light state
        def _set_light_state(clientID, name, handle, targetState, targetColor, opMode):

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
           _set_light_state(self.clientID, str(self.lightNames[i]), self.lightHandles[i], targetState[i], targetColor[i*3:(i+1)*3], self._set_light_op_mode)

    def _reward(self):
        """ calculate reward based on observation of prximity sensor"""
        self.reward = np.mean(self.observation[:self.prox_sensor_num])
        return self.reward
    
    def _reward_visitor(self):
        """
        Calculate reward for visitor
        """
        self.reward_visitor = 0
        return self.reward_visitor
    
    def _self_observe(self):
        """
        This observe function is for LAS:
            proximity sensors
            light state
            light color
        """
        proxStates, proxPosition = self._get_all_prox_data()
        lightStates, lightDiffsePart, lightSpecularPart = self._get_all_light_data()
        self.observation = np.concatenate((proxStates, lightStates, lightDiffsePart.flatten()))
        return self.observation
    
    def _self_observe_for_red_excited_visitor(self,bodyName):
        """
        This obervave function is for visitors:
            light state: observation[:lightNum]
            light color: observation[lightNum:lightNum * 4]
            light position: observation[lightNum * 4:lightNum * 5]
            visitor position: observation[lightNum*5:]
        """
        lightStates, lightDiffsePart, lightSpecularPart = self._get_all_light_data()
        lightPositions = self._get_all_light_position()
        visitorBodyPosition = self._get_single_visitor_body_position(bodyName)
        self.obser_for_red_light_excited_visitor = np.concatenate((lightStates,
                                                                   lightDiffsePart.flatten(),
                                                                   lightPositions.flatten(),
                                                                   visitorBodyPosition.flatten()))
        #print("length self.obser_for_red_light_excited_visitor:{}".format(len(self.obser_for_red_light_excited_visitor)))
        return self.obser_for_red_light_excited_visitor
    
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
        Get all light data
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
    
    def _get_all_light_position(self):
        """
        Get all lights position
        """
        lightNum = self.lights_num
        #print("_get_all_light_position lightNum:{}".format(lightNum))
        lightPositions = np.zeros([lightNum, 3]) # 3: (x, y, z)
        for i in range(lightNum):
            res, lightPositions[i,:] = vrep.simxGetObjectPosition(self.clientID, self.lightHandles[i], -1, self._get_light_op_mode)
        return lightPositions
    
    def reset_env_for_LAS_red_light_excited_visitor(self, bodyName):
        vrep.simxStartSimulation(self.clientID, self._def_op_mode)
        observationForLAS = self._self_observe()
        observationForRedLightExcitedVisitor = self._self_observe_for_red_excited_visitor(bodyName)
        
        done = False
        rewardLAS = 0
        rewardVisitor = 0
        info = []
        return observationForLAS, observationForRedLightExcitedVisitor, rewardLAS, rewardVisitor, done, info
    
    def reset(self):
        #vrep.simxStopSimulation(self.clientID, self._def_op_mode)
        vrep.simxStartSimulation(self.clientID, self._def_op_mode)
        
        self._self_observe()
        self._reward()
        self._reward_visitor()
        done = False
        return self.observation, self.reward, self.reward_visitor, done
        
    def destroy(self):
        vrep.simxStopSimulation(self.clientID, self._def_op_mode)
        vrep.simxFinish(self.clientID)
        
    def close_connection(self):
        vrep.simxFinish(self.clientID)



#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  2 17:46:02 2018

@author: jack.lingheng.meng
"""

try:
    import vrep
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
    def __init__(self):
        print ('Program started')
        # connect to V-REP server
        vrep.simxFinish(-1) # just in case, close all opened connections
        self.clientID = vrep.simxStart('127.0.0.1',19997,True,True,5000,5) # Connect to V-REP
        if self.clientID!=-1:
            print ('Connected to remote API server')
        else:
            print ('Failed connecting to remote API server')
        # start simulate
        self._def_op_mode = vrep.simx_opmode_blocking
        self._set_joint_op_mode = vrep.simx_opmode_oneshot
        self._set_light_op_mode = vrep.simx_opmode_oneshot
        self._set_visitor_op_mode = vrep.simx_opmode_blocking
        
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
        
        # visitor
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
        
        
        proxSensorHandles = np.array(proxSensorHandles)
        proxSensorNames = np.array(proxSensorNames)
        lightHandles = np.array(lightHandles)
        lightNames = np.array(lightNames)
        jointHandles = np.array(jointHandles)
        jointNames = np.array(jointNames)
        visitorHandles = np.array(visitorHandles)
        visitorNames = np.array(visitorNames)
        self.proxSensorHandles = proxSensorHandles[proxSensorIndex]
        self.proxSensorNames = proxSensorNames[proxSensorIndex]
        self.lightHandles = lightHandles[lightIndex]
        self.lightNames = lightNames[lightIndex]
        self.jointHandles = jointHandles[jointIndex]
        self.jointNames = jointNames[jointIndex]
        self.visitorNames = visitorNames[visitorIndex]
        self.visitorHandles = visitorHandles[visitorIndex]

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
    
    def _set_single_visitor_position(self, name, position):
        visitorIndex = np.where(self.visitorNames == name)
        #print("visitorNames: {}".format(visitorNames))
        print("visitorIndex: {}".format(visitorIndex))
        if len(visitorIndex[0]) == 0:
            print("Not found visitor: {}".format(name))
        else:
            vrep.simxSetObjectPosition(self.clientID, self.visitorHandles[visitorIndex], -1, [position[0],position[1],0], self._set_visitor_op_mode)
    
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
        
        """
        proxStates, proxPosition = self._get_all_prox_data()
        lightStates, lightDiffsePart, lightSpecularPart = self._get_all_light_data()
        self.observation = np.concatenate((proxStates, lightStates, lightDiffsePart.flatten()))
        return self.observation
    
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

# Multiprocessing parallelizes multiple LAS-agents 
def multiprocessing_LAS_agent(Env, observation, reward, done, LASAgent):
    while not done:
        action = LASAgent.perceive_and_act(observation, reward, done)
        observation, reward, done, info = Env.step_LAS(action)
# Multiprocessing parallelizes multiple visitor-agents
def multiprocessing_Visitor_agent(Env, observation, reward, done, VisitorAgent):
    while not done:
        action = VisitorAgent.perceive_and_act(observation, reward, done)
        observation, reward, done, info = Env.step_visitor(action)

class LASAgent():
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
        #lights_state = np.random.randint(2,size = 39)
        lights_state = np.ones(self._light_num)
        lights_color = np.random.uniform(0,1,self._light_num*3)
        action = np.concatenate((smas, lights_state, lights_color))
        return action
    
class VisitorsControlledByOneAgent():
    def __init__(self):
        self._visitor_num = 4
        
    def perceive_and_act(self, observation, reward, done):
        self._observation = observation
        self._reward = reward
        self._done = done
        
        self._actionNew = self._act()
        return self._actionNew
    
    def _act(self):
        position = np.random.uniform(-7,7,self._visitor_num * 2) # 2: (x, y)
        return position
    
class VisitorAgent():
    def __init__(self, name):
        self._visitorName = name
        
    def perceive_and_act(self, observation, reward, done):
        self._observation = observation
        self._reward = reward
        self._done = done
        
        self._actionNew = self._act()
        return self._actionNew
    
    def _act(self):
        position = np.random.uniform(-7,7, 2) # 2: (x, y)
        return self._visitorName, position

if __name__ == '__main__':
    
    # instantiate LAS-agent
    LASAgent1 = LASAgent()
    # instantiate
    VisitorsAgent = VisitorsControlledByOneAgent()
    # instantiate a single visitor
    visitorAgent1 = VisitorAgent("TargetPosition_Visitor#1")
    # instantiate environment object
    env = LivingArchitectureEnv()
    observation, rewardLAS, rewardVisitor, done = env.reset()
    # trival agent
    i = 1
#    while not done:
#        # simple LAS-agent takes random actions
#        action = LASAgent1.perceive_and_act(observation, rewardLAS, done)
#        observation, rewardLAS, done, info = env.step_LAS(action)
#        print("Step: {}, reward: {}".format(i, rewardLAS))
#        i = i+1
#        time.sleep(0.1)

#    while not done:
#        # All visitors are controlled by one agent, and actions are randomly picked
#        visitorAction = VisitorsAgent.perceive_and_act(observation,rewardVisitor, done)
#        observation, rewardVisitor, done, info = env.step_visitor(visitorAction)
#        print("Visitor Step: {}, rewardVisitor: {}".format(i, rewardVisitor))
#        i = i+1
#        time.sleep(0.1)
    
    while not done:
        # All visitors are controlled by one agent, and actions are randomly picked
        name, visitorAction = visitorAgent1.perceive_and_act(observation,rewardVisitor, done)
        observation, rewardVisitor, done, info = env.step_single_visitor(name, visitorAction)
        print("Visitor: {} Step: {}, rewardVisitor: {}".format(name, i, rewardVisitor))
        i = i+1
        time.sleep(3)

#    for step in range(10):
#        # random actions
#        smas = np.random.randn(39)
#        #lights_state = np.random.randint(2,size = 39)
#        lights_state = np.ones(39)
#        lights_color = np.random.uniform(0,1,39*3)
#        action = np.concatenate((smas, lights_state, lights_color))
#
#        observation, reward, done, info = env.step_LAS(action)
#        print("Step: {}, reward: {}".format(i, reward))
#        i = i+1
#        time.sleep(0.1)
        
#    for step in range(5):
#        # random position
#        # the target position must be whthin a small range, if not
#        # current script cannot plan a path, so the visitor will keep still.
#        position = np.random.uniform(-7,7,2)
#        observation, reward, done, info = env.step_visitor(position)
#        
#        print("Visitor Step: {}, reward: {}".format(step, reward))
#        i = i+1
#        time.sleep(0.1)
    
    
    env.destroy()

def stop_unstoped_simulation():
    """
    If not stoped normally, call this function, or copy the following three
    lines and run them.
    """
    clientID = vrep.simxStart('127.0.0.1',19997,True,True,5000,5)
    vrep.simxStopSimulation(clientID, vrep.simx_opmode_blocking)
    vrep.simxFinish(clientID)
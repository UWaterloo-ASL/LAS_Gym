#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  2 17:46:02 2018

@author: jack.lingheng.meng
"""

try:
    from VrepRemoteApiBindings import vrep
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
        #lights_state = np.random.randint(2,size = 39)
        lights_state = np.ones(self._light_num)
        lights_color = np.random.uniform(0,1,self._light_num*3)
        #lights_color = np.array([1,0,0]*self._light_num)
        action = np.concatenate((smas, lights_state, lights_color))
        return action
    
class VisitorsControlledByOneAgent():
    """
    Single visitor agent control all visitors i.e. non-distributed
    """
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
    """
    One agent only control one visitor i.e. distributed visitor control
    """
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

# Multiprocessing parallelizes multiple LAS-agents 
def multiprocessing_LAS_agent(Env, observation, reward, done, LASAgent):
    """
    Parallel processing handles interaction of LAS
    Input:
       Env, observation, reward, done, LASAgent
    """
    i = 1
    while not done:
        action = LASAgent.perceive_and_act(observation, reward, done)
        observation, reward, done, info = Env.step_LAS(action)
        print(observation[:3])
        print("LAS Step: {}, reward: {}".format(i, reward))
        i = i+1
        time.sleep(3)
        
# Multiprocessing parallelizes multiple visitor-agents
def multiprocessing_Visitor_agent(Env, observation, reward, done, VisitorAgent):
    """
    Parallel processing handles interaction of visitor
    Input:
       Env, observation, reward, done, VisitorAgent
    """
    i = 1
    while not done:
        action = VisitorAgent.perceive_and_act(observation, reward, done)
        observation, reward, done, info = Env.step_visitor(action)
        print("Visitor Step: {}, reward: {}".format(i, reward))
        i = i+1
        time.sleep(3)

# Multiprocessing parallelizes multiple visitor
def multiprocessing_single_visitor_agent(Env, observation, reward, done, visitorAgent):
    """
    Each processing handles one single visitor
    """
    i = 1
    while not done:
        name, visitorAction = visitorAgent.perceive_and_act(observation,reward, done)
        observation, rewardVisitor, done, info = env.step_single_visitor(name, visitorAction)
        print("Visitor: {} Step: {}, rewardVisitor: {}".format(name, i, rewardVisitor))
        i = i+1
        time.sleep(3)

class RedLightExcitedVisitorAgent():
    """
    Visitor who is only excited about red light.
        Return:
            name: visitor name
            action: [move, x, y, z] if move = 0, don't move, else move.
    """
    def __init__(self, name):
        self._targetPositionName = "TargetPosition_"+name
        self._bodayName = "Body_"+name
        self.red_light_num = 0
        # threshold distance between last destination and current location
        self._distanceThreshold = np.sqrt((0.1**2) + (0.1**2))
        self._lastTargetPositionMaintainThreshold = 1000 # at least maintain 25 steps
        
        self._lastTargetPositionMaintainCounter = 0 # count how may step have elapsed for last target position
        
        self._lastDestination = []
        self._currLocation = []
        self._firstStep = True
    def perceive_and_act(self, observation, reward, done):
        self._observation = observation
        self._currLocation = observation[-3:-1] # ignore z coordinate
        self._reward = reward
        self._done = done
        
        self._actionNew = self._act()
        #print("_actionNew = {}".format(self._actionNew))
        return self._targetPositionName, self._bodayName, self._actionNew
    
    def _act(self):
        # for first step there is no lastDestination
        if self._firstStep:
            distance = 0
        else:
            # distance between lastDestination and current location
            distance = self._distance_lastDestination_currLocation(self._lastDestination, self._currLocation)
        
        red_light_positions = self._red_light_position()
        
        # only when there is red light and close to target position and maintain a maximum number to approach to target
        if self.red_light_num > 0 and \
            distance <= self._distanceThreshold and \
            self._lastTargetPositionMaintainCounter > self._lastTargetPositionMaintainThreshold:
            move = 1
            print("Red light number: {}".format(self.red_light_num))
            random_red_light = np.random.randint(0,self.red_light_num)
            position = red_light_positions[random_red_light,:]
            # each time change destination, _lastDestination will be updated 
            self._lastDestination = position[0:2] #ignore z coordinate
            self._lastTargetPositionMaintainCounter = 1
            print("Visitor Destination:{}".format(self._lastDestination))
            action = np.concatenate(([move], position.flatten()))    
        else:
            move = 0
            action = [move, 0, 0, 0]
            self._lastTargetPositionMaintainCounter += 1 # increase one
    
        return action
    
    def _distance_lastDestination_currLocation(self, lastDestination, currLocation):
        return np.sqrt(np.sum((np.array(lastDestination) - np.array(currLocation))**2))
    
    def _red_light_position(self):
        """
        Function find where are red lights:
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

if __name__ == '__main__':
    
    # Iinstantiate LAS-agent
    LASAgent1 = LASAgent()
    # Instantiate
    VisitorsAgent = VisitorsControlledByOneAgent()
    # instantiate a single visitor
    visitor = RedLightExcitedVisitorAgent("Visitor#0")
    # Instantiate environment object
    env = LivingArchitectureEnv()
    observationForLAS, observationForVisitor, rewardLAS, rewardVisitor, done, [] = env.reset_env_for_LAS_red_light_excited_visitor(visitor._bodayName)
    
    # Step counter
    i = 1
    last_time = time.time()
    while not done:
        action_LAS = LASAgent1.perceive_and_act(observationForLAS, rewardLAS, done)
        observationForLAS, rewardLAS, done, info = env.step_LAS(action_LAS)
        #print("Step: {}, reward: {}".format(i, rewardLAS))
        

        last_time = time.time()
        targetPositionName, bodyName, action_visitor = visitor.perceive_and_act(observationForVisitor,rewardVisitor,done)
        observationForVisitor, reward, done, [] = env.step_red_light_excited_visitor(targetPositionName, bodyName, action_visitor)




    """
    Test parallel
    """
#    pool = mp.Pool(processes = 2)
#    pool.apply_async(multiprocessing_LAS_agent, args = (env, 
#                                                       observation, 
#                                                       rewardLAS,
#                                                       done,
#                                                       LASAgent1))
#    
#    pool.apply_async(multiprocessing_single_visitor_agent, args = (env,
#                                                                   observation,
#                                                                   rewardVisitor,
#                                                                   done,
#                                                                   visitorAgent0))
##    pool.apply_async(multiprocessing_Visitor_agent, args = (env,
##                                                           observation,
##                                                           rewardVisitor,
##                                                           done,
##                                                           VisitorsAgent))
#    pool.close()
#    pool.join()
    
    """
    ***************************************************************************
    All actuators are controlled by a single agent i.e. non-distributed LAS-agent
    ***************************************************************************
    """
#    while not done:
#        # simple LAS-agent takes random actions
#        action = LASAgent1.perceive_and_act(observation, rewardLAS, done)
#        observation, rewardLAS, done, info = env.step_LAS(action)
#        print("Step: {}, reward: {}".format(i, rewardLAS))
#        i = i+1
#        time.sleep(0.1)
    """
    ***************************************************************************
    All visitors is controlled by a single agent i.e. non-distributed visitors-agent
    ***************************************************************************
    """
#    while not done:
#        # All visitors are controlled by one agent, and actions are randomly picked
#        visitorAction = VisitorsAgent.perceive_and_act(observation,rewardVisitor, done)
#        observation, rewardVisitor, done, info = env.step_visitor(visitorAction)
#        print("Visitor Step: {}, rewardVisitor: {}".format(i, rewardVisitor))
#        i = i+1
#        time.sleep(3)
    """
    ***************************************************************************
    Each visitor is controlled by a independent agent i.e. distributed visitor-agents
    ***************************************************************************
    """
#    while not done:
#        # All visitors are controlled by one agent, and actions are randomly picked
#        name0, visitorAction0 = visitorAgent0.perceive_and_act(observation,rewardVisitor, done)
#        observation, rewardVisitor, done, info = env.step_single_visitor(name0, visitorAction0)
#        print("Visitor: {} Step: {}, rewardVisitor: {}".format(name0, i, rewardVisitor))
#        
##        name1, visitorAction1 = visitorAgent1.perceive_and_act(observation,rewardVisitor, done)
##        observation, rewardVisitor, done, info = env.step_single_visitor(name1, visitorAction1)
##        print("Visitor: {} Step: {}, rewardVisitor: {}".format(name1, i, rewardVisitor))        
##        
##        name2, visitorAction2 = visitorAgent2.perceive_and_act(observation,rewardVisitor, done)
##        observation, rewardVisitor, done, info = env.step_single_visitor(name2, visitorAction2)
##        print("Visitor: {} Step: {}, rewardVisitor: {}".format(name2, i, rewardVisitor))
##
##        name3, visitorAction3 = visitorAgent3.perceive_and_act(observation,rewardVisitor, done)
##        observation, rewardVisitor, done, info = env.step_single_visitor(name3, visitorAction3)
##        print("Visitor: {} Step: {}, rewardVisitor: {}".format(name3, i, rewardVisitor))
#
#        i = i+1
#        time.sleep(3) # should find a way to delete the sleep and let an angent finishes its movement before setting another target

    """
    ***************************************************************************
    Randon actions: not encapsulted in agent
    ***************************************************************************
    """

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
    
    # Relase occupuied port
    env.destroy()

def stop_unstoped_simulation():
    """
    If not stoped normally, call this function, or copy the following three
    lines and run them.
    """
    clientID = vrep.simxStart('127.0.0.1',19997,True,True,5000,5)
    vrep.simxStopSimulation(clientID, vrep.simx_opmode_blocking)
    vrep.simxFinish(clientID)
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  8 21:37:59 2018

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
    def __init__(self, IP = '127.0.0.1', port = 1997):
        print ('Program started')
        # connect to V-REP server
        vrep.simxFinish(-1) # just in case, close all opened connections
        self.clientID = vrep.simxStart(IP,port,True,True,5000,5) # Connect to V-REP
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

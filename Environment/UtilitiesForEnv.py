#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  9 20:31:18 2018

@author: jack.lingheng.meng
"""

import numpy as np
import functools
import warnings

def get_all_object_name_and_handle(clientID, opMode, vrep):
    """
    This function will abstract objects' name and handle by distinguishing their 
    corresponding object type, as long as these object naming with substring "_node#". 
    
    Parameters
    ----------
    clientID:
        The clientID return from vrep.simxStart().
    opMode: 
        The operation mode to call vrep.simxGetObjectGroupData().
    vrep:
        vrep
        
    # When call vrep.simxGetObjectGroupData to abstract object name and handle
    # choose appropriate objectType parameter:
        #                   joint:  vrep.sim_object_joint_type
        #        proximity sensor:  vrep.sim_object_proximitysensor_type
        #                   light:  vrep.sim_object_light_type
        #        visitor position:  vrep.sim_object_dummy_type
    """
    dataType = 0    # 0: retrieves the object names (in stringData.)
    print("Get objects' names and handles ...")
    # proximity sensor
    proxSensorIndex = []
    rc = vrep.simx_return_initialize_error_flag
    while rc != vrep.simx_return_ok:
        rc, proxSensorHandles, intData, floatData, proxSensorNames = vrep.simxGetObjectGroupData(clientID,vrep.sim_object_proximitysensor_type, dataType, opMode)
        if rc==vrep.simx_return_ok:
            print ('Get Prox Sensor Success!!!!!') # display the reply from V-REP (in this case, just a string)
            for i, name in enumerate(proxSensorNames):
                if "_node" in name:
                    print("Proximity Sensor: {}, and handle: {}".format(name, proxSensorHandles[i]))
                    proxSensorIndex.append(i)
            break
        else:
            print ('Fail to get proximity sensors!!!')
    # light 
    lightIndex = []
    rc = vrep.simx_return_initialize_error_flag
    while rc != vrep.simx_return_ok:
        rc, lightHandles, intData, floatData, lightNames = vrep.simxGetObjectGroupData(clientID,vrep.sim_object_light_type, dataType, opMode)
        if rc==vrep.simx_return_ok:
            print ('Get Lihgt Success!!!!!') # display the reply from V-REP (in this case, just a string)
            for i, name in enumerate(lightNames):
                if "_node" in name:
                    print("Light: {}, and handle: {}".format(name, lightHandles[i]))
                    lightIndex.append(i)
            break
        else:
            print ('Fail to get lights!!!')
    # joint
    jointIndex = []
    rc = vrep.simx_return_initialize_error_flag
    while rc != vrep.simx_return_ok:
        rc, jointHandles, intData, floatData, jointNames = vrep.simxGetObjectGroupData(clientID,vrep.sim_object_joint_type, dataType, opMode)
        if rc==vrep.simx_return_ok:
            print ('Get Joint Success!!!!!') # display the reply from V-REP (in this case, just a string)
            for i, name in enumerate(jointNames):
                if "_node" in name:
                    print("Joint: {}, and handle: {}".format(name, jointHandles[i]))
                    jointIndex.append(i)
            break
        else:
            print ('Fail to get joints!!!')
    
    # Visitor targetPosition: the cylinder that visitor will approach to.
    visitorIndex = []
    rc = vrep.simx_return_initialize_error_flag
    while rc != vrep.simx_return_ok:
        rc, visitorHandles, intData, floatData, visitorNames = vrep.simxGetObjectGroupData(clientID,vrep.sim_object_dummy_type, dataType, opMode)
        if rc==vrep.simx_return_ok:
            print ('Get Visitor Target Objest Success!!!!!') # display the reply from V-REP (in this case, just a string)
            for i, name in enumerate(visitorNames):
                if "Visitor" in name:
                    print("Visitor Target: {}, and handle: {}".format(name, visitorHandles[i]))
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
    
    # All objects handels and names
    proxSensorHandles = proxSensorHandles[proxSensorIndex]
    proxSensorNames = proxSensorNames[proxSensorIndex]
    lightHandles = lightHandles[lightIndex]
    lightNames = lightNames[lightIndex]
    jointHandles = jointHandles[jointIndex]
    jointNames = jointNames[jointIndex]
    visitorTargetNames = visitorNames[visitorIndex]
    visitorTargetHandles = visitorHandles[visitorIndex]
    
    return proxSensorHandles, proxSensorNames, \
           lightHandles, lightNames, \
           jointHandles, jointNames, \
           visitorTargetNames, visitorTargetHandles

def deprecated(msg=''):
    def dep(func):
        '''This is a decorator which can be used to mark functions
        as deprecated. It will result in a warning being emitted
        when the function is used.'''

        @functools.wraps(func)
        def new_func(*args, **kwargs):
            warnings.warn_explicit(
                "Call to deprecated function {}. {}".format(func.__name__, msg),
                category=DeprecationWarning,
                filename=func.func_code.co_filename,
                lineno=func.func_code.co_firstlineno + 1
            )
            return func(*args, **kwargs)

        return new_func

    return deprecated























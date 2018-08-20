#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  8 22:25:09 2018

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

def stop_unstoped_simulation():
    """
    If not stoped normally, call this function, or copy the following three
    lines and run them.
    """
    clientID = vrep.simxStart('127.0.0.1',19997,True,True,5000,5)
    vrep.simxStopSimulation(clientID, vrep.simx_opmode_blocking)
    vrep.simxFinish(clientID)
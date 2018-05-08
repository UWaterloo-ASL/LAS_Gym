#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  8 17:12:14 2018

@author: jack.lingheng.meng
"""

try:
    from VrepRemoteApiBindings import vrep
    #from VrepRemoteApiBindings import testModule
    #import VrepRemoteApiBindings.vrep as vrep
except:
    print ('--------------------------------------------------------------')
    print ('"vrep.py" could not be imported. This means very probably that')
    print ('either "vrep.py" or the remoteApi library could not be found.')
    print ('Make sure both are in the same folder as this file,')
    print ('or appropriately adjust the file "vrep.py"')
    print ('--------------------------------------------------------------')
    print ('')
    
print("Good")
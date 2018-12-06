#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 16:18:39 2018

@author: jack.lingheng.meng
"""

from VrepRemoteApiBindings import vrep
import pandas as pd
import numpy as np
import csv
import math
import warnings
import random

node_num = 24


if __name__ == '__main__':
    clientID = vrep.simxStart('127.0.0.1',19997,True,True,5000,5)
    if clientID != -1:
        print('Connected to remote API server')
    else:
        print('Failed connecting to remote API server')
    
    position = pd.read_csv('./output.csv')
    for i in range(len(position)):
        if 'A'in position.iloc[i]['Actuator ID'] or 'B'in position.iloc[i]['Actuator ID']:
            print('Load node: {}'.format(position.iloc[i]['Actuator ID']))
            # load model
            res, handle = vrep.simxLoadModel(clientID, '/Users/jack.lingheng.meng/GoogleDrive/OverDRIVE site25926 : ASL-UWaterloo-LAS-Project/github_repositories/LAS_Gym/LAS-Scenes/ROM_Canopy_node_model/ROM_Canopy_node_model_final.ttm', 0, vrep.simx_opmode_blocking)
            if res != 0:
                print('Load model error..')
            # set position
            p_tmp = [position.iloc[i]['x'], position.iloc[i]['z'], 0.5]
            vrep.simxSetObjectPosition(clientID, handle, -1, p_tmp, vrep.simx_opmode_oneshot)
            # set orientation
            orientation_alpha = 0
            orientation_beta = 0
            orientation_gamma = random.uniform(0, 1) * 360
            vrep.simxSetObjectOrientation(clientID, handle, -1, [orientation_alpha, orientation_beta, orientation_gamma], vrep.simx_opmode_oneshot)
#    for i in range(node_num):
        
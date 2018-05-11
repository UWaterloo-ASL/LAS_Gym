#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 10 15:22:30 2018

@author: jack.lingheng.meng
"""
import tensorflow as tf
import keras.backend as K
from Environment.VrepRemoteApiBindings import vrep

from Environment.LASEnv import LASEnv
from LASAgent.ExtrinsicallyMotivatedLASAgent import ExtrinsicallyMotivatedLASAgent

if __name__ == '__main__':
    """
    This script is for interaction between extrinsically motivated LASAgent and
    Environment.
    
    Note
    ----
    You should instantiate Environment first, because LASAgent need using Environment
    object as parameter to instantiate.
    
    """
    
    sess = tf.Session()
    K.set_session(sess)
    
    # Instantiate LASEnv
    envLAS = LASEnv('127.0.0.1', 19997)
    
    # Instantiate LAS-agent
    Ext_Mot_LASAgent = ExtrinsicallyMotivatedLASAgent(envLAS, sess)
    
    # Step counter
    i = 1
    observationForLAS, rewardLAS, done, info = envLAS.reset()
    while not done:

        actionLAS = Ext_Mot_LASAgent.perceive_and_act(observationForLAS, rewardLAS, done)
        observationForLAS, rewardLAS, done, info = envLAS.step_LAS(actionLAS)
        print("LAS Step: {}, reward: {}".format(i, rewardLAS))
        i += 1
        
    envLAS.destroy()
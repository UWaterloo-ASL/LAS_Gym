#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 11 16:58:02 2018

@author: jack.lingheng.meng
"""
import matplotlib.pyplot as plt
import tensorflow as tf
import keras.backend as K

from Environment.LASEnv import LASEnv
from LASAgent.ExtrinsicallyMotivatedLASAgent import ExtrinsicallyMotivatedLASAgent
from LASAgent.RandomLASAgent import RandomLASAgent

def plot_cumulative_reward(cumulativeReward):
    line, = plt.plot(cumulativeReward)
    plt.ion()
    #plt.ylim([0,10])
    plt.show()
    plt.pause(0.0001)

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
    
    # Instantiate Extrinsically Motivated LAS-agent
    Ext_Mot_LASAgent = ExtrinsicallyMotivatedLASAgent(envLAS, 
                                                      sess, 
                                                      learnFromScratch = True)
    # Iinstantiate Random Action LAS-agent
    Random_LASAgent = RandomLASAgent(envLAS)
    
    # Step counter
    i = 1
    observationForLAS, rewardLAS, done, info = envLAS.reset()
    #while not done:
    for temp in range(10000):

        actionLAS = Ext_Mot_LASAgent.perceive_and_act(observationForLAS, rewardLAS, done)
        observationForLAS, rewardLAS, done, info = envLAS.step_LAS(actionLAS)
        print("Ext_Mot_LASAgent Step: {}, reward: {}".format(i, rewardLAS))
        i += 1
    
    observationForLAS, rewardLAS, done, info = envLAS.reset()
    for temp in range(10000):

        actionLAS = Random_LASAgent.perceive_and_act(observationForLAS, rewardLAS, done)
        observationForLAS, rewardLAS, done, info = envLAS.step_LAS(actionLAS)
        print("Random_LASAgent Step: {}, reward: {}".format(i, rewardLAS))
        i += 1
        
    envLAS.destroy()
    
    plot_cumulative_reward(Ext_Mot_LASAgent._cumulativeRewardMemory)
    plot_cumulative_reward(Random_LASAgent._cumulativeRewardMemory)
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 12 21:54:30 2018

@author: jack.lingheng.meng
"""

import time
import tensorflow as tf
import keras.backend as K
import gym
import time
from collections import deque
import matplotlib.pyplot as plt

from Environment.LASEnv import LASEnv
from LASAgent.ExtrinsicallyMotivatedLASAgent import ExtrinsicallyMotivatedLASAgent


def plot_cumulative_reward(cumulativeReward):
    line, = plt.plot(cumulativeReward)
    plt.ion()
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
#    envLAS = LASEnv('127.0.0.1', 19997)
    envLAS = gym.make("Pendulum-v0")
    
    # Instantiate LAS-agent
    Ext_Mot_LASAgent = ExtrinsicallyMotivatedLASAgent(envLAS, 
                                                      sess, 
                                                      learnFromScratch = True)
    
    # Step counter
    i = 1
    observationForLAS = envLAS.reset()
    rewardLAS = 0
    
    max_episod = 6000
    episodCumulativeReward = deque(maxlen = max_episod)

    for episod in range(max_episod):
        i = 1
        observationForLAS = envLAS.reset()
        done = False
        temp_episodCumulativeReward = 0
        while not done:
            #envLAS.render()
            actionLAS = Ext_Mot_LASAgent.perceive_and_act(observationForLAS, rewardLAS, done)
            observationForLAS, rewardLAS, done, info = envLAS.step(actionLAS)
            temp_episodCumulativeReward += rewardLAS
            #print("episod:{}, Step: {}, action: {}, reward: {}".format(episod, i, actionLAS, rewardLAS))
            i += 1
        print("episod:{}, reward: {}".format(episod, temp_episodCumulativeReward))
        #plot_cumulative_reward(episodCumulativeReward)

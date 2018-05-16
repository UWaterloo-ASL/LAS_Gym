#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 10 15:22:30 2018

@author: jack.lingheng.meng
"""
import time
import tensorflow as tf
import keras.backend as K
import numpy as np

from Environment.LASEnv import LASEnv
from LASAgent.ExtrinsicallyMotivatedLASAgent import ExtrinsicallyMotivatedLASAgent

class OrnsteinUhlenbeckActionNoise:
    def __init__(self, mu, sigma=0.3, theta=.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
                self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)

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
    Ext_Mot_LASAgent = ExtrinsicallyMotivatedLASAgent(envLAS, 
                                                      sess, 
                                                      learnFromScratch = True)
    
    actor_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(envLAS.action_space.shape[0]))
    # Step counter
    i = 1
    observationForLAS = envLAS.reset()
    rewardLAS = 0
    done = False
    while not done:
    #for temp in range(10000):

        actionLAS = Ext_Mot_LASAgent.perceive_and_act(observationForLAS, rewardLAS, done)
        actionLAS = actionLAS + actor_noise()
        observationForLAS, rewardLAS, done, info = envLAS.step(actionLAS)
        print("LAS Step: {}, reward: {}".format(i, rewardLAS))
        i += 1
        #time.sleep(0.01)
        
    envLAS.destroy()
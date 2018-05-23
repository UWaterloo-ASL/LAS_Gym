#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 15 09:43:11 2018

@author: jack.lingheng.meng
"""

import tensorflow as tf
import numpy as np
import gym
from gym import wrappers
import tflearn
import argparse
import pprint as pp
from collections import deque

from IPython.core.debugger import Tracer

from Environment.LASEnv import LASEnv
from LASAgent.LASAgent_Actor_Critic import LASAgent_Actor_Critic


import matplotlib.pyplot as plt
def plot_cumulative_reward(cumulativeReward):
    line, = plt.plot(cumulativeReward)
    plt.ion()
    plt.show()
    plt.pause(0.0001)
    
if __name__ == '__main__':

    with tf.Session() as sess:

        #env = gym.make('Pendulum-v0')
        #env = gym.make('MountainCarContinuous-v0')
        env = LASEnv('127.0.0.1', 19997)
        
        LASAgent = LASAgent_Actor_Critic(sess, env)

        #LASAgent.train()
        
        # Learning records
        episod_reward_memory = deque(maxlen = 10000)
        
        # Train parameters
        max_episodes = 50000
        max_episode_len = 1000
        render_env = False
        reward = 0
        done = False
        for i in range(max_episodes):
            observation = env.reset()   
            ep_reward = 0    
            for j in range(max_episode_len):
    
                if render_env == True:
                    env.render()
    
                # Added exploration noise
                action = LASAgent.perceive_and_act(observation,reward,done)
    
                observation, reward, done, info = env.step(action[0])
                ep_reward += reward
                if done or j == (max_episode_len-1):
                    print('| Reward: {:d} | Episode: {:d} '.format(int(ep_reward),i))
                    episod_reward_memory.append(ep_reward)
                    plot_cumulative_reward(episod_reward_memory)
                    break
        
        #env.destroy()
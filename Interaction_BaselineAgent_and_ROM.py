#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  9 08:46:49 2018

@author: daiwei.lin
"""

import time
from Environment.LASROMEnv import LASROMEnv
from LASAgent.LASBaselineAgent import LASBaselineAgent
import matplotlib.pyplot as plt



if __name__ == '__main__':

    # Instantiate environment object

    ROMenv = LASROMEnv(IP='127.0.0.1',
                 Port=19997,
                 reward_function_type='ir')
    agent = LASBaselineAgent(24,24,num_observation=1,env=ROMenv)

    observation = ROMenv.reset()
    # print("ROMenv Observation shape =", str(observation.shape))
    reward = 0
    done = False
    for _ in range(0, 20000):
        action = agent.interact(observation,reward,done)
        # action = ROMenv.action_space.sample()
        # time.sleep(0.01)
        observation, reward, done, info = ROMenv.step(action)
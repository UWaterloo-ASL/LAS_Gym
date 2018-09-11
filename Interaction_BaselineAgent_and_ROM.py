#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  9 08:46:49 2018

@author: daiwei.lin
"""


from Environment.LASROMEnv import LASROMEnv
from LASAgent.LASBaselineAgent import LASBaselineAgent




if __name__ == '__main__':

    # Instantiate environment object

    ROMenv = LASROMEnv(IP='127.0.0.1',
                 Port=19997,
                 reward_function_type='ir')
    agent = LASBaselineAgent(24,24,num_observation=10,env=ROMenv, load_pretrained_agent_flag=True)

    observation = ROMenv.reset()

    reward = 0
    done = False
    for i in range(0, 4000):
        take_action_flag, action = agent.feed_observation(observation)
        # action = agent.interact(observation,reward,done)
        if take_action_flag == True:
            # print("action taken at " + str(i) + "th step")
            observation, reward, done, info = ROMenv.step(action)
    agent.stop()


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  9 08:46:49 2018

@author: daiwei.lin
"""


from Environment.LASROMEnv import LASROMEnv
from LASAgent.LASBaselineAgent import LASBaselineAgent
from Learning import Learning


class Learning_System():
    def __init__(self, env):
        self.env = env
        self.observation = env.reset()
        self.action = env.action_space.sample()

    def get_observation(self):
        take_action_flag = True
        return take_action_flag, self.observation

    def reset(self):
        self.env.reset()

    def take_action(self, action):
        self.action = action
        # print("action :", action)
        observation, reward, done, info = self.env.step(action)
        self.observation = observation


if __name__ == '__main__':

    # Instantiate environment object

    ROMenv = LASROMEnv(IP='127.0.0.1',
                 Port=19997,
                 reward_function_type='ir')
    agent = LASBaselineAgent('LAS_Baseline_Agent', 24,6,num_observation=1,env=ROMenv, load_pretrained_agent_flag=False)

    observation = ROMenv.reset()

    reward = 0
    done = False
    for i in range(0, 1000):
        take_action_flag, action = agent.feed_observation(observation)
        # print(i)
        # action = agent.interact(observation,reward,done)
        if take_action_flag == True:
            # print("action taken at " + str(i) + "th step")
            # print(action)
            observation, reward, done, info = ROMenv.step(action)
    agent.stop()

    #==========================================#
    # Testing the integration with learning.py #
    #==========================================#

    # ROMenv = LASROMEnv(IP='127.0.0.1',
    #                                 Port=19997,
    #                                 reward_function_type='ir')
    # learning_system = Learning_System(ROMenv)
    # learning = Learning(learning_system)
    #
    # learning.setup_learning()
    #
    # # TODO: put initialization work for master script in here
    # # Check if all interactions are done.
    # while True:
    #     if learning.check_if_interactions_done():
    #         break
    #

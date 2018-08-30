#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 26 20:03:57 2018

@author: jack.lingheng.meng
"""
import logging
import tensorflow as tf
#tf.logging.set_verbosity(tf.logging.ERROR)
from datetime import datetime
import os
import gym
import matplotlib.pyplot as plt

from Environment.LASEnv import LASEnv
from LASAgent.InternalEnvOfAgent import InternalEnvOfAgent

# Logging
experiment_results_dir = os.path.join(os.path.abspath('..'), 'ROM_Experiment_results')
if not os.path.exists(experiment_results_dir):
    os.makedirs(experiment_results_dir)
logging.basicConfig(filename = os.path.join(experiment_results_dir,'ROM_experiment_'+datetime.now().strftime("%Y%m%d_%H%M%S")+'.log'), 
                    level = logging.DEBUG,
                    format='%(asctime)s:%(levelname)s: %(message)s')

if __name__ == '__main__':
    sess = tf.Session()
    # Instantiate LAS environment object
#    envLAS = LASEnv('127.0.0.1', 19997, reward_function_type = 'occupancy')
#    observation = envLAS.reset()
    env = gym.make('Pendulum-v0')
    observation = env.reset()
    #######################################################################
    #                          Instatiate LAS-Agent                       #
    #######################################################################
    # Note: 1. Set load_pretrained_agent_flag to "True" only when you have 
    #           and want to load pretrained agent.
    #       2. Keep observation unchanged if using pretrained agent.
    agent_name = 'CartPole_v0'
    observation_space = env.observation_space
    action_space = env.action_space
    observation_space_name = [], 
    action_space_name = []
    x_order_MDP = 1
    x_order_MDP_observation_type = 'concatenate_observation'
    occupancy_reward_type = 'IR_distance'
    interaction_mode = 'virtual_interaction'
    load_pretrained_agent_flag = False
    
    agent = InternalEnvOfAgent(agent_name, 
                               observation_space, 
                               action_space,
                               observation_space_name, 
                               action_space_name,
                               x_order_MDP,
                               x_order_MDP_observation_type,
                               occupancy_reward_type,
                               interaction_mode,
                               load_pretrained_agent_flag)
    #######################################################################
    max_episode_num = 1000
    try:
        for episode in range(max_episode_num):
            observation = env.reset()
            done = False
            reward = 0
            i = 1
            cumulative_reward = 0
            while not done:
                env.render()
                take_action_flag, action = agent.feed_observation(observation, reward, done)
                if take_action_flag == True:
                    observation, reward, done, info = env.step(action)
                    cumulative_reward += reward
                print('Episode: {}, Step: {}, Reward: {}'.format(episode, i, reward))
                i += 1
            plt.scatter(episode, cumulative_reward, c="r")
            if episode % 5 == 0:
                plt.show()
                plt.pause(0.05)
    except KeyboardInterrupt:
        agent.stop()

    
    
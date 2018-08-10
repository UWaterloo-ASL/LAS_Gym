#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  8 14:44:34 2018

@author: jack.lingheng.meng
"""


import tensorflow as tf
import numpy as np
import time
import gym

from Environment.LASEnv import LASEnv
from LASAgent.RandomLASAgent import RandomLASAgent
from LASAgent.LASAgent_Actor_Critic import LASAgent_Actor_Critic

from Environment.VisitorEnv import VisitorEnv
from VisitorAgent.RedLightExcitedVisitorAgent import RedLightExcitedVisitorAgent


if __name__ == '__main__':
    
    with tf.Session() as sess:
        # Instantiate MountainCar environment
        mountain_car_env = gym.make('MountainCarContinuous-v0')
        pendulum_env = gym.make('Pendulum-v0')
        
        observation_For_LAS = pendulum_env.reset()
        # Iinstantiate LAS-agent

        LASAgent1 = LASAgent_Actor_Critic(sess, pendulum_env,
                                             actor_lr = 0.0001, actor_tau = 0.001,
                                             critic_lr = 0.0001, critic_tau = 0.001, gamma = 0.99,
                                             minibatch_size = 64,
                                             max_episodes = 50000, max_episode_len = 1000,
                                             # Exploration Strategies
                                             exploration_action_noise_type = 'ou_0.2',
                                             exploration_epsilon_greedy_type = 'none',
                                             # Save Summaries
                                             save_dir = '../ROM_Experiment_results/LASAgentActorCritic/',
                                             experiment_runs = 'run5',
                                             # Save and Restore Actor-Critic Model
                                             restore_actor_model_flag = False,
                                             restore_critic_model_flag = False)
        
        # Step counter
        i = 1
        
        epo_num = 100
        for epo_i in range(epo_num):
            done = False
            reward_for_LAS = 0
            observation_For_LAS = pendulum_env.reset()
            while not done:
                pendulum_env.render()
                # LAS interacts with environment.
                actionLAS = LASAgent1.perceive_and_act(observation_For_LAS, reward_for_LAS, done)
                # delay the observing of consequence of LASAgent's action
                observation_For_LAS, reward_for_LAS, done, info = pendulum_env.step(actionLAS)
                print("LAS Step: {}, reward: {}".format(i, reward_for_LAS))
                
                i += 1
        

    

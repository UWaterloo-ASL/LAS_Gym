# -*- coding: utf-8 -*-
"""
Created on Tue Aug  7 23:16:20 2018

@author: lingheng
"""

import tensorflow as tf
import numpy as np
import time

from Environment.LASEnv import LASEnv
from LASAgent.RandomLASAgent import RandomLASAgent
from LASAgent.LASAgent_Actor_Critic import LASAgent_Actor_Critic

from Environment.VisitorEnv import VisitorEnv
from VisitorAgent.RedLightExcitedVisitorAgent import RedLightExcitedVisitorAgent


if __name__ == '__main__':
    
    with tf.Session() as sess:
        # Instantiate LAS environment object
        envLAS = LASEnv('127.0.0.1', 19997, reward_function_type = 'occupancy')
        observation_For_LAS= envLAS.reset()
        # Iinstantiate LAS-agent
#        LASAgent1 = RandomLASAgent(envLAS)
        LASAgent1 = LASAgent_Actor_Critic(sess, envLAS,
                                             actor_lr = 0.0001, actor_tau = 0.001,
                                             critic_lr = 0.0001, critic_tau = 0.001, gamma = 0.99,
                                             minibatch_size = 64,
                                             max_episodes = 50000, max_episode_len = 1000,
                                             # Exploration Strategies
                                             exploration_action_noise_type = 'ou_0.2',
                                             exploration_epsilon_greedy_type = 'none',
                                             # Save Summaries
                                             save_dir = '../ROM_Experiment_results/LASAgentActorCritic/',
                                             experiment_runs = 'run3',
                                             # Save and Restore Actor-Critic Model
                                             restore_actor_model_flag = False,
                                             restore_critic_model_flag = False)
        
        # Step counter
        i = 1
        done = False
        reward_for_LAS = 0
        while not done:
            # LAS interacts with environment.
            actionLAS = LASAgent1.perceive_and_act(observation_For_LAS, reward_for_LAS, done)
            # delay the observing of consequence of LASAgent's action
            observation_For_LAS, reward_for_LAS, done, info = envLAS.step(actionLAS)
            print("LAS Step: {}, reward: {}".format(i, reward_for_LAS))
            
            i += 1
        
        envLAS.destroy()
    

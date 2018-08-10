#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  7 18:30:12 2018

@author: jack.lingheng.meng
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
                                             experiment_runs = 'run1',
                                             # Save and Restore Actor-Critic Model
                                             restore_actor_model_flag = False,
                                             restore_critic_model_flag = False)
        
        # Instantiate visitor environment object
        envVisitor = VisitorEnv('127.0.0.1', 19997)
        # Instantiate a red light excited visitor0
        visitor0 = RedLightExcitedVisitorAgent("Visitor#0")
        # Instantiate a red light excited visitor1
        visitor1 = RedLightExcitedVisitorAgent("Visitor#1")
        # Instantiate a red light excited visitor2
        visitor2 = RedLightExcitedVisitorAgent("Visitor#2")
        # Instantiate a red light excited visitor2
        visitor3 = RedLightExcitedVisitorAgent("Visitor#3")
        
        observation_For_Visitor0 = envVisitor._self_observe(visitor0._visitorName)
        observation_For_Visitor1 = envVisitor._self_observe(visitor1._visitorName)
        observation_For_Visitor2 = envVisitor._self_observe(visitor2._visitorName)
        observation_For_Visitor3 = envVisitor._self_observe(visitor3._visitorName)
        
        # Step counter
        i = 1
        done = False
        reward_for_LAS = 0
        reward_for_Visitor0 = 0
        reward_for_Visitor1 = 0
        reward_for_Visitor2 = 0
        reward_for_Visitor3 = 0
        while not done:
            # LAS interacts with environment.
            actionLAS = LASAgent1.perceive_and_act(observation_For_LAS, reward_for_LAS, done)
            # delay the observing of consequence of LASAgent's action
            observation_For_LAS, reward_for_LAS, done, info = envLAS.step(actionLAS)
            print("LAS Step: {}, reward: {}".format(i, reward_for_LAS))
            
            # Visitor0 interacts with environment.
            visitorName0, action = visitor0.perceive_and_act(observation_For_Visitor0,reward_for_Visitor0,done)
            observation_For_Visitor0, reward_for_Visitor0, done, [] = envVisitor.step(visitorName0, action)
            print("Visitor Step: {}, Red light number: {}".format(i, visitor0.red_light_num))
            
            # Visitor1 interacts with environment.
            visitorName1, action = visitor1.perceive_and_act(observation_For_Visitor1,reward_for_Visitor1,done)
            observation_For_Visitor1, reward_for_Visitor1, done, [] = envVisitor.step(visitorName1, action)
            print("Visitor Step: {}, Red light number: {}".format(i, visitor1.red_light_num))
            
            # Visitor2 interacts with environment.
            visitorName2, action = visitor2.perceive_and_act(observation_For_Visitor2,reward_for_Visitor2,done)
            observation_For_Visitor2, reward_for_Visitor2, done, [] = envVisitor.step(visitorName2, action)
            print("Visitor Step: {}, Red light number: {}".format(i, visitor2.red_light_num))
            
            # Visitor3 interacts with environment.
            visitorName3, action = visitor3.perceive_and_act(observation_For_Visitor3,reward_for_Visitor3,done)
            observation_For_Visitor3, reward_for_Visitor3, done, [] = envVisitor.step(visitorName3, action)
            print("Visitor Step: {}, Red light number: {}".format(i, visitor3.red_light_num))
            
#            time.sleep(0.01)
            
            
            # reset all visitors out of the range of LAS
            move = 1
            action = [move, -6, np.random.randint(-3,0), 0]
            envVisitor.step(visitorName0, action)
            envVisitor.step(visitorName1, action)
            envVisitor.step(visitorName2, action)
            envVisitor.step(visitorName3, action)
            
            
            i += 1
        
        envLAS.destroy()
        envVisitor.destroy()
    

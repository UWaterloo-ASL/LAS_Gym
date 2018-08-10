#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 18 22:38:30 2018

@author: jack.lingheng.meng
"""

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
import time

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
        env = LASEnv(IP = '127.0.0.1',
                     Port = 19997,
                     reward_function_type = 'red_light_dense')
        
        LASAgent = LASAgent_Actor_Critic(sess, env,
                                         actor_lr = 0.0001, actor_tau = 0.001,
                                         critic_lr = 0.0001, critic_tau = 0.001, gamma = 0.99,
                                         minibatch_size = 64,
                                         max_episodes = 50000, max_episode_len = 1000,
                                         # Exploration Strategies
                                         exploration_action_noise_type = 'ou_0.2',
                                         exploration_epsilon_greedy_type = 'none',
                                         # Save Summaries
                                         save_dir = '../LAS_gym_results/LASAgentActorCritic/',
                                         experiment_runs = 'run1',
                                         # Save and Restore Actor-Critic Model
                                         restore_actor_model_flag = False,
                                         restore_critic_model_flag = False)

        # Learning records
        episod_reward_memory = deque(maxlen = 10000)
        
        # Train parameters
        max_episodes = 50000
        max_episode_len = 1000
        render_env = False
        reward = 0
        done = False
        start_time = time.time()
        
        try:
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
                    #time.sleep(0.5)
                print("Time elapsed:{}".format(time.time()-start_time))
        except KeyboardInterrupt:
            sess.close()
            env.destroy()
            print("Shut Down.")
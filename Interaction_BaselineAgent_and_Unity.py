#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Feb 6 08:46:49 2019

@author: daiwei.lin
"""


from LASAgent.LASBaselineAgent import *
import sys
from collections import deque
from mlagents.envs import UnityEnvironment


def initialize_unit_env(train_mode=True):

    """
    :param train_mode: True if to run the environment in training, false if in inference mode
    :return:
    """
    # Instantiate environment object

    env_name = "3DBall-OnePlatform/Unity Environment"  # Name of the Unity environment binary to launch

    print("Python version:")
    print(sys.version)

    # check Python version
    if sys.version_info[0] < 3:
        raise Exception("ERROR: ML-Agents Toolkit (v0.3 onwards) requires Python 3")

    env = UnityEnvironment(file_name=env_name)

    # Set the default brain to work with
    default_brain = env.brain_names[0]
    brain = env.brains[default_brain]

    # # Reset the environment
    # env_info = env.reset(train_mode=train_mode)[default_brain]

    # # Examine the state space for the default brain
    # print("Agent state looks like: \n{}".format(env_info.vector_observations[0]))

    return env, default_brain, brain



if __name__ == '__main__':

    train_mode = True
    unity_env, default_brain, brain = initialize_unit_env(train_mode)

    env_obs_convert = np.array([1 / 3.15, 1 / 3.15, 1 / 3.15, 1 / 4, 1 / 4, 1 / 4, 1 / 10, 1 / 10])
    agent = BaselineAgent('Baseline_Agent', 8, action_dim=2, env=unity_env, env_type='Unity', load_pretrained_agent_flag=False)

    print("Learning:")
    env_info = unity_env.reset(train_mode=train_mode)[default_brain]
    done = False
    reward = 0
    observation = env_info.vector_observations[0] * env_obs_convert
    episode_rewards = deque(10*[0], 10)
    episode_r = 0
    episode = 0
    while True:

        action = agent.interact(observation, reward, done)

        env_info = unity_env.step(action)[default_brain]
        reward = env_info.rewards[0]
        done = env_info.local_done[0]
        observation = env_info.vector_observations[0] * env_obs_convert

        episode_r += reward

        if done:
            episode += 1
            episode_rewards.append(episode_r)
            print("Total reward of {}th episode: {}".format(episode, episode_r))
            episode_r = 0

            if min(episode_rewards) >= 50 :
                print("Episode reward history {}".format(episode_rewards))
                break

    unity_env.close()


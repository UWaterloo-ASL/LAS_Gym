#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  9 08:46:49 2018

@author: daiwei.lin
"""

import time
from Environment.LASROMEnv import LASROMEnv

import argparse
import time
import os
import logging
from baselines import logger, bench
from baselines.common.misc_util import (
    set_global_seeds,
    boolean_flag,
)
import LASAgent.training as training
from baselines.ddpg.models import Actor, Critic
from baselines.ddpg.memory import Memory
from baselines.ddpg.noise import *

import gym
import tensorflow as tf
from mpi4py import MPI

import numpy as np
import matplotlib.pyplot as plt

def run(share, env, noise_type, layer_norm, evaluation, **kwargs):
    # Configure things.
    rank = MPI.COMM_WORLD.Get_rank()
    if rank != 0:
        logger.set_level(logger.DISABLED)

    # Create envs.
    # env = gym.make(env_id)
    # env = bench.Monitor(env, logger.get_dir() and os.path.join(logger.get_dir(), str(rank)))

    # if evaluation and rank == 0:
    #     eval_env = gym.make(env_id)
    #     eval_env = bench.Monitor(eval_env, os.path.join(logger.get_dir(), 'gym_eval'))
    #     env = bench.Monitor(env, None)
    # else:
    #     eval_env = None
    eval_env = None

    # Parse noise_type
    action_noise = None
    param_noise = None
    nb_actions = env.action_space.shape[-1]
    for current_noise_type in noise_type.split(','):
        current_noise_type = current_noise_type.strip()
        if current_noise_type == 'none':
            pass
        elif 'adaptive-param' in current_noise_type:
            _, stddev = current_noise_type.split('_')
            param_noise = AdaptiveParamNoiseSpec(initial_stddev=float(stddev), desired_action_stddev=float(stddev))
        elif 'normal' in current_noise_type:
            _, stddev = current_noise_type.split('_')
            action_noise = NormalActionNoise(mu=np.zeros(nb_actions), sigma=float(stddev) * np.ones(nb_actions))
        elif 'ou' in current_noise_type:
            _, stddev = current_noise_type.split('_')
            action_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(nb_actions),
                                                        sigma=float(stddev) * np.ones(nb_actions))
        else:
            raise RuntimeError('unknown noise type "{}"'.format(current_noise_type))

    # Configure components.
    memory = Memory(limit=int(1e6), action_shape=env.action_space.shape, observation_shape=env.observation_space.shape)
    critic = Critic(layer_norm=layer_norm, share=share)
    actor = Actor(nb_actions, layer_norm=layer_norm, share=share)

    # Seed everything to make things reproducible.
    # seed = seed + 1000000 * rank
    # logger.info('rank {}: seed={}, logdir={}'.format(rank, seed, logger.get_dir()))
    tf.reset_default_graph()
    # set_global_seeds(seed)
    # env.seed(seed)
    # if eval_env is not None:
    #     eval_env.seed(seed)

    # Disable logging for rank != 0 to avoid noise.
    if rank == 0:
        start_time = time.time()
    r_history = training.train(env=env, eval_env=eval_env, param_noise=param_noise,
                               action_noise=action_noise, actor=actor, critic=critic, memory=memory, **kwargs)
    env.close_connection()
    if eval_env is not None:
        eval_env.close()
    if rank == 0:
        logger.info('total runtime: {}s'.format(time.time() - start_time))

    return r_history


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # parser.add_argument('--env-id', type=str, default='HalfCheetah-v1')
    boolean_flag(parser, 'render-eval', default=False)
    boolean_flag(parser, 'layer-norm', default=True)
    boolean_flag(parser, 'render', default=False)
    boolean_flag(parser, 'normalize-returns', default=False)
    boolean_flag(parser, 'normalize-observations', default=True)
    # parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--critic-l2-reg', type=float, default=1e-2)
    parser.add_argument('--batch-size', type=int, default=64)  # per MPI worker
    parser.add_argument('--actor-lr', type=float, default=1e-4)
    parser.add_argument('--critic-lr', type=float, default=1e-3)
    boolean_flag(parser, 'popart', default=False)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--reward-scale', type=float, default=1.)
    parser.add_argument('--clip-norm', type=float, default=None)
    parser.add_argument('--nb-epochs', type=int, default=20)  # with default settings (500), perform 1M steps total
    parser.add_argument('--nb-epoch-cycles', type=int, default=20)
    parser.add_argument('--nb-train-steps', type=int, default=50)  # per epoch cycle and MPI worker
    parser.add_argument('--nb-eval-steps', type=int, default=100)  # per epoch cycle and MPI worker
    parser.add_argument('--nb-rollout-steps', type=int, default=100)  # per epoch cycle and MPI worker
    parser.add_argument('--noise-type', type=str,
                        default='adaptive-param_0.2')  # choices are adaptive-param_xx, ou_xx, normal_xx, none
    parser.add_argument('--num-timesteps', type=int, default=None)
    boolean_flag(parser, 'evaluation', default=False)
    args = parser.parse_args()
    # we don't directly specify timesteps for this script, so make sure that if we do specify them
    # they agree with the other parameters
    if args.num_timesteps is not None:
        assert (args.num_timesteps == args.nb_epochs * args.nb_epoch_cycles * args.nb_rollout_steps)
    dict_args = vars(args)
    del dict_args['num_timesteps']
    return dict_args


def avg_smoothing(x, window_len):
    """
    Smoothing the plot by taking moving average with a window length = windown_len

    """

    length = len(x)
    x_smo = np.zeros([1,length])
    half_window = int((window_len - 1 )/2)
    for i in range(0,length):
        if i < half_window:
            x_smo[0,i] = np.mean(x[0:i + half_window + 1])
        elif i >= (length - half_window):
            x_smo[0,i] = np.mean(x[i - half_window :i])
        else:
            x_smo[0,i] = np.mean(x[i - half_window : i + half_window + 1])
    return x_smo[0,:]



if __name__ == '__main__':

    # Instantiate environment object

    env = LASROMEnv(IP='127.0.0.1',
                 Port=19997,
                 reward_function_type='ir')

    # run DDPG with args defined
    args = parse_args()
    if MPI.COMM_WORLD.Get_rank() == 0:
        logger.configure()
    # Run actual script.
    reward_history = np.zeros((1, 400))
    # r_history[0, :] = np.array(run(**args, share=True))
    reward_history = np.array(run(**args,env=env, share=False))
    x = np.linspace(1,400,400)
    reward_history_smo = avg_smoothing(reward_history,15)
    plt.plot(x,reward_history)
    plt.plot(x,reward_history_smo)

import argparse
import time
import datetime
import os
import logging
import pickle
import csv
from collections import deque


from baselines.ddpg.ddpg import DDPG
import baselines.common.tf_util as U

from baselines import logger, bench
from baselines.common.misc_util import (
    set_global_seeds,
    boolean_flag,
)
from baselines.ddpg.models import Actor, Critic
from baselines.ddpg.memory import Memory
from baselines.ddpg.noise import *

import gym
from gym import spaces
import tensorflow as tf
from mpi4py import MPI

import numpy as np


class LASBaselineAgent:
    def __init__(self, agent_name, observation_dim, action_dim, num_observation=20, env=None, load_pretrained_agent_flag=False ):
        self.baseline_agent = BaselineAgent(agent_name, observation_dim, action_dim, env, load_pretrained_agent_flag)
        self.internal_env = InternalEnvironment(observation_dim, action_dim, num_observation)

    def feed_observation(self,observation):
        """
        Diagram of structure:

        -----------------------------------------------------------------
        |                                             LASBaselineAgent   |
        |                                                                |
        |  action,flag         observation                               |
        |    /\                    |                                     |
        |    |                    \/                                     |
        |  -------------------------------                               |
        |  |    Internal Environment     |                               |
        |  -------------------------------                               |
        |   /\                     |  Flt observation, reward, flag      |
        |   |  action             \/                                     |
        |  ---------------------------                                   |
        |  |      Baseline agent     |                                   |
        |  ---------------------------                                   |
        |                                                                |
        ------------------------------------------------------------------

        """
        is_new_observation, filtered_observation, reward = self.internal_env.feed_observation(observation)
        if is_new_observation:
            action = self.baseline_agent.interact(filtered_observation, reward, done=False)
            take_action_flag, action = self.internal_env.take_action(action)
            return take_action_flag, action
        else:
            return False,[]

    def stop(self):
        self.baseline_agent.stop()


class InternalEnvironment:
    def __init__(self,observation_dim, action_dim, num_observation):
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.num_observation = num_observation

        self.observation_cnt = 0
        self.observation_group = np.zeros((num_observation, observation_dim))

    def feed_observation(self, observation):
        """
        1. Feed observation into internal environment
        2. perform filtering
        3. calculate reward
        :param observation:
        :return:
        """

        flt_observation = np.zeros((1,self.observation_dim), dtype=np.float32)
        reward = 0
        # stack observations
        self.observation_group[self.observation_cnt] = observation
        self.observation_cnt += 1

        # Apply filter once observation group is fully updated
        # After that, calculate the reward based on filtered observation
        if self.observation_cnt >= self.num_observation:
            self.observation_cnt = 0
            # self.flt_prev_observation = self.flt_observation
            flt_observation = self._filter(self.observation_group)
            is_new_observation = True

            reward = self._cal_reward(flt_observation)

        else:
            is_new_observation = False

        return is_new_observation, flt_observation, reward

    def take_action(self,action):
        take_action_flag = True
        return take_action_flag, action

    def _cal_reward(self, flt_observation):
        """
        Calculate the extrinsic rewards based on the filtered observation
        Filtered observation should have same size as observation space
        :return: reward
        """
        reward = 0
        for i in range(flt_observation.shape[0]):
            reward += flt_observation[i]
        return reward

    def _filter(self, signal):
        """
        Averaging filter

        signal: numpy matrix, one row is one observation

        """
        return np.mean(signal, axis = 0)


class BaselineAgent:
    def __init__(self, agent_name, observation_dim, action_dim, env=None, env_type='VREP', load_pretrained_agent_flag=False ):

        self.name = agent_name
        #=======================================#
        # Get parameters defined in parse_arg() #
        #=======================================#
        args = self.parse_args()
        noise_type = args['noise_type']
        layer_norm = args['layer_norm']
        if MPI.COMM_WORLD.Get_rank() == 0:
            logger.configure()

        # share = False
        rank = MPI.COMM_WORLD.Get_rank()
        if rank != 0:
            logger.set_level(logger.DISABLED)

        eval_env = None

        # ===================================== #
        # Define observation and action space   #
        # ===================================== #
        if env is None:
            self.env = None
            obs_max = np.array([1.] * observation_dim)
            obs_min = np.array([0.] * observation_dim)
            act_max = np.array([1] * action_dim)
            act_min = np.array([-1] * action_dim)
            self.observation_space = spaces.Box(obs_min, obs_max, dtype=np.float32)
            self.action_space = spaces.Box(act_min, act_max, dtype=np.float32)
        else:
            self.env = env
            if env_type == 'VREP':
                self.action_space = env.action_space
                self.observation_space = env.observation_space
            elif env_type == 'Unity':
                obs_max = np.array([1.] * observation_dim)
                obs_min = np.array([-1] * observation_dim)
                act_max = np.array([1] * action_dim)
                act_min = np.array([-1] * action_dim)
            self.observation_space = spaces.Box(obs_min, obs_max, dtype=np.float32)
            self.action_space = spaces.Box(act_min, act_max, dtype=np.float32)

        self.reward = 0
        self.action = np.zeros(self.action_space.shape[0])
        self.prev_observation = np.zeros(self.observation_space.shape[0], dtype=np.float32 )
        # =============#
        # Define noise #
        # =============#

        # Parse noise_type
        action_noise = None
        param_noise = None
        nb_actions = self.action_space.shape[-1]

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

        #============================================#
        # Configure neural nets for actor and critic #
        #============================================#

        self.memory = Memory(limit=int(1e6), action_shape=self.action_space.shape,
                        observation_shape=self.observation_space.shape)
        critic = Critic(layer_norm=layer_norm)#, share=share)
        actor = Actor(nb_actions, layer_norm=layer_norm) #, share=share)

        tf.reset_default_graph()

        # Disable logging for rank != 0 to avoid noise.
        if rank == 0:
            start_time = time.time()




        assert (np.abs(self.action_space.low) == self.action_space.high).all()  # we assume symmetric actions.
        max_action = self.action_space.high
        # logger.info('scaling actions by {} before executing in env'.format(max_action))

        #=======================#
        # Create learning agent #
        #=======================#

        # Get learning parameters from args
        gamma = args['gamma']
        tau = 0.01 # <==== according to training .py   tau=0.01 by default
        normalize_returns = args['normalize_returns']
        normalize_observations = args['normalize_observations']
        self.batch_size = args['batch_size'] # used in interact() as well
        critic_l2_reg = args['critic_l2_reg']
        actor_lr = args['actor_lr']
        critic_lr = args['critic_lr']
        popart = args['popart']
        clip_norm = args['clip_norm']
        reward_scale = args['reward_scale']

        # create learning agent
        self.agent = DDPG(actor, critic, self.memory, self.observation_space.shape, self.action_space.shape,
                     gamma=gamma, tau=tau, normalize_returns=normalize_returns,
                     normalize_observations=normalize_observations,
                     batch_size=self.batch_size, action_noise=action_noise, param_noise=param_noise,
                     critic_l2_reg=critic_l2_reg,
                     actor_lr=actor_lr, critic_lr=critic_lr, enable_popart=popart, clip_norm=clip_norm,
                     reward_scale=reward_scale)
        # logger.info('Using agent with the following configuration:')
        # logger.info(str(self.agent.__dict__.items()))

        # Reward histories

        # eval_episode_rewards_history = deque(maxlen=100)
        self.episode_rewards_history = deque(maxlen=100)
        self.avg_episode_rewards_history = []

        #===========================#
        # Training cycle parameter #
        #==========================#

        self.nb_epochs = args['nb_epochs']
        self.epoch_cnt = 0
        self.nb_epoch_cycles = args['nb_epoch_cycles']
        self.epoch_cycle_cnt = 0
        self.nb_rollout_steps = args['nb_rollout_steps']
        self.rollout_step_cnt = 0
        self.nb_train_steps = args['nb_train_steps']
        self.training_step_cnt = 0

        #========================#
        # Model saving           #
        #========================#
        self.model_dir = os.path.join(os.path.abspath('..'), 'save','model')
        self.log_dir = os.path.join(os.path.abspath('..'), 'save','log')
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)


        #=======================#
        # Initialize tf session #
        #=======================#

        self.sess = U.make_session(num_cpu=1, make_default=True)

        self.agent.initialize(self.sess)
        self.saver = tf.train.Saver()
        if load_pretrained_agent_flag == True:
            self._load_model(self.model_dir)

        # self.sess.graph.finalize()


        self.agent.reset()

        #==============#
        # logging info #
        #==============#
        self.episode_reward = 0.
        self.episode_step = 0
        self.episodes = 0
        self.t = 0

        # epoch = 0
        self.start_time = time.time()

        self.epoch_episode_rewards = []
        self.epoch_episode_steps = []
        self.epoch_episode_eval_rewards = []
        self.epoch_episode_eval_steps = []
        self.epoch_start_time = time.time()
        self.epoch_actions = []
        self.epoch_qs = [] # Q values
        self.epoch_episodes = 0
        self.param_noise_adaption_interval = 50

        #==========================================#
        # default prescripted behaviour parameters #
        #==========================================#
        # Actions are a list of values:
        # [1a moth, 1a RS, 1b moth, 1b RS, 1c moth, 1c RS, 1d, 2, 3, 4, 5a, 5b, 6a, 6b, 7, 8a, 8b]        #
        # constant parameters: 5a, 5b, 6a, 6b, 8a, 8b
        # min_val = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 15000, 60000, 0, 0, 0, 1000, 5, 200])
        #
        # max_val = np.array(
        #     [5000, 5000, 5000, 5000, 5000, 5000, 255, 5000, 5000, 60000, 100000, 10000, 100, 5000, 5000, 200, 400])
        #
        # default_val = np.array(
        #     [1500, 1500, 1000, 1000, 2500, 2500, 200, 1500, 300, 45000, 90000, 5000, 40, 1800, 700, 120, 240])
        #
        # self.default_para = (default_val - min_val) / (max_val - min_val)
        # self.variable_para_index = [0,1,2,3,4,5,6,7,8,9,14]
        # assert len(self.variable_para_index) == self.action_space.shape[0], "variable_para_index={}, action_space={}".format(len(self.variable_para_index), self.action_space.shape[0])


    def interact(self, observation, reward = 0, done = False):
        """
        Receive observation and produce action

        """

        # # For the case of simulator only,
        # # since with the simulator, we always use interact() instead of feed_observation()
        # if self.env is not None:
        #     self.observe(observation)
        with self.sess.as_default():
            action, q = self.agent.pi(observation, apply_noise=True, compute_Q=True)
            assert action.shape == self.action_space.shape

            # Execute next action.


            # scale for execution in env (as far as DDPG is concerned, every action is in [-1, 1])
            # if self.rollout_step_cnt == self.nb_rollout_steps - 1:
            #     done = True
            self.t += 1

            self.episode_reward += reward  # <<<<<<<<<<<<<<<<<<<<<<<<
            self.episode_step += 1


            # Book-keeping.
            self.epoch_actions.append(action)
            self.epoch_qs.append(q)
            # Note: self.action correspond to prev_observation
            #       reward correspond to observation
            if self.action is not None:
                self.agent.store_transition(self.prev_observation, self.action, reward, observation, done)
            self.action = action
            self.reward = reward
            self.prev_observation = observation

            self._save_log(self.log_dir,[datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),self.prev_observation, self.action, reward])

            # Logging the training reward info for debug purpose
            if done:
                # Episode done.
                self.epoch_episode_rewards.append(self.episode_reward)
                self.episode_rewards_history.append(self.episode_reward)
                self.epoch_episode_steps.append(self.episode_step)
                self.avg_episode_rewards_history.append(self.episode_reward / self.episode_step)
                self.episode_reward = 0.
                self.episode_step = 0
                self.epoch_episodes += 1
                self.episodes += 1

                self.agent.reset()
                if self.env is not None:
                    obs = self.env.reset()

            self.rollout_step_cnt += 1
            # Training
            # Everytime interact() is called, it will train the model by nb_train_steps times
            if self.rollout_step_cnt >= self.nb_rollout_steps:

                self.epoch_actor_losses = []
                self.epoch_critic_losses = []
                self.epoch_adaptive_distances = []
                for t_train in range(self.nb_train_steps):
                    # Adapt param noise, if necessary.
                    if self.memory.nb_entries >= self.batch_size and t_train % self.param_noise_adaption_interval == 0:
                        distance = self.agent.adapt_param_noise()
                        self.epoch_adaptive_distances.append(distance)

                    cl, al = self.agent.train()
                    self.epoch_critic_losses.append(cl)
                    self.epoch_actor_losses.append(al)
                    self.agent.update_target_net()

                self.rollout_step_cnt = 0
                self.epoch_cycle_cnt += 1

            #==============#
            # Create stats #
            #==============#
            if self.epoch_cycle_cnt >= self.nb_epoch_cycles:
                # rank = MPI.COMM_WORLD.Get_rank()
                mpi_size = MPI.COMM_WORLD.Get_size()
                # Log stats.
                # XXX shouldn't call np.mean on variable length lists
                duration = time.time() - self.start_time
                stats = self.agent.get_stats()
                combined_stats = stats.copy()
                combined_stats['rollout/return'] = np.mean(self.epoch_episode_rewards)
                combined_stats['rollout/return_history'] = np.mean(self.episode_rewards_history)
                combined_stats['rollout/episode_steps'] = np.mean(self.epoch_episode_steps)
                combined_stats['rollout/actions_mean'] = np.mean(self.epoch_actions)
                combined_stats['rollout/Q_mean'] = np.mean(self.epoch_qs)
                combined_stats['train/loss_actor'] = np.mean(self.epoch_actor_losses)
                combined_stats['train/loss_critic'] = np.mean(self.epoch_critic_losses)
                combined_stats['train/param_noise_distance'] = np.mean(self.epoch_adaptive_distances)
                combined_stats['total/duration'] = duration
                combined_stats['total/steps_per_second'] = float(self.t) / float(duration)
                combined_stats['total/episodes'] = self.episodes
                combined_stats['rollout/episodes'] = self.epoch_episodes
                combined_stats['rollout/actions_std'] = np.std(self.epoch_actions)

                def as_scalar(x):
                    if isinstance(x, np.ndarray):
                        assert x.size == 1
                        return x[0]
                    elif np.isscalar(x):
                        return x
                    else:
                        raise ValueError('expected scalar, got %s' % x)

                combined_stats_sums = MPI.COMM_WORLD.allreduce(np.array([as_scalar(x) for x in combined_stats.values()]))
                combined_stats = {k: v / mpi_size for (k, v) in zip(combined_stats.keys(), combined_stats_sums)}

                # Total statistics.
                combined_stats['total/epochs'] = self.epoch_cnt + 1
                combined_stats['total/steps'] = self.t

                for key in sorted(combined_stats.keys()):
                    logger.record_tabular(key, combined_stats[key])
                logger.dump_tabular()
                logger.info('')


                self.epoch_cycle_cnt = 0
                self.epoch_cnt += 1
            #===================#
            # Stop the learning #
            #===================#

            if self.epoch_cnt >= self.nb_epochs:
                self.stop()

        return action

    def _load_model(self, model_dir):
        """
        Load a pre-trained model from file specified in model directory
        """
        with self.sess.as_default():
            # saver = tf.train.Saver()
            file_dir = os.path.join(model_dir,'param_action.ckpt')
            self.saver.restore(self.sess, file_dir)
            print("Model loaded from {}", format(file_dir))

    def _save_model(self, model_dir):
        """
        Save a model to specified directory
        :param model_dir:

        """

        # saver = tf.train.Saver() # move this up to initialization
        file_dir = os.path.join(model_dir, 'param_action.ckpt')
        path = self.saver.save(self.sess, file_dir)
        print("Model saved at {}".format(path))

    def _save_log(self, save_dir, data):
        """
        Save action, observation and rewards in a local file
        :param save_dir:
        """
        date = datetime.datetime.today().strftime('%Y-%m-%d')
        file_dir = os.path.join(save_dir, date + ".csv")
        with open(file_dir, 'a') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(data)

    def parse_args(self):
        """
        This is the place to define training variables. Still using the code from OpenAI Baseline library

        # total step = nb_epochs * nb_epoch_cycles * nb_rollout_steps

        """
        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        # parser.add_argument('--env-id', type=str, default='HalfCheetah-v1')
        boolean_flag(parser, 'render-eval', default=False)
        boolean_flag(parser, 'layer-norm', default=True)
        boolean_flag(parser, 'render', default=False)
        boolean_flag(parser, 'normalize-returns', default=False)
        boolean_flag(parser, 'normalize-observations', default=True) # default = True
        # parser.add_argument('--seed', help='RNG seed', type=int, default=0)
        parser.add_argument('--critic-l2-reg', type=float, default=1e-2)
        parser.add_argument('--batch-size', type=int, default=64)  # per MPI worker
        parser.add_argument('--actor-lr', type=float, default=1e-4)
        parser.add_argument('--critic-lr', type=float, default=1e-3)
        boolean_flag(parser, 'popart', default=False)
        parser.add_argument('--gamma', type=float, default=0.99)
        parser.add_argument('--reward-scale', type=float, default=1.)
        parser.add_argument('--clip-norm', type=float, default=None)
        parser.add_argument('--nb-epochs', type=int, default=500)  # with default settings (500), perform 1M steps total
        parser.add_argument('--nb-epoch-cycles', type=int, default=10)
        parser.add_argument('--nb-train-steps', type=int, default=20)  # per epoch cycle and MPI worker
        parser.add_argument('--nb-eval-steps', type=int, default=100)  # per epoch cycle and MPI worker
        parser.add_argument('--nb-rollout-steps', type=int, default=10)  # per epoch cycle and MPI worker  default 50
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

    def stop(self):
        """
        Stop learning and store the information

        """
        # if using the simulator
        if self.env is not None:
            print("close connection to simulator")
            self.env.close_connection()

        # save the model
        self._save_model(self.model_dir)
        # close the tf session
        self.sess.close()



#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 15 09:41:09 2018

@author: jack.lingheng.meng
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import gym
from gym import wrappers
import tflearn
from tflearn.models.dnn import DNN
import argparse
import pprint as pp
from collections import deque

from IPython.core.debugger import Tracer

from Environment.LASEnv import LASEnv

from LASAgent.replay_buffer import ReplayBuffer

# ===========================
#   Actor and Critic DNNs
# ===========================

class ActorNetwork(object):
    """
    Input to the network is the state, output is the action
    under a deterministic policy.

    The output layer activation is a tanh to keep the action
    between -action_bound and action_bound
    """

    def __init__(self, sess, observation_space,  action_space,
                 learning_rate, tau, batch_size,
                 restore_model_flag=False,
                 actor_model_save_path_and_name = 'results/models/actor_model.ckpt',
                 target_actor_model_save_path_and_name = 'results/models/target_actor_model.ckpt'):
        
        self.sess = sess
        self.s_dim = observation_space.shape[0]
        self.a_dim = action_space.shape[0]
        self.action_bound_high = action_space.high
        self.action_bound_low = action_space.low
        self.learning_rate = learning_rate
        self.tau = tau
        self.batch_size = batch_size
        
        # Info for load pre-trained actor models
        self.restore_model_flag = restore_model_flag
        self.actor_model_save_path_and_name = actor_model_save_path_and_name
        self.target_actor_model_save_path_and_name = target_actor_model_save_path_and_name
        
        # Initialize or Restore Actor Network
        self.inputs, self.out, self.scaled_out, self._actor_model = self.create_actor_network()
        if self.restore_model_flag == True:
            print('restore actor model')
            self._actor_model.load(self.actor_model_save_path_and_name)
        
        self.network_params = tf.trainable_variables()

        # Initialize or Restore Target Network
        self.target_inputs, self.target_out, self.target_scaled_out, self._target_actor_model = self.create_actor_network()
        if self.restore_model_flag == True:
            print('restore target actor model')
            self._target_actor_model.load(self.target_actor_model_save_path_and_name)
            
            
        self.target_network_params = tf.trainable_variables()[
            len(self.network_params):]

        # Op for periodically updating target network with online network
        # weights
        self.update_target_network_params = \
            [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) +
                                                  tf.multiply(self.target_network_params[i], 1. - self.tau))
                for i in range(len(self.target_network_params))]

        # This gradient will be provided by the critic network
        self.action_gradient = tf.placeholder(tf.float32, [None, self.a_dim])

        # Combine the gradients here
        self.unnormalized_actor_gradients = tf.gradients(
            self.scaled_out, self.network_params, -self.action_gradient)
        
        self.actor_gradients = list(map(lambda x: tf.divide(x, self.batch_size), self.unnormalized_actor_gradients))

        # Optimization Op
        self.optimize = tf.train.AdamOptimizer(self.learning_rate).\
            apply_gradients(zip(self.actor_gradients, self.network_params))

        self.num_trainable_vars = len(
            self.network_params) + len(self.target_network_params)

    def create_actor_network(self):
        """
        
        """
        inputs = tflearn.input_data(shape=[None, self.s_dim],name = 'ActorInput')
        net = tflearn.fully_connected(inputs, 400)
        net = tflearn.layers.normalization.batch_normalization(net)
        net = tflearn.activations.relu(net)
        net = tflearn.fully_connected(net, 300)
        net = tflearn.layers.normalization.batch_normalization(net)
        net = tflearn.activations.relu(net)
        # Final layer weights are init to Uniform[-3e-3, 3e-3]
        w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
        out = tflearn.fully_connected(
            net, self.a_dim, activation='tanh', weights_init=w_init, name = 'ActorOutput') # action space is shifted to [-1,1]
        # Scale output to -action_bound to action_bound
        scaled_out = tf.multiply(out, self.action_bound_high, name = 'ActorScaledOutput')
        model = DNN(scaled_out, tensorboard_verbose = 3)
        return inputs, out, scaled_out, model
        
    def save_actor_network(self):
        """save actor and target actor model"""
        self._actor_model.save(self.actor_model_save_path_and_name)
        self._target_actor_model.save(self.target_actor_model_save_path_and_name)
        print('Save actor networks.')

    def train(self, inputs, a_gradient):
        self.sess.run(self.optimize, feed_dict={
            self.inputs: inputs,
            self.action_gradient: a_gradient
        })

    def predict(self, inputs):
        return self.sess.run(self.scaled_out, feed_dict={
            self.inputs: inputs
        })
#        return self._actor_model.predict(inputs)

    def predict_target(self, inputs):
        return self.sess.run(self.target_scaled_out, feed_dict={
            self.target_inputs: inputs
        })
#        return self._target_actor_model.predict(inputs)

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)

    def get_num_trainable_vars(self):
        return self.num_trainable_vars


class CriticNetwork(object):
    """
    Input to the network is the state and action, output is Q(s,a).
    The action must be obtained from the output of the Actor network.

    """

    def __init__(self, sess, observation_space, action_space,
                 learning_rate, tau, gamma, num_actor_vars,
                 restore_model_flag=False,
                 critic_model_save_path_and_name = 'results/models/critic_model.ckpt',
                 target_critic_model_save_path_and_name = 'results/models/target_critic_model.ckpt'):
        
        self.sess = sess
        self.s_dim = observation_space.shape[0]
        self.a_dim = action_space.shape[0]
        self.learning_rate = learning_rate
        self.tau = tau
        self.gamma = gamma
        
        # Info for load pre-trained actor models
        self.restore_model_flag = restore_model_flag
        self.critic_model_save_path_and_name = critic_model_save_path_and_name
        self.target_critic_model_save_path_and_name = target_critic_model_save_path_and_name
        
        # Create the critic network
        self.inputs, self.action, self.out, self._critic_model = self.create_critic_network()
        if self.restore_model_flag == True:
            print('restore critic model')
            self._critic_model.load(self.critic_model_save_path_and_name)

        self.network_params = tf.trainable_variables()[num_actor_vars:]

        # Target Network
        self.target_inputs, self.target_action, self.target_out, self._target_critic_model = self.create_critic_network()
        if self.restore_model_flag == True:
            print('restore target critic model')
            self._target_critic_model.load(self.target_critic_model_save_path_and_name)
            
        self.target_network_params = tf.trainable_variables()[(len(self.network_params) + num_actor_vars):]

        # Op for periodically updating target network with online network
        # weights with regularization
        self.update_target_network_params = \
            [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) \
            + tf.multiply(self.target_network_params[i], 1. - self.tau))
                for i in range(len(self.target_network_params))]

        # Network target (y_i)
        self.predicted_q_value = tf.placeholder(tf.float32, [None, 1])

        # Define loss and optimization Op
        self.loss = tflearn.mean_square(self.predicted_q_value, self.out)
        self.optimize = tf.train.AdamOptimizer(
            self.learning_rate).minimize(self.loss)

        # Get the gradient of the net w.r.t. the action.
        # For each action in the minibatch (i.e., for each x in xs),
        # this will sum up the gradients of each critic output in the minibatch
        # w.r.t. that action. Each output is independent of all
        # actions except for one.
        self.action_grads = tf.gradients(self.out, self.action)

    def create_critic_network(self):
        inputs = tflearn.input_data(shape=[None, self.s_dim], name = 'CriticInputState')
        action = tflearn.input_data(shape=[None, self.a_dim], name = 'CriticInputAction')
        h1_inputs = tflearn.fully_connected(inputs, 400)
        h1_norm_inputs = tflearn.layers.normalization.batch_normalization(h1_inputs)
        h1_act_inputs = tflearn.activations.relu(h1_norm_inputs)

        # Add the action tensor in the 2nd hidden layer
        # Use two temp layers to get the corresponding weights and biases
        t1 = tflearn.fully_connected(h1_act_inputs, 300)
        t2 = tflearn.fully_connected(action, 300)

        net = tflearn.activation(
            tf.matmul(h1_act_inputs, t1.W) + tf.matmul(action, t2.W) + t2.b, activation='relu')

        # linear layer connected to 1 output representing Q(s,a)
        # Weights are init to Uniform[-3e-3, 3e-3]
        w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
        out = tflearn.fully_connected(net, 1, weights_init=w_init, name = 'CriticOutput')
        model = DNN(out, tensorboard_verbose = 3)
        return inputs, action, out, model

    def save_critic_network(self):
        """
        Function used to save critic and target critic model
        """
        self._critic_model.save(self.critic_model_save_path_and_name)
        self._target_critic_model.save(self.target_critic_model_save_path_and_name)
        print('Save critic networks.')

    def train(self, inputs, action, predicted_q_value):
        return self.sess.run([self.out, self.optimize], feed_dict={
            self.inputs: inputs,
            self.action: action,
            self.predicted_q_value: predicted_q_value
        })

    def predict(self, inputs, action):
        return self.sess.run(self.out, feed_dict={
            self.inputs: inputs,
            self.action: action
        })

    def predict_target(self, inputs, action):
        return self.sess.run(self.target_out, feed_dict={
            self.target_inputs: inputs,
            self.target_action: action
        })

    def action_gradients(self, inputs, actions):
        return self.sess.run(self.action_grads, feed_dict={
            self.inputs: inputs,
            self.action: actions
        })

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)

# Taken from https://github.com/openai/baselines/blob/master/baselines/ddpg/noise.py, which is
# based on http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
class OrnsteinUhlenbeckActionNoise:
    def __init__(self, mu, sigma=0.3, theta=.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
                self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)

class LASAgent_Actor_Critic():
    def __init__(self, sess, env,
                 actor_lr = 0.0001, actor_tau = 0.001,
                 critic_lr = 0.0001, critic_tau = 0.001, gamma = 0.99,
                 minibatch_size = 64,
                 max_episodes = 50000, max_episode_len = 1000,
                 summary_dir = './results/LASAgent_Actor_Critic/',
                 experiemnt_runs = 'run1',
                 restore_actor_model_flag = False,
                 actor_model_save_path_and_name = 'results/models/actor_model.ckpt',
                 target_actor_model_save_path_and_name = 'results/models/target_actor_model.ckpt',
                 restore_critic_model_flag=False,
                 critic_model_save_path_and_name = 'results/models/critic_model.ckpt',
                 target_critic_model_save_path_and_name = 'results/models/target_critic_model.ckpt'):
        """
        Intialize LASAgent.
        
        Parameters
        ----------
        actor_lr: float default = 0.0001
            actor model learning rate
        actor_tau: float default = 0.001
            target actor model updating weight
        critic_lr: float default = 0.0001
            critic model learning rate
        critic_tau: float default = 0.001
            target critic model updating weight
        gamma default:int = 0.99
            future reward discounting paramter
        minibatch_size:int default = 64
            size of minibabtch
        max_episodes:int default = 50000
            maximum number of episodes
        max_episode_len: int default = 1000
            maximum lenght of each episode
        summary_dir: string default='./results/LASAgent_Actor_Critic')
            directory to save tensorflow summaries
        experiemnt_runs: str default = 'run1'
            directory to save summaries of a specific run 
        restore_actor_model_flag: bool default = False
            indicate whether load pre-trained actor model
        actor_model_save_path_and_name: str default = 'results/models/actor_model.ckpt'
            directory given where to save and load pre-trained actor model
        target_actor_model_save_path_and_name = 'results/models/target_actor_model.ckpt'
            directory given where to save and load pre-trained target actor model
        restore_critic_model_flag: bool default = False
            indicate whetther load pre-trained critic model
        critic_model_save_path_and_name: str default = 'results/models/critic_model.ckpt'
            directory given where to save and load pre-trained critic model
        target_critic_model_save_path_and_name: str default = 'results/models/target_critic_model.ckpt')
            directory given where to save and load pre-trained target critic model
        """
        # Init Environment Related Parameters
        self.sess = sess
        self.env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space 
        
        # Temporary Memory
        self.first_experience = True
        self.observation_old = []
        self.action_old = []
        self.reward_new = []
        self.observation_new = []
        
        # Reply buffer or Hard Memory
        self.buffer_size = 1000000
        self.random_seed = 1234
        self.replay_buffer = ReplayBuffer(self.buffer_size, self.random_seed)
        
        self.minibatch_size = 64
        # =================================================================== #
        #                     Initialize Actor Model                          #
        # =================================================================== #
        self.actor_lr = actor_lr
        self.actor_tau = actor_tau
        # Restore Pre-trained Actor Modles
        self.restore_actor_model_flag = restore_actor_model_flag
        self.actor_model_save_path_and_name = actor_model_save_path_and_name
        self.target_actor_model_save_path_and_name = target_actor_model_save_path_and_name
        
        self.actor_model = ActorNetwork(self.sess, 
                                        self.observation_space, 
                                        self.action_space,
                                        self.actor_lr, 
                                        self.actor_tau,
                                        self.minibatch_size,
                                        self.restore_actor_model_flag,
                                        self.actor_model_save_path_and_name,
                                        self.target_actor_model_save_path_and_name)
        # =================================================================== #
        #                     Initialize Critic Model                         #
        # =================================================================== #
        self.critic_lr = critic_lr
        self.critic_tau = critic_tau
        self.gamma = gamma
        # Restore Pre-trained Critic Model
        self.restore_critic_model_flag = restore_critic_model_flag,
        self.critic_model_save_path_and_name = critic_model_save_path_and_name
        self.target_critic_model_save_path_and_name = target_critic_model_save_path_and_name
        
        self.critic_model = CriticNetwork(self.sess,
                                          self.observation_space,
                                          self.action_space,
                                          self.critic_lr,
                                          self.critic_tau,
                                          self.gamma,
                                          self.actor_model.get_num_trainable_vars(),
                                          self.restore_critic_model_flag,
                                          self.critic_model_save_path_and_name,
                                          self.target_critic_model_save_path_and_name)
        # Exploration Strategies
        # 1. Actor noise to maintain exploration
        self.actor_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(self.action_space.shape[0]))
        # 2. epsilon-greedy
        self.epsilon = 1
        self.epsilon_decay = 0.999
        
        # Training Hyper-parameters and initialization
        self.max_episodes = max_episodes
        self.max_episode_len = max_episode_len
        self.episode_counter = 1
        self.steps_counter = 1  # Steps elapsed in one episode
        self.render_env = False
        
        # Initialize Tranable Variables
        self.sess.run(tf.global_variables_initializer())
        self.actor_model.update_target_network()
        self.critic_model.update_target_network()
        
        # Init Summary Ops
        self.summary_dir = summary_dir
        self.experiemnt_runs = experiemnt_runs
        self.episode_rewards = 0
        self.summary_ops, self.summary_vars = self._init_build_summaries()
        self.writer = tf.summary.FileWriter(self.summary_dir+self.experiemnt_runs, self.sess.graph)
        
    def _init_build_summaries(self):
        """
        Function used for building summaries.
        """
        episode_rewards = tf.Variable(0.)
        tf.summary.scalar("Accumulated_Rewards", episode_rewards)
    
        summary_vars = [episode_rewards]
        summary_ops = tf.summary.merge_all()
    
        return summary_ops, summary_vars
    
    def perceive_and_act(self, observation, reward, done):
        """
        Perceive observation and reward, then return action based on current
        observation.
        
        Parameters
        ----------
        observation: ndarray
            observation
        reward: float
            reward of previous action
        done: bool
            whether current simulation is done
            
        Returns
        -------
        action: ndarray
            action generated by agent
        """
        self.observation_new = observation
        self.reward_new = reward
        self.done = done
        #Tracer()()
        # If this is the first action, no one single complete experience to remember
        if self.first_experience:
            action = self.actor_model.predict(np.reshape(self.observation_new, (1, self.actor_model.s_dim))) + self.actor_noise()
            
            self.action_old = action
            self.observation_old = self.observation_new
            self.first_experience = False
            
            return action
        # Action, added exploration noise
        action = self._act()
        
        self.episode_rewards += self.reward_new
        # Save Episode Summaries
        if self.steps_counter == self.max_episode_len:
            #Tracer()()
            summary_str = self.sess.run(self.summary_ops, feed_dict = {self.summary_vars[0]: self.episode_rewards})
            self.writer.add_summary(summary_str,self.episode_counter)
            self.writer.flush()
            # Reset Summary Data
            self.steps_counter = 1
            self.episode_rewards = 0
            self.episode_counter += 1
            
            # Save trained models each episode
            self.actor_model.save_actor_network()
            self.critic_model.save_critic_network()
        else:
            self.steps_counter += 1
        
        # Remember experience
        self.replay_buffer.add(np.reshape(self.observation_old, (self.actor_model.s_dim,)),
                               np.reshape(self.action_old, (self.actor_model.a_dim,)),
                               self.reward_new,
                               self.done,
                               np.reshape(self.observation_new, (self.actor_model.s_dim,)))

        # Train
        self._train()
        
        # Before return, set observation and action as old.
        self.observation_old = self.observation_new
        self.action_old = action
        
        return action
    
    def _act(self):
        """
        Produce action based on current observation.
        """
#        if np.random.rand(1) <= self.epsilon:
#            action = self.action_space.sample()
#            if self.epsilon > 0.05:
#                self.epsilon *= self.epsilon_decay
#            #print("epsilon:{}".format(self.epsilon))
#            return action
        # Action, added exploration noise
        action = self.actor_model.predict(np.reshape(self.observation_new, (1, self.actor_model.s_dim))) + self.actor_noise()
        return action
 
    def _train(self):
        """
        Train Actor-Critic Model
        """
        # Keep adding experience to the memory until
        # there are at least minibatch size samples
        if self.replay_buffer.size() > self.minibatch_size:
            s_batch, a_batch, r_batch, t_batch, s2_batch = \
                self.replay_buffer.sample_batch(int(self.minibatch_size))

            # Calculate targets
            target_q = self.critic_model.predict_target(
                s2_batch, self.actor_model.predict_target(s2_batch))

            y_i = []
            for k in range(int(self.minibatch_size)):
                if t_batch[k]:
                    y_i.append(r_batch[k])
                else:
                    y_i.append(r_batch[k] + self.critic_model.gamma * target_q[k])

            # Update the critic given the targets
            predicted_q_value, _ = self.critic_model.train(
                s_batch, a_batch, np.reshape(y_i, (int(self.minibatch_size), 1)))
            
            # Update the actor policy using the sampled gradient
            a_outs = self.actor_model.predict(s_batch)
            grads = self.critic_model.action_gradients(s_batch, a_outs)
            self.actor_model.train(s_batch, grads[0])

            # Update target networks
            self.actor_model.update_target_network()
            self.critic_model.update_target_network()
            
            
            


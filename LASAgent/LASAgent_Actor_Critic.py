#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 15 09:41:09 2018

@author: jack.lingheng.meng
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
from tensorflow import keras
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
from LASAgent.noise import AdaptiveParamNoiseSpec,NormalActionNoise,OrnsteinUhlenbeckActionNoise

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
# ===========================
#   Environment Model DNNs
# ===========================
class EnvironmentModelNetwork(object):
    def __init__(self, sess, observation_space, action_space,
                 learning_rate = 0.0001,
                 # Save Environment Model
                 env_model_save_path = 'results/models/',
                 # Restore Environment Model
                 env_restore_flag = False,
                 env_model_restore_path_and_name = 'results/models/env_model.ckpt'):
        """
        Initialize environment model.
        
        Parameters
        ----------
        learning_rate: float default = 0.0001
            learning rate
        env_model_save_path: str default = 'results/models/'
            the path to save environment model
                 # Restore Environment Model
        env_restore_flag: bool default = False
            idicate whether restore a previously trained environment model
        env_model_restore_path_and_name: str default = 'results/models/env_model.ckpt'
            the path and name of previously trained environment model that is 
            going to restore
        """
        self.sess = sess
        self.observation_dim = observation_space.shape[0]
        self.action_dim = action_space.shape[0]
        self.learning_rate = learning_rate
        # Parameters for save environment model
        self.env_model_save_path = env_model_save_path
        # Parameters for restoring environment model
        self.env_restore_flag = env_restore_flag
        self.env_model_restore_path_and_name = env_model_restore_path_and_name
        
        # Create Environment Model
        self.obs_input, self.act_input, self.obs_output, self.reward_output, self.env_model = self.create_environment_model_network()
        if self.env_restore_flag == True:
            print('restore environment model')
            self.env_model = keras.models.load_model(self.env_model_restore_path_and_name)
       
    def create_environment_model_network(self):
        """
        Create environment model network.
        
        Returns
        -------
        observation_inputs: tf.Tensor
            tensor of observation input
        action_inputs: tf.Tensor
            tensor of action input
        observation_output: tf.Tensor
            tensor of observation output
        reward_output: tf.Tensor
            tensor of reward output
        model: Keras model
        """
        observation_inputs = keras.Input(shape=(self.observation_dim,), name='EnvModel_observation_input')
        obs_h1 = keras.layers.Dense(units = 400, activation='relu')(observation_inputs)
        obs_h1 = keras.layers.BatchNormalization()(obs_h1)
        action_inputs = keras.Input(shape=(self.action_dim,), name='EnvModel_action_input')
        act_h1 = keras.layers.Dense(units = 400, activation='relu')(action_inputs)
        act_h1 = keras.layers.BatchNormalization()(act_h1)
        
        merged_inputs = keras.layers.add([obs_h1, act_h1], name = 'EnvModel_merged_input')
        merged_h1 = keras.layers.Dense(units=300, activation = 'relu')(merged_inputs)
        merged_h1 = keras.layers.BatchNormalization()(merged_h1)
        
        observation_output = keras.layers.Dense(units = (self.observation_dim), activation = 'relu')(merged_h1)
        reward_output = keras.layers.Dense(units = (1), activation = 'relu')(merged_h1)
        
        model = keras.Model(inputs = [observation_inputs, action_inputs],
                            outputs = [observation_output, reward_output])
        adam = keras.optimizers.Adam(lr = 0.001)
        model.compile(optimizer = adam, loss = 'mse')
        # Plot Environment Model
        keras.utils.plot_model(model, to_file = 'actor_model.png',show_shapes=True, show_layer_names=True)
        
        return observation_inputs, action_inputs, observation_output, reward_output, model 
    
    def train(self, observation_inputs, action_inputs, observation_output, reward_output):
        """
        Train environment model.
        
        Parameters
        ----------
        observation_inputs: ndarray
            observation at time step t
        action_inputs: ndarray
            action at time step t
        observation_output: ndarray
            array combined of (observation,reward): 
                observation at time step t+1
                reward at time step t
        """
        self.env_model.fit(x=[observation_inputs,action_inputs],
                           y=[observation_output, reward_output],
                           verbose = 0)
    
    def evaluate(self, observation_inputs, action_inputs, observation_output, reward_output):
        """
        Evaluate environment model. The testing samples should be different from
        training samples.
        
        Parameters
        ----------
        observation_inputs: ndarray
            observation at time step t
        action_inputs: ndarray
            action at time step t
        observation_output: ndarray
            array combined of (observation,reward): 
                observation at time step t+1
                reward at time step t
        """
        loss = self.env_model.evaluate(x=[observation_inputs,action_inputs],
                                        y=[observation_output, reward_output],
                                        verbose = 0)
        return np.mean(loss)
    
    def save_environment_model_network(self, env_save_time):
        """
        save environment model
        Parameters
        ----------
        env_save_time: int
            the time step when saving the environment model
        """
        keras.models.save_model(self.env_model,
                                filepath = self.env_model_save_path + '/env_model_' + str(env_save_time) + '.ckpt')
        
    def predict_target(self, observation, action):
        """
        Predict next step observation and reward based on current observation
        and action.
        
        Parameters
        ----------
        observation: ndarray
            observation at time step t
        action: ndarray
            action taken at time step t
        Returns
        -------
        prediction: list [predicted_observation, predicted_reward]
            prediction of observation and reward resulted from taken action in 
            observation.
        """
        return self.env_model.predict([observation, action])


# ===========================
#   Living Architecture System Agent
# ===========================

class LASAgent_Actor_Critic():
    def __init__(self, sess, env,
                 actor_lr = 0.0001, actor_tau = 0.001,
                 critic_lr = 0.0001, critic_tau = 0.001, gamma = 0.99,
                 minibatch_size = 64,
                 max_episodes = 50000, max_episode_len = 1000,
                 # Exploration Strategies
                 exploration_action_noise_type = 'ou_0.2',
                 exploration_epsilon_greedy_type = 'none',
                 # Save Summaries
                 save_dir = './results/LASAgentActorCritic_5NodesEnv/',
                 experiment_runs = 'run1',
                 # Save and Restore Actor-Critic Model
                 restore_actor_model_flag = False,
                 restore_critic_model_flag = False):
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
        exploration_action_noise_type: str default = 'ou_0.2',
            set up action noise. Options:
                1. 'none' (no action noise)
                2. 'adaptive-param_0.2'
                3. 'normal_0.2'
                4. 'ou_0.2' 
        exploration_epsilon_greedy_type: str default = 'none',
            set up epsilon-greedy.
            1. If exploration_epsilon_greedy_type == 'none', no epsilon-greedy.
            2. 'epsilon-greedy-max_1_min_0.05_decay_0.999'
        save_dir: string default='./results/LASAgent_Actor_Critic')
            directory to save tensorflow summaries and pre-trained models
        experiment_runs: str default = 'run1'
            directory to save summaries of a specific run 
        restore_actor_model_flag: bool default = False
            indicate whether load pre-trained actor model
        restore_critic_model_flag: bool default = False
            indicate whetther load pre-trained critic model
        """
        # Produce a string describes experiment setting
        self.experiment_setting = ['LAS Environment:' + '<br />' +\
                                   '1. action_space: ' + str(env.action_space.shape) + '<br />' +\
                                   '2. observation_space: ' + str(env.observation_space.shape) + '<br /><br />' +\
                                   'LASAgent Hyper-parameters: ' + '<br />' +\
                                   '1. actor_lr: ' + str(actor_lr) + '<br />' +\
                                   '2. actor_tau: ' + str(actor_tau) + '<br />' +\
                                   '3. critic_lr: ' + str(critic_lr) + '<br />' +\
                                   '4. critic_tau: ' + str(critic_tau) + '<br />' +\
                                   '5. gamma: ' + str(gamma) + '<br />' +\
                                   '6. minibatch_size: ' + str(minibatch_size) + '<br />' +\
                                   '7. max_episodes: ' + str(max_episodes) + '<br />' +\
                                   '8. max_episode_len: ' + str(max_episode_len) + '<br />' +\
                                   '9. action_noise_type: ' + str(exploration_action_noise_type) + '<br />' +\
                                   '10.epsilon_greedy_type: ' + str(exploration_epsilon_greedy_type) + '<br />' +\
                                   '11.restore_actor_model_flag: ' + str(restore_actor_model_flag) + '<br />' +\
                                   '12.restore_critic_model_flag: ' + str(restore_critic_model_flag)][0]
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
        # =================================================================== #
        #                 Initialize Replay Buffers for                       #
        #         Extrinsic and Intrinsic Policy, and Environment Model        #
        # =================================================================== #         
        # ********************************************* #
        #         Replay Buffer for Extrinsic Policy    #
        # ********************************************* #
        self.buffer_size = 1000000
        self.random_seed = 1234
        self.replay_buffer = ReplayBuffer(self.buffer_size, self.random_seed)
        # ********************************************* #
        #        Replay Buffer for Environment Model    #
        # ********************************************* #
        # 5% experience will be saved in test buffer
        self.env_model_buffer_test_ratio = 0.05
        # 1. Training Buffer
        self.env_model_train_buffer_size = 100000
        self.env_model_train_buffer = ReplayBuffer(self.env_model_train_buffer_size, self.random_seed)
        # 2. Test Buffer:
        #    For examing whether our environemnt model is converged, we can 
        #    save a small set of testing samples that will not be used to 
        #    training environment. Note that this test set should not have too
        #    much past experiences either too much recent experiences.
        self.env_model_test_buffer_size = 100000
        self.env_model_test_buffer = ReplayBuffer(self.env_model_test_buffer_size, self.random_seed)
        # ********************************************* #
        #        Replay Buffer for Intrinsic Policy     #
        # ********************************************* #
        
        
        # =================================================================== #
        #      Initialize Parameters for Both Actor and Critic Model          #
        # =================================================================== #        
        self.minibatch_size = 64
        # Common Saving Directory
        self.models_dir = save_dir + 'models/' + experiment_runs
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)
        # Restore Pre-trained Actor Modles
        self.restore_actor_model_flag = restore_actor_model_flag
        self.actor_model_save_path_and_name = self.models_dir + '/actor_model.ckpt'
        self.target_actor_model_save_path_and_name = self.models_dir + '/target_actor_model.ckpt'
        # Restore Pre-trained Critic Model
        self.restore_critic_model_flag = restore_critic_model_flag
        self.critic_model_save_path_and_name = self.models_dir + 'critic_model.ckpt'
        self.target_critic_model_save_path_and_name = self.models_dir + 'target_critic_model.ckpt'
        # =================================================================== #
        #       Initialize Extrinsically Motivated Actor-Critic Model         #
        # =================================================================== #
        # Extrinsically Motivated Actor
        self.actor_lr = actor_lr
        self.actor_tau = actor_tau
        self.actor_model = ActorNetwork(self.sess, 
                                        self.observation_space, 
                                        self.action_space,
                                        self.actor_lr, 
                                        self.actor_tau,
                                        self.minibatch_size,
                                        self.restore_actor_model_flag,
                                        self.actor_model_save_path_and_name,
                                        self.target_actor_model_save_path_and_name)
        # Extrinsically Motivated Critic
        self.critic_lr = critic_lr
        self.critic_tau = critic_tau
        self.gamma = gamma
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
        # =================================================================== #
        #                     Initialize Environment Model                    #
        # =================================================================== #
        self.env_model_lr = 0.0001
        self.env_model_minibatch_size = 200
        self.env_model_save_path = self.models_dir
        self.env_restore_flag = False
        self.env_model_restore_path_and_name = 'results/models/env_model.ckpt'
        self.environment_model = EnvironmentModelNetwork(self.sess,
                                                         self.observation_space,
                                                         self.action_space,
                                                         self.env_model_lr,
                                                         self.env_model_save_path,
                                                         self.env_restore_flag,
                                                         self.env_model_restore_path_and_name)
        # =================================================================== #
        #       Initialize Intrinsically Motivated Actor-Critic Model         #
        # =================================================================== #
        # Intrinsically Motivated Actor
        
        # Intrinsically Motivated Critic
        
        # =================================================================== #
        #                  Initialize Exploration Strategies                  #
        # =================================================================== #        
        # 1. Action Noise to Maintain Exploration
        self.exploration_action_noise_type = exploration_action_noise_type
        self.actor_noise = self._init_action_noise(self.exploration_action_noise_type, self.action_space.shape[0])
        # 2. Epsilon-Greedy
        self.exploration_epsilon_greedy_type = exploration_epsilon_greedy_type # 'epsilon-greedy-max_1_min_0.05_decay_0.999'
        self.epsilon_max, self.epsilon_min, self.epsilon_decay = self._init_epsilon_greedy(self.exploration_epsilon_greedy_type)
        self.epsilon = self.epsilon_max
        # 3. Intrinsic Motivation (for future implementation)
        
        # =================================================================== #
        #                Initialize Training Hyper-parameters                 #
        # =================================================================== #        
        self.max_episodes = max_episodes
        self.max_episode_len = max_episode_len
        self.episode_counter = 1
        self.steps_counter = 1      # Steps elapsed in one episode
        self.total_step_counter = 1 # Steps elapsed in whole life
        self.render_env = False
        # =================================================================== #
        #                       Initialize Summary Ops                        #
        # =================================================================== #        
        self.save_dir = save_dir
        self.experiment_runs = experiment_runs
        self.episode_rewards = 0
        self.summary_ops_accu_rewards, self.summary_vars_accu_rewards = self._init_summarize_accumulated_rewards()
        self.summary_ops_action_reward, self.summary_action, self.summary_reward = self._init_summarize_action_and_reward()
        self.writer = tf.summary.FileWriter(self.save_dir+'summary/'+self.experiment_runs, self.sess.graph)
        # Summarize Experiment Setting
        self.summary_ops_experiment_setting, self.summary_experiment_setting = self._init_summarize_experiment_setting()
        summary_str_experiment_setting = self.sess.run(self.summary_ops_experiment_setting,
                                                       feed_dict = {self.summary_experiment_setting: self.experiment_setting})
        self.writer.add_summary(summary_str_experiment_setting)
        # Summarize Environment Model Training
        self.summary_ops_env_loss, self.summary_env_loss = self._init_summarize_environment_model()
        # =================================================================== #
        #                    Initialize Tranable Variables                    #
        # =================================================================== #        
        self.sess.run(tf.global_variables_initializer()) # Make sure to initialze tensors before use
        self.actor_model.update_target_network()
        self.critic_model.update_target_network()

# =================================================================== #
#                       Main Interaction Functions                    #
# =================================================================== #
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
        
        # *********************************** # 
        #            Produce Action           #
        # *********************************** #
        # If this is the first action, no one single complete experience to remember
        if self.first_experience:
            action = self._act()
            
            self.action_old = action
            self.observation_old = self.observation_new
            self.first_experience = False
            self.total_step_counter += 1
            return action
        # Action, added exploration noise
        action = self._act()
        
        # *********************************** # 
        #    Write Summaries for Analysis     #
        # *********************************** #
        # 1. Save Step Summaries
        summary_str_action_rewards = self.sess.run(self.summary_ops_action_reward,
                                                   feed_dict = {self.summary_action: self.action_old,
                                                                self.summary_reward: self.reward_new})
        self.writer.add_summary(summary_str_action_rewards, self.total_step_counter)
        # 2. Save Episode Summaries
        self.episode_rewards += self.reward_new
        if self.steps_counter == self.max_episode_len or done == True:
            #Tracer()()
            summary_str = self.sess.run(self.summary_ops_accu_rewards,
                                        feed_dict = {self.summary_vars_accu_rewards: self.episode_rewards})
            self.writer.add_summary(summary_str,self.episode_counter)
            self.writer.flush()
            # Reset Summary Data
            self.steps_counter = 1
            self.episode_rewards = 0
            self.episode_counter += 1
            
            # Save trained models each episode
            self.actor_model.save_actor_network()
            self.critic_model.save_critic_network()
            # Save Environment Model
            self.environment_model.save_environment_model_network(self.episode_counter)
        else:
            self.steps_counter += 1
        
        # *********************************** # 
        #        Remember Experiences         #
        # *********************************** #
        # 1. Extrinsic Policy Replay Buffer
        self.replay_buffer.add(np.reshape(self.observation_old, (self.actor_model.s_dim,)),
                               np.reshape(self.action_old, (self.actor_model.a_dim,)),
                               self.reward_new,
                               self.done,
                               np.reshape(self.observation_new, (self.actor_model.s_dim,)))
        # 2. Environment Model Replay Buffer
        if np.random.rand(1) <= self.env_model_buffer_test_ratio:
            self.env_model_test_buffer.add(np.reshape(self.observation_old, (self.actor_model.s_dim,)),
                                           np.reshape(self.action_old, (self.actor_model.a_dim,)),
                                           self.reward_new,
                                           self.done,
                                           np.reshape(self.observation_new, (self.actor_model.s_dim,)))
        else:
            self.env_model_train_buffer.add(np.reshape(self.observation_old, (self.actor_model.s_dim,)),
                                            np.reshape(self.action_old, (self.actor_model.a_dim,)),
                                            self.reward_new,
                                            self.done,
                                            np.reshape(self.observation_new, (self.actor_model.s_dim,)))
        # 3. Intrinsc Policy Replay Buffer
        #    The Learning Progress plays the role of intrinsic reward
        
        
        
        # *********************************** # 
        #             Train Models            #
        # *********************************** #
        self._train()
        
        # Before return, set observation and action as old.
        self.observation_old = self.observation_new
        self.action_old = action
        self.total_step_counter += 1
        return action
    
    def _act(self):
        """
        Produce action based on current observation.
        """
        # Epsilon-Greedy
        if self.exploration_epsilon_greedy_type != 'none':
            if np.random.rand(1) <= self.epsilon:
                action = np.reshape(self.action_space.sample(), [1,self.action_space.shape[0]])
                if self.epsilon > self.epsilon_min:
                    self.epsilon *= self.epsilon_decay
                print("epsilon:{}".format(self.epsilon))
                return action
        # Action Noise
        if self.exploration_action_noise_type != 'none':
            action = self.actor_model.predict(np.reshape(self.observation_new, (1, self.actor_model.s_dim))) + self.actor_noise() #The noise is too huge.
        else:
            action = self.actor_model.predict(np.reshape(self.observation_new, (1, self.actor_model.s_dim)))
        
        return action
 
    def _train(self):
        """
        Train Actor-Critic Model
        """
        # Keep adding experience to the memory until
        # there are at least minibatch size samples
        # *********************************** # 
        #        Train Actor-Critic Model     #
        # *********************************** #
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
        # *********************************** # 
        #        Train Environment Model      #
        # *********************************** #
        if self.env_model_train_buffer.size() > self.env_model_minibatch_size:
            #Tracer()()
            s_batch, a_batch, r_batch, t_batch, s2_batch = self.env_model_train_buffer.sample_batch(int(self.env_model_minibatch_size))
            self.environment_model.train(s_batch,
                                         a_batch,
                                         s2_batch,
                                         np.reshape(r_batch, (int(self.env_model_minibatch_size), 1)))
            # evaluate on test buffer
            if self.env_model_test_buffer.size() > 0:
                s_batch_test, a_batch_test, r_batch_test, t_batch_test, s2_batch_test =\
                        self.env_model_test_buffer.sample_batch(int(self.env_model_test_buffer.size()))
                env_loss = self.environment_model.evaluate(s_batch_test,
                                                           a_batch_test,
                                                           s2_batch_test,
                                                           np.reshape(r_batch_test, (int(self.env_model_test_buffer.size()), 1)))
                # Summaries of Training Environment Model
                summary_env_loss_str = self.sess.run(self.summary_ops_env_loss,
                                                     feed_dict = {self.summary_env_loss: env_loss})
                self.writer.add_summary(summary_env_loss_str, self.total_step_counter)
            

# =================================================================== #
#                    Initialization Helper Functions                  #
# =================================================================== # 
    def _init_epsilon_greedy(self, exploration_epsilon_greedy_type):
        """
        Initialize hyper-parameters for epsilon-greedy.
        Parameters
        ----------
        exploration_epsilon_greedy_type: str default = 'epsilon-greedy-max_1_min_0.05_decay_0.999'
            str for setting epsilon greedy. Please keep the format and just change float numbers.
            For default 'epsilon-greedy-max_1_min_0.05_decay_0.999', it means:
                maximum epsilon = 1
                minimum spsilom = 0.05
                epsilon decay = 0.999
            If exploration_epsilon_greedy_type == 'none', no epsilon-greedy.
        
        Returns
        -------
        epsilon_max: float
            maximum epsilon
        epsilon_min: float
            minimum spsilom
        epsilon_decay: float
            epsilon decay
        """
        if exploration_epsilon_greedy_type == 'none':
            epsilon_max=0
            epsilon_min=0
            epsilon_decay=0
        else:
            _, epsilon_max, _, epsilon_min, _, epsilon_decay = exploration_epsilon_greedy_type.split('_')
        
        return epsilon_max, epsilon_min, epsilon_decay
    
    def _init_action_noise(self, action_noise_type='ou_0.2', nb_actions=1):
        """
        Initialize action noise object.
        
        Parameters
        ----------
        action_noise_type: str default = 'ou_0.2'
            type of action noise:
                1. 'none' (no action noise)
                2. 'adaptive-param_0.2'
                3. 'normal_0.2'
                4. 'ou_0.2'
        nb_actions: int default = 1
            dimension of action space
        
        Returns
        -------
            action_noise: object of ActionNoise class.
        """
        if action_noise_type == 'none':
            pass
        elif 'adaptive-param' in action_noise_type:
            _, stddev = action_noise_type.split('_')
            param_noise = AdaptiveParamNoiseSpec(initial_stddev=float(stddev), desired_action_stddev=float(stddev))
            return param_noise
        elif 'normal' in action_noise_type:
            _, stddev = action_noise_type.split('_')
            action_noise = NormalActionNoise(mu=np.zeros(nb_actions), sigma=float(stddev) * np.ones(nb_actions))
            return action_noise
        elif 'ou' in action_noise_type:
            _, stddev = action_noise_type.split('_')
            action_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(nb_actions), sigma=float(stddev) * np.ones(nb_actions))
            return action_noise
        else:
            raise RuntimeError('unknown noise type "{}"'.format(action_noise_type))
    
    def _init_summarize_accumulated_rewards(self):
        """
        Function used for building summaries.
        """
        episode_rewards = tf.Variable(0.)
        episode_rewards_sum = tf.summary.scalar("Accumulated_Rewards", episode_rewards)
        
        summary_ops = tf.summary.merge([episode_rewards_sum])
    
        return summary_ops, episode_rewards
    
    def _init_summarize_action_and_reward(self):
        """
        Histogram summaries of action and reward
        
        Returns
        -------
        summary_ops: tf.ops
            ops to summarize action and reward
        action: tf.placeholder
            placeholder for feeding action
        reward: tf.placeholder
            placeholder for feeding reward
        """
        action = tf.placeholder(tf.float32, shape=[1,self.action_space.shape[0]])
        reward = tf.placeholder(tf.float32)
        
        action_sum = tf.summary.histogram("action", action)
        reward_sum = tf.summary.histogram("reward", reward)
        
        summary_ops = tf.summary.merge([action_sum, reward_sum])
        return summary_ops, action, reward
    
    def _init_summarize_experiment_setting(self):
        """
        Summarize experiment setting
        """
        experiemnt_setting = tf.placeholder(tf.string)
        experiemnt_setting_sum = tf.summary.text('Experiment_setting', experiemnt_setting)
        summary_ops = tf.summary.merge([experiemnt_setting_sum])
        return summary_ops, experiemnt_setting
       
    def _init_summarize_environment_model(self):
        """
        Summarize environment model training
        """
        loss = tf.placeholder(dtype = tf.float32)
        loss_sum = tf.summary.scalar('loss_env_model', loss)
        loss_sum_op = tf.summary.merge([loss_sum])
        return loss_sum_op, loss


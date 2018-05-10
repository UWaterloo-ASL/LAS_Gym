#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 10 09:08:07 2018

@author: jack.lingheng.meng
"""
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input
from keras.layers.merge import Add, Multiply
from keras.optimizers import Adam
import keras.backend as K
import tensorflow as tf
import random
import numpy as np
from collections import deque

class ExtrinsicallyMotivatedLASAgent:
    def __init__(self, env, sess):
        """
        Initialize extrinsically motivated LASAgent.
        
        Parameters
        ----------
        env: environment object
            environment object, mainly to get "env.action_space" and "env.observation_space"
        sess: tensorflow session
            
        """
        self.env = env
        self.sess = sess
        # 
        self.actionSpace = env.action_space             # gym.spaces.Box object
        self.observationSpace = env.observation_space   # gym.spaces.Box object
        # ========================================================================= #
        #            Initialize hyper-parameters for learning model                 #
        # ========================================================================= # 
        self._learningRate = 0.001
        self._epsilon = 1.0
        self._epsilonDecay = 0.995
        self._gamma = 0.95
        self._tau = 0.125
        # ========================================================================= #
        #                 Initialize Memory and Environment model                   #
        # ========================================================================= #        
        # Long-term hard memory: storing every experience
        self._memory = deque(maxlen = 5000)
        # Temporary memory: variables about last single experience
        self._observationOld = []   # observation at time t
        self._observationNew = []   # observation at time t+1
        self._actionOld = []        # action at time t
        self._actionNew = []        # action at time t+1
        # Environment model or World model (this is different from environment,
        # because this environment is learned from experiences)
        #   mapping from (currentState, action) to (nextState, reward)
        self._environmentModel = self._create_environment_model()
        # ========================================================================= #
        #              Initialize extrinsically motivated Actor-Critic model        #
        #                    i.e. extrinsic policy and value function               #
        # ========================================================================= #
        # ************************************************************************* #
        #                             Extrinsic Actor Model                         #
        # ************************************************************************* #         
        self._actorStateInput, self._actorModel = self._create_actor_model()
        _, self._targetActorModel = self._create_actor_model()
        self._actorCriticGrad = tf.placeholder(tf.float32, [None, self.actionSpace.shape[0]])
        
        actorModelWeights = self._actorModel.trainable_weights
        # why the initial gradient in ys is negative of gradient from actor-critic??
        self._actorGrad = tf.gradients(self._actorModel.output, actorModelWeights, -self._actorCriticGrad) 
        grads = zip(self._actorGrad, actorModelWeights)
        self.optimize = tf.train.AdamOptimizer(self._learningRate).apply_gradients(grads)
        # ************************************************************************* #
        #                            Extrinsic Critic Model                         #
        # ************************************************************************* #
        self._criticStateInput, self._criticActionInput, self._criticModel = self._create_critic_model()
        _, _, self._targetCriticModel = self._create_critic_model()
        self._criticGrads = tf.gradients(self._criticModel.output, self._criticActionInput) 
        
        # Initialize for later gradient calculations
        self.sess.run(tf.global_variables_initializer())

    # ========================================================================= #
    #                   Components or Model Definitions                         #
    # ========================================================================= #
    def _create_environment_model(self):
        """
        Create environemnt model. Environment model relys on various time-dependence, 
        so it's different from sensorimotor model which only considers one step
        time-dependence.
        
        Function
        --------
        Mapping: (currentState, action) -> (nextState, reward)
        
        To do
        -----
        I think we will need long-short-term memory to implement this.
        
        """
        model = []
        return model
    
    def _create_actor_model(self):
        """
        Actor model corresponds to a policy that maps from currentState to action.
        """
        stateInput = Input(shape=self.observationSpace.shape)
        h1 = Dense(48, activation = 'relu')(stateInput)
        h2 = Dense(64, activation = 'relu')(h1)
        h3 = Dense(48, activation = 'relu')(h2)
        actionOutput = Dense(self.actionSpace.shape[0], activation = 'relu')(h3)
        
        model = Model(input = stateInput, output = actionOutput)
        adam = Adam(lr = 0.001)
        model.compile(optimizer = adam, loss = 'mse')
        return stateInput, model
    
    def _create_critic_model(self):
        """
        Critic model corresponds to a Q-value function that maps from 
        (currentState, action) to Q-value.
        """
        stateInput = Input(shape = self.observationSpace.shape)
        stateH1 = Dense(48, activation = 'relu')(stateInput)
        stateH2 = Dense(64, activation = 'relu')(stateH1)
        
        actionInput = Input(shape = self.actionSpace.shape)
        actionH1 = Dense(48, activation = 'relu')(actionInput)
        actionH2 = Dense(64, activation = 'relu')(actionH1)
        
        mergedStateAction = Add()([stateH2, actionH2])
        mergedH1 = Dense(48, activation = 'relu')(mergedStateAction)
        mergedH2 = Dense(24, activation = 'relu')(mergedH1)
        # Since our reward is non-negative, we can use 'relu'. Otherwise, we need
        # to use 'linear'.
        valueOutput = Dense(1, activation = 'relu')(mergedH2)
        model = Model(input = [stateInput, actionInput], output = valueOutput)
        adam = Adam(lr = 0.001)
        model.compile(optimizer = adam, loss = 'mse')
        return stateInput, actionInput, model

    # ========================================================================= #
    #                      Perceive and Remember experiences                    #
    # ========================================================================= #
    def perceive_and_act(self, observation, reward, done = False):
        """
        Perceive observation and reward from environment after an interaction
        Input: observation, reward, done, info
        """
        self._observationNew = observation
        self._reward = reward
        self._done = done
        # Store experience: (observationOld, actionOld, observationNew, reward, done)
        self._remember(self._observationOld, self._actionOld, self._observationNew, self._reward, self._done)
        
        # Decide new action according to new observation
        self._actionNew = self._act(self._observationNew) # return action from actor model
        # Update temporary memory
        self._observationOld = self._observationNew
        self._actionOld = self._actionNew
        
        # Call training in a parallel process
        self.train()
        
        return self._actionNew
        
    def _remember(self, observationOld, actionOld, observationNew, reward, done):
        """
        Store (observationOld, action_Old, observationNew, reward, done)
        """
        self._memory.append([observationOld, actionOld, observationNew, reward, done])  
    
    # Other intrinsic motivation related remember work can be added here later
    
    # ========================================================================= #
    #                       Components or Model Training                        #
    # ========================================================================= #
    def _train(self):
        """
        Train all components or models in this function. To ensure responsive
        interaction, this function should run in parallel with action return.
        
        To do
        -----
        Let training of different model run in parallel.
        """
        batchSize = 32
        if len(self._memory) < batchSize:
            return
        
        samples = random.sample(self._memory, batchSize)
        
        # To speed up training the following trainig should be run in parallel
        # explicitly.
        self._train_critic_model(samples)
        self._train_actor_model(samples)
        
        # Other training should be added here later
        
        # Update target actor-critic model to newly trained model
        #   
        self._update_target_actor_critic_model()
        
    def _train_critic_model(self, samples):
        """
        Train critic model.
        
        Parameters
        ----------
        samples: list
        """
        for sample in samples:
            observationOld, actionOld, observationNew, reward, done = sample
            if not done:
                targetAction = self._targetActorModel.predict(observationNew)
                futureQValue = self._targetCriticModel.predict([observationNew, targetAction])
                targetQValue = reward + self._gamma * futureQValue
            
            self._criticModel.fit(x = [observationOld, actionOld],
                                  y = targetQValue,
                                  verbose  = 0)
    
    def _train_actor_model(self, samples):
        """
        Train actor model.
        
        Parameters
        ----------
        samples: list
        """
        for sample in samples:
            observationOld, actionOld, observationNew, reward, done = sample
            predictedAction = self._actorModel.predict(observationOld)
            # First get gradient of Q-Value with respect to action
            grads = self.sess.run(self._criticGrads, 
                                  feed_dict = {self._criticStateInput: observationOld,
                                               self._criticActionInput: predictedAction})[0]
            self.sess.run(self.optimize, feed_dict = {self._actorStateInput: observationOld,
                                                      self._actorCriticGrad: grads})
    
    # Other training method for other models should be added here later
    
    # ========================================================================= #
    #                         Target Model Updating                             #
    # ========================================================================= #    
    def _update_target_actor_model(self):
        """
        Update target actor model by copying newly trained actor model weights 
        to target actor model weights. This could help reducing unstable convergence.
        
        """
        actorModelWeights = self._actorModel.get_weights()
        targetActorModelWeights = self._targetActorModel.get_weights()
        
        for i in range(len(actorModelWeights)):
            targetActorModelWeights[i] = self._tau * actorModelWeights[i] + (1-self._tau) * targetActorModelWeights[i]
        
        self._targetActorModel.set_weights(targetActorModelWeights)
        
    def _update_target_critic_model(self):
        """
        Update target critic model by copying newly trained critic model weights
        to target critic model weights. This could help reducing unstable convergence.
        """
        criticModelWeights = self._criticModel.get_weights()
        targetCriticModelWeights = self._targetCriticModel.get_weights()
        
        for i in range(len(criticModelWeights)):
            targetCriticModelWeights[i] = self._tau * criticModelWeights[i] + (1-self._tau) * targetCriticModelWeights[i] 
        
        self._targetCriticModel.set_weights(targetCriticModelWeights)
       
    def _update_target_actor_critic_model(self):
        """
        Update target actor-cirtic model i.e. update both actor and critic model.
        """
        self._update_target_actor_model()
        self._update_target_critic_model()
        
    # ========================================================================= #
    #                              Model Predictions                            #
    # ========================================================================= #        
    def _act(self, observation):
        """
        Return action according to new observation.
        
        Parameters
        ----------
        observation: ndarray
            New observation
        
        Returns
        -------
        action: ndarray
            New action according to currently new observation
        """
        # You should call target actor model to predict new action, rather than
        # on training actor model for two reasons:
        #   1. target actor model could produce more stable action
        #   2. target actor model could run in parallel with on training actor model,
        #      so interactive action will not be stuck.
        
        # Reduce exploration rate gradually
        #   Note: this hyper-parameter mgiht need more tunning, since in our case
        #         the learning needs more exploration to get a reward.
        self._epsilon *= self._epsilonDecay
        
        if np.random.random() <= self._epsilon:
            return self.actionSpace.sample()
        
        return self._actorModel.predict(observation)
        
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
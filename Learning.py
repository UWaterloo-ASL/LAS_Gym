#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 23 00:37:13 2018

@author: jack.lingheng.meng
"""

#original file: Integration_Demo_for_ROM_Exhibit_new.py

import logging
logger = logging.getLogger(__name__)

from datetime import datetime, date
from threading import Timer
import os
import numpy as np
from gym import spaces
import time

from LASAgent.InternalEnvOfAgent import InternalEnvOfAgent
from LASAgent.InternalEnvOfCommunity import InternalEnvOfCommunity
from LASAgent.LASBaselineAgent import LASBaselineAgent

class Learning():
    def __init__(self, learning_system):
        """
        Note: make sure initialize Learning with these two functions.
        
        Args:
            learning_system: the object of Learning_System which has
                get_observation (function):
                take_action (function):
        """
        self.learning_system = learning_system

    def setup_learning(self):
        starttime = time.time()
        #######################################################################
        #                      Schedual thress experiments                    #
        # 1. Daiwei's Experiment: Agent controls parameterized actions
        # 2. Lingheng's Experiment 1: Single-Agent controls raw actions
        # 3. Lingheng's Experiment 2: Agent-Community contrls raw actions
        # 
        # Note:
        #   1. It takes some time to initialize learning agents, so call 
        #      setup_learning() before 9:45am everyday.
        #   2. Single_Agent.stop() will take about 3 minutes to save learned models.
        #   3. Agent_Community.stop() will take about 10 minutes to save learned models.
        # Problem:
        #   We don't want visitors to feel the pause when saving learned models
        # Solution: 
        #   To get rid of time-gap when switching behavior modes, use multiple threads
        #   to interact with visitors with different interaction modes i.e. when one 
        #   interaction mode is done, another interaction mode starts interacting 
        #   immediately at the same time the previous thread keeps saving learned models.
        #######################################################################
        # Initialize observation and action space
        self.instantiate_observation_and_action_space()
        # Initialize baseline agent which only collects data under Adam's default 
        # parameters.
        # self.instantiate_LAS_Baseline_Agent_parameterized_action()
        # Initialize Learning Agents
        self.instantiate_LAS_Agent_parameterized_action()
        self.instantiate_LAS_Agent_raw_action()
        # self.instantiate_LAS_Agent_Community_raw_action()
        # Schedule Experiments
        self.schedule_experiments()
        # Start Threads
        self.start_threads()
    
    def instantiate_observation_and_action_space(self):
        """
        Create observation and action space:
            observation space:
                24 IRs shared by all agents:
                    self.observation_space_name (string list): sensor names in observation space
                    self.observation_space (gym.spaces.Box): observation space shared by all agents
            action space:
                1. based on 17 parameters:
                    self.para_action_space_name (string list): parameter names in parameter action space
                    self.para_action_space (gym.spaces.Box): parameter action space
                2. based on raw actuators:
                    self.raw_action_space_name
                    self.raw_action_space
        """
        # Observation space shared by all agents
        self.observation_space_name = ['IR1-1','IR1-2','IR2-1','IR2-2','IR3-1', 'IR3-2',
                                  'IR4-1', 'IR4-2', 'IR5-1', 'IR5-2', 'IR6-1', 'IR6-2', 
                                  'IR7-1', 'IR7-2', 'IR8-1', 'IR8-2', 'IR9-1', 'IR9-2', 
                                  'IR10-1', 'IR10-2', 'IR11-1', 'IR11-2', 'IR12-1', 'IR12-2']
        sensors_dim = 24    # 24 IRs
        obs_max = np.array([1.]*sensors_dim)      
        obs_min = np.array([0.]*sensors_dim)
        self.observation_space = spaces.Box(obs_min, obs_max, dtype = np.float32)
        
        # Parameterized action space
        #   1.a) ramp up time: the time it takes for the actuator to fade to its maximum value
        #   1.b) hold time: the time it that the actuator holds at the maximum value
        #   1.c) ramp down time: the time it takes for the actuator to fade to 0
        #   1.d) maximum brightness
        #   2) the time gap between the moth starting to fade and the protocell starting to fade
        #   3) time between activation of each SMA arm on breathing pore
        #   4) time between activation of each breathing pore
        #   5a) minimum time to wait before activating background behaviour
        #   5b) maximum time to wait before activating background behaviour
        #   6a) time to wait before trying to pick an actuator
        #   6b) probability of successfully choosing an actuator
        #   7) time between choosing SMA to actuate
        #   8a) minimum time to wait before performing sweep
        #   8b) maximum time to wait before performing sweep
        self.para_action_space_name = ['1_a', '1_b', '1_c', '1_d', '2', '3', '4', 
                                       '5_a', '5_b', '6_a', '6_b', '7', '8_a', '8_b']
        para_actuators_dim = 17  # 17 Parameters  
        para_act_max = np.array([1]*para_actuators_dim)
        para_act_min = np.array([-1]*para_actuators_dim)
        self.para_action_space = spaces.Box(para_act_max, para_act_min, dtype = np.float32)
        
        # Raw actuator action space
        #   24 nodes, each of which has 6 SMAs, 1 Moth and 1 LED.
        self.raw_action_space_name = ['sma1_node#0', 'sma2_node#0', 'sma3_node#0', 'sma4_node#0',
                                      'sma5_node#0', 'sma6_node#0', 'sma1_node#1', 'sma2_node#1',
                                      'sma3_node#1', 'sma4_node#1', 'sma5_node#1', 'sma6_node#1',
                                      'sma1_node#2', 'sma2_node#2', 'sma3_node#2', 'sma4_node#2',
                                      'sma5_node#2', 'sma6_node#2', 'sma1_node#3', 'sma2_node#3',
                                      'sma3_node#3', 'sma4_node#3', 'sma5_node#3', 'sma6_node#3',
                                      'sma1_node#4', 'sma2_node#4', 'sma3_node#4', 'sma4_node#4',
                                      'sma5_node#4', 'sma6_node#4', 'sma1_node#5', 'sma2_node#5',
                                      'sma3_node#5', 'sma4_node#5', 'sma5_node#5', 'sma6_node#5',
                                      'sma1_node#6', 'sma2_node#6', 'sma3_node#6', 'sma4_node#6',
                                      'sma5_node#6', 'sma6_node#6', 'sma1_node#7', 'sma2_node#7',
                                      'sma3_node#7', 'sma4_node#7', 'sma5_node#7', 'sma6_node#7',
                                      'sma1_node#8', 'sma2_node#8', 'sma3_node#8', 'sma4_node#8',
                                      'sma5_node#8', 'sma6_node#8', 'sma1_node#9', 'sma2_node#9',
                                      'sma3_node#9', 'sma4_node#9', 'sma5_node#9', 'sma6_node#9',
                                      'sma1_node#10', 'sma2_node#10', 'sma3_node#10', 'sma4_node#10',
                                      'sma5_node#10', 'sma6_node#10', 'sma1_node#11', 'sma2_node#11',
                                      'sma3_node#11', 'sma4_node#11', 'sma5_node#11', 'sma6_node#11',
                                      'sma1_node#12', 'sma2_node#12', 'sma3_node#12', 'sma4_node#12',
                                      'sma5_node#12', 'sma6_node#12', 'sma1_node#13', 'sma2_node#13',
                                      'sma3_node#13', 'sma4_node#13', 'sma5_node#13', 'sma6_node#13',
                                      'sma1_node#14', 'sma2_node#14', 'sma3_node#14', 'sma4_node#14',
                                      'sma5_node#14', 'sma6_node#14', 'sma1_node#15', 'sma2_node#15',
                                      'sma3_node#15', 'sma4_node#15', 'sma5_node#15', 'sma6_node#15',
                                      'sma1_node#16', 'sma2_node#16', 'sma3_node#16', 'sma4_node#16',
                                      'sma5_node#16', 'sma6_node#16', 'sma1_node#17', 'sma2_node#17',
                                      'sma3_node#17', 'sma4_node#17', 'sma5_node#17', 'sma6_node#17',
                                      'sma1_node#18', 'sma2_node#18', 'sma3_node#18', 'sma4_node#18',
                                      'sma5_node#18', 'sma6_node#18', 'sma1_node#19', 'sma2_node#19',
                                      'sma3_node#19', 'sma4_node#19', 'sma5_node#19', 'sma6_node#19',
                                      'sma1_node#20', 'sma2_node#20', 'sma3_node#20', 'sma4_node#20',
                                      'sma5_node#20', 'sma6_node#20', 'sma1_node#21', 'sma2_node#21',
                                      'sma3_node#21', 'sma4_node#21', 'sma5_node#21', 'sma6_node#21',
                                      'sma1_node#22', 'sma2_node#22', 'sma3_node#22', 'sma4_node#22',
                                      'sma5_node#22', 'sma6_node#22', 'sma1_node', 'sma2_node',
                                      'sma3_node', 'sma4_node', 'sma5_node', 'sma6_node',
                                      'light_node#0', 'light_node#1', 'light_node#2', 'light_node#3',
                                      'light_node#4', 'light_node#5', 'light_node#6', 'light_node#7',
                                      'light_node#8', 'light_node#9', 'light_node#10', 'light_node#11',
                                      'light_node#12', 'light_node#13', 'light_node#14', 'light_node#15',
                                      'light_node#16', 'light_node#17', 'light_node#18', 'light_node#19',
                                      'light_node#20', 'light_node#21', 'light_node#22', 'light_node',
                                      'moth_node#0', 'moth_node#1', 'moth_node#2', 'moth_node#3',
                                      'moth_node#4', 'moth_node#5', 'moth_node#6', 'moth_node#7',
                                      'moth_node#8', 'moth_node#9', 'moth_node#10', 'moth_node#11',
                                      'moth_node#12', 'moth_node#13', 'moth_node#14', 'moth_node#15',
                                      'moth_node#16', 'moth_node#17', 'moth_node#18', 'moth_node#19',
                                      'moth_node#20', 'moth_node#21', 'moth_node#22', 'moth_node']
        raw_actuators_dim = (6+1+1)*24 # (1 moth + 1 LED + 6 SMAs) * 24 nodes
        raw_act_max = np.array([1]*raw_actuators_dim)
        raw_act_min = np.array([-1]*raw_actuators_dim)
        self.raw_action_space = spaces.Box(raw_act_max, raw_act_min, dtype = np.float32)

    def instantiate_LAS_Agent_parameterized_action(self):

        #######################################################################
        #                          Instatiate LAS-Agent                       #
        #######################################################################
        # Note: 1. Set load_pretrained_agent_flag to "True" only when you have 
        #           and want to load pretrained agent.
        #       2. Keep initializing parameters unchanged if using pretrained agent.
        #agent_name = 'LAS_Single_Agent_Parameterized_Action'
        agent_name = 'L'
        x_order_sensor_reading = 1
        x_order_sensor_reading_sliding_window = 5
        x_order_sensor_reading_preprocess_type = 'max_pool_sensory_readings'#'average_pool_sensory_readings'#'concatenate_sensory_readings'
        occupancy_reward_type = 'IR_distance'
        interaction_mode = 'real_interaction'
        load_pretrained_agent_flag = True
        
        # self.single_agent_parameterized_action = InternalEnvOfAgent(agent_name,
        #                                                             self.observation_space,
        #                                                             self.para_action_space,
        #                                                             self.observation_space_name,
        #                                                             self.para_action_space_name,
        #                                                             x_order_sensor_reading,
        #                                                             x_order_sensor_reading_sliding_window,
        #                                                             x_order_sensor_reading_preprocess_type,
        #                                                             occupancy_reward_type,
        #                                                             interaction_mode,
        #                                                             load_pretrained_agent_flag)

        self.single_agent_parameterized_action = LASBaselineAgent(agent_name,
                                                                    self.observation_space.shape[0],
                                                                    self.para_action_space.shape[0],
                                                                    num_observation=x_order_sensor_reading,
                                                                    load_pretrained_agent_flag = load_pretrained_agent_flag)

        logger.info('Instantiate {} done!'.format(agent_name))

    def instantiate_LAS_Baseline_Agent_parameterized_action(self):

        #######################################################################
        #                          Instatiate LAS-Agent                       #
        #######################################################################
        # Note: 1. Set load_pretrained_agent_flag to "True" only when you have 
        #           and want to load pretrained agent.
        #       2. Keep initializing parameters unchanged if using pretrained agent.
        #agent_name = 'LAS_Single_Agent_Parameterized_Action'
        agent_name = 'L_baseline'
        x_order_sensor_reading = 20
        x_order_sensor_reading_sliding_window = 5
        x_order_sensor_reading_preprocess_type = 'max_pool_sensory_readings'#'average_pool_sensory_readings'#'concatenate_sensory_readings'
        occupancy_reward_type = 'IR_distance'
        interaction_mode = 'real_interaction'
        load_pretrained_agent_flag = False
        baseline_agent_flag = True
        
        self.single_baseline_agent_parameterized_action = InternalEnvOfAgent(agent_name,
                                                                             self.observation_space, 
                                                                             self.para_action_space,
                                                                             self.observation_space_name, 
                                                                             self.para_action_space_name,
                                                                             x_order_sensor_reading,
                                                                             x_order_sensor_reading_sliding_window,
                                                                             x_order_sensor_reading_preprocess_type,
                                                                             occupancy_reward_type,
                                                                             interaction_mode,
                                                                             load_pretrained_agent_flag,
                                                                             baseline_agent_flag)
        logger.info('Instantiate {} done!'.format(agent_name))

    def instantiate_LAS_Agent_raw_action(self):

        #######################################################################
        #                          Instatiate LAS-Agent                       #
        #######################################################################
        # Note: 1. Set load_pretrained_agent_flag to "True" only when you have 
        #           and want to load pretrained agent.
        #       2. Keep initializing parameters unchanged if using pretrained agent.
        agent_name = 'LAS_Single_Agent_Raw_Action'
        x_order_sensor_reading = 20
        x_order_sensor_reading_sliding_window = 5
        x_order_sensor_reading_preprocess_type = 'max_pool_sensory_readings'#'average_pool_sensory_readings'#'concatenate_sensory_readings'
        occupancy_reward_type = 'IR_distance'
        interaction_mode = 'real_interaction'
        load_pretrained_agent_flag = False
        
        self.single_agent_raw_action = InternalEnvOfAgent(agent_name,
                                                          self.observation_space, 
                                                          self.raw_action_space,
                                                          self.observation_space_name, 
                                                          self.raw_action_space_name,
                                                          x_order_sensor_reading,
                                                          x_order_sensor_reading_sliding_window,
                                                          x_order_sensor_reading_preprocess_type,
                                                          occupancy_reward_type,
                                                          interaction_mode,
                                                          load_pretrained_agent_flag)
        logger.info('Instantiate {} done!'.format(agent_name))

    def instantiate_LAS_Agent_Community_raw_action(self):
        
        #######################################################################
        #                    Instatiate LAS-Agent-Community                   #
        #######################################################################
        # Note: 1. Set load_pretrained_agent_flag to "True" only when you have and want 
        #          to load pretrained agent.
        #       2. Keep initializing parameters unchanged if using pretrained agent.
        community_name = 'LAS_Agent_Community_raw_action'
        community_size = 3
        x_order_sensor_reading = 20
        x_order_sensor_reading_sliding_window = 5
        x_order_sensor_reading_preprocess_type = 'max_pool_sensory_readings'#'average_pool_sensory_readings'#'concatenate_sensory_readings'
        occupancy_reward_type = 'IR_distance'
        interaction_mode = 'real_interaction'
        load_pretrained_agent_flag = False
        
        self.LAS_agent_community_raw_action = InternalEnvOfCommunity(community_name, 
                                                                     community_size,
                                                                     self.observation_space,
                                                                     self.raw_action_space, 
                                                                     self.observation_space_name,
                                                                     self.raw_action_space_name,
                                                                     x_order_sensor_reading,
                                                                     x_order_sensor_reading_sliding_window,
                                                                     x_order_sensor_reading_preprocess_type,
                                                                     occupancy_reward_type,
                                                                     interaction_mode,
                                                                     load_pretrained_agent_flag)
        logger.info('Instantiate {} done!'.format(community_name))

    def interact_with_learning_agent(self, agent, end_time):
        print("LEARNING-------------------------------------------------")
        """
        Note:
            self.get_observation() and self.take_action(action) are functions
            passed into when initializing the object.
            
        Args:
            agent (learning agent object)
            end_time (str): (in format %HH%MM%SS) the end time of interaction
        """
        logger.info('{}: Start interaction. Default End_time: {}'.format(agent.name, end_time))
        # Interact untill end_time
        while not datetime.now().strftime("%H%M%S") > end_time:
            new_observation_flag, observation = self.learning_system.get_observation()
            if new_observation_flag:
                take_action_flag, action = agent.feed_observation(observation)
                if take_action_flag == True:
                    self.learning_system.take_action(action)
        # Save learned model
        logger.info('{}: Interaction is done. Saving learned models...'.format(agent.name))
        agent.stop()
        logger.info('{}: Saving learned models done.'.format(agent.name))
    

    def interact_with_prescribed_behavior(self, agent, end_time):
        """
        TODO: Please put prescribed behavior in this function.
        
        Args:
            agent (learning agent object):not important paramter just for keeping the same format with interact_with_learning_agent
            end_time (str):(in format %HH%MM%SS) the end time of interaction
        """
        logger.info('{}: Start interaction. Default End_time: {}'.format(agent, end_time))
        # Interact untill end_time
        self.learning_system.reset()
        # Note: After reset still need to get_observation(), because we are only 
        #       allowed to collect data from 1pm to 4pm. 
        #       Thus, Adam doesn't need to save any data for us.
        while not datetime.now().strftime("%H%M%S") > end_time:
            new_observation_flag, observation = self.learning_system.get_observation()
            if new_observation_flag:
                # Note: only collect data
                take_action_flag, action = agent.feed_observation(observation)
                    
        logger.info('{}: Interaction is done.'.format(agent.name))

    def interaction_mode_scheduler(self, interaction_mode, agent, 
                                   start_time, end_time, schedule_start_time):
        """
        Args:
            interaction_mode (func): function name
            agent: depends on interaction mode
                1. agent object: for learning agent
                2. 'priscribed_behavior': for priscribed behavior
            start_time (str):(with format'hhmmss')
            end_time (str): (with format'hhmmss')
            schedule_start_time (datetime object): 
            
        Returns:
            interaction_thread: a delayed thread for an interaction mode which will start at a given time.
        """
        start_delay = (datetime.strptime(date.today().strftime("%Y%m%d")+'-'+start_time, '%Y%m%d-%H%M%S') - schedule_start_time).total_seconds()
        if start_delay < 0:
            logger.error('{} starts earlier than schedualing time!'.format(interaction_mode.__name__))

        interaction_thread = Timer(interval = start_delay,
                                   function = interaction_mode,
                                        kwargs={'agent': agent,
                                                'end_time': end_time})
        return interaction_thread

    def schedule_experiments(self):
        # Get current time to calculate interaction start-time-delay
        schedule_start_time = datetime.now()

        # # Schedule prescribed-behavior 1
        # # TODO: set start and end times to '093000' and '130000'
        # self.prescribed_behavior_start_time_1 = '093001' # format: %H%M%S e.g. 10:00am is 100000
        # self.prescribed_behavior_end_time_1 = '130000'   # format: %H%M%S e.g. 1:00pm is 130000
        # self.prescribed_behavior_thread_1 = self.interaction_mode_scheduler(self.interact_with_prescribed_behavior,
        #                                                                     self.single_baseline_agent_parameterized_action,
        #                                                                     self.prescribed_behavior_start_time_1,
        #                                                                     self.prescribed_behavior_end_time_1,
        #                                                                     schedule_start_time)
        
        # Schedule first experiment:
        #     Daiwei's Experiment: Agent controls parameterized actions
        # TODO: set start and end times to '130002' and '140000'
        self.first_experiment_start_time = '000702'  # format: %H%M%S e.g. 1:00pm is 130000
        self.first_experiment_end_time = '000900'    # format: %H%M%S e.g. 2:30pm is 143000
        self.first_experiment_thread = self.interaction_mode_scheduler(self.interact_with_learning_agent,
                                                                       self.single_agent_parameterized_action,
                                                                       self.first_experiment_start_time, 
                                                                       self.first_experiment_end_time, 
                                                                       schedule_start_time)
        # Schedule second experiment:
        #     Lingheng's Experiment 1: Single-Agent controls raw actions
        # TODO: set start and end times to '140002' and '150000'
        self.second_experiment_start_time = '212801' # format: %H%M%S e.g. 2:30pm is 143000
        self.second_experiment_end_time = '214800'   # format: %H%M%S e.g. 4:00pm is 160000
        self.second_experiment_thread = self.interaction_mode_scheduler(self.interact_with_learning_agent,
                                                                        self.single_agent_raw_action,
                                                                        self.second_experiment_start_time,
                                                                        self.second_experiment_end_time,
                                                                        schedule_start_time)
        # # Schedule third experiment:
        # #     Lingheng's Experiment 2: Agent-Community contrls raw actions
        # # TODO: set start and end times to '150002' and '160000'
        # self.third_experiment_start_time = '212801' # format: %H%M%S e.g. 2:30pm is 143000
        # self.third_experiment_end_time = '214800'   # format: %H%M%S e.g. 4:00pm is 160000
        # self.third_experiment_thread = self.interaction_mode_scheduler(self.interact_with_learning_agent,
        #                                                                 self.LAS_agent_community_raw_action,
        #                                                                 self.third_experiment_start_time,
        #                                                                 self.third_experiment_end_time,
        #                                                                 schedule_start_time)
        # # Schedule baseline:
        # #   Baseline experiment: only collect data with Adam's default parameters
        # # TODO: set start and end time to '' and ''
        # self.baseline_experiment_start_time = '150002'
        # self.baseline_experiment_end_time = '160000'
        # self.baseline_experiment_thread = self.interaction_mode_scheduler(self.interact_with_prescribed_behavior,
        #                                                                   self.single_baseline_agent_parameterized_action,
        #                                                                   self.baseline_experiment_start_time,
        #                                                                   self.baseline_experiment_end_time,
        #                                                                   schedule_start_time)
        # Schedule prescribed-behavior 2
        # # TODO: set start and end times to '160002' and '173000'
        # self.prescribed_behavior_start_time_2 = '160002' # format: %H%M%S e.g. 4:00pm is 160000
        # self.prescribed_behavior_end_time_2 = '173000'   # format: %H%M%S e.g. 5:30pm is 173000
        # self.prescribed_behavior_thread_2 = self.interaction_mode_scheduler(self.interact_with_prescribed_behavior,
        #                                                                     self.single_baseline_agent_parameterized_action,
        #                                                                     self.prescribed_behavior_start_time_2,
        #                                                                     self.prescribed_behavior_end_time_2,
        #                                                                     schedule_start_time)

    def start_threads(self):
        # Schedule interaction with learning agent
        self.first_experiment_thread.start()
        logger.info('first_experiment_thread scheduled: {}-{}'.format(self.first_experiment_start_time, self.first_experiment_end_time))
#        self.second_experiment_thread.start()
#        logger.info('second_experiment_thread scheduled: {}-{}'.format(self.second_experiment_start_time, self.second_experiment_end_time))
#        self.third_experiment_thread.start()
#        logger.info('second_experiment_thread scheduled: {}-{}'.format(self.third_experiment_start_time, self.third_experiment_end_time))
#        # Schedule interaction with presribed-behavior
#        self.prescribed_behavior_thread_1.start()
#        logger.info('prescribed_behavior_thread_1 scheduled: {}-{}'.format(self.prescribed_behavior_start_time_1, self.prescribed_behavior_end_time_1))
#        self.prescribed_behavior_thread_2.start()
#        logger.info('prescribed_behavior_thread_2 scheduled: {}-{}'.format(self.prescribed_behavior_start_time_2, self.prescribed_behavior_end_time_2))
#         self.baseline_experiment_thread.start()
#         logger.info('baseline_experiment_thread scheduled: {}-{}'.format(self.baseline_experiment_start_time, self.baseline_experiment_end_time))

    def check_if_interactions_done(self):
        if not self.first_experiment_thread.is_alive():
            # and not self.second_experiment_thread.is_alive()\
            # and not self.third_experiment_thread.is_alive()\
            # and not self.baseline_experiment_thread.is_alive()\
            # and not self.prescribed_behavior_thread_1.is_alive()\
            # and not self.prescribed_behavior_thread_2.is_alive():
            logger.info('All interactions are done.')
            return True 
        else:
            return False
        
if __name__ == '__main__':
    """
    TODO: make sure initialize "Learning" with instance of Learning_System.
    Args:
        learning_system (object of Learning_System)
    """
    learning_system = Learning_System()
    learning = Learning(learning_system)
    
    learning.setup_learning()
    
    # TODO: put initialization work for master script in here
    # Check if all interactions are done.
    while True:
        if learning.check_if_interactions_done():
           break
  

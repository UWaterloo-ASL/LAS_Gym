#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  9 09:20:12 2018

@author: jack.lingheng.meng
"""
###############################################################################
#                   TODO: add this part in master_script                      #
###############################################################################
import os
import logging
from datetime import datetime
experiment_results_dir = os.path.join(os.path.abspath('..'), 'ROM_Experiment_results')
if not os.path.exists(experiment_results_dir):
    os.makedirs(experiment_results_dir)
logging.basicConfig(filename = os.path.join(experiment_results_dir,'ROM_experiment_'+datetime.now().strftime("%Y%m%d_%H%M%S")+'.log'), 
                    level = logging.DEBUG,
                    format='%(asctime)s:%(name)s:%(levelname)s: %(message)s')
###############################################################################

from Environment.LASEnv import LASEnv
from Learning import Learning

class LearningSystem(object):
    def __init__(self):
        self.env = LASEnv('127.0.0.1', 19997, reward_function_type = 'occupancy')
        
    def get_observation(self):
        return True, self.env._self_observe()
    
    def take_action(self, action):
        self.env.step(action)
        
    def reset(self):
        self.env.reset()

learning_system = LearningSystem()

learner = Learning(learning_system)
learner.setup_learning()
        
## Random agent
#for i in range(100):
#    new_observation_flag, observation = learning_system.get_observation()
#    if new_observation_flag:
#        learning_system.take_action(learning_system.env.action_space.sample())
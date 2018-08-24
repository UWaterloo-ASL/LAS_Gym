#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 23 10:13:03 2018

@author: jack.lingheng.meng
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 23 00:37:13 2018

@author: jack.lingheng.meng
"""
import logging
import tensorflow as tf
import numpy as np
import time

from datetime import datetime, date
from threading import Timer

import os

import sched, time

from Environment.LASEnv import LASEnv
from LASAgent.InternalEnvOfAgent import InternalEnvOfAgent
from LASAgent.InternalEnvOfCommunity import InternalEnvOfCommunity

# Logging
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.basicConfig(filename = '../ROM_Experiment_results/ROM_experiment_'+datetime.now().strftime("%Y%m%d-%H%M%S")+'.log', 
                    level = logging.INFO,
                    format='%(asctime)s:%(levelname)s: %(message)s')

#######################################################################
#                      Schedual two experiments                       #
#######################################################################
env = 'envLAS'
def get_observation():
    print('get_observation')
def interact_with_learning_agent(agent, end_time = '143000'):
    try:
        logging.info('Run {}, Start_time: {}, End_time: {}'.format(agent, datetime.now().strftime("%H%M%S"), end_time))
        # Interact untill end_time or interrupted by 'Ctrl+c'
        while not datetime.now().strftime("%H%M%S") > end_time:
            get_observation()
        get_observation()
        logging.info('agent.stop()')
        logging.info('{}, Actual_End_time: {}'.format(agent, datetime.now().strftime("%H%M%S")))
    except KeyboardInterrupt:
        # Save learned model
        logging.info('agent.stop()')
        logging.info('{}, Actual_End_time: {}'.format(agent, datetime.now().strftime("%H%M%S")))

scheduler = sched.scheduler(time.time, time.sleep)
# Assume ROM open at 10am and this script is run at 10am
open_time = datetime.now()#10
# Schedule first experiment
first_experiment_start_time = '005400'  # 1:00pm
first_experiment_end_time = '005410'    # 2:30pm
first_experiment_start_delay = (datetime.strptime(date.today().strftime("%Y%m%d")+'-'+first_experiment_start_time, '%Y%m%d-%H%M%S') - open_time).total_seconds()
if first_experiment_start_delay < 0:
    logging.error('Start time earlier than ROM open-time!')
    
first_experiment_thread = Timer(interval = first_experiment_start_delay,
                                function = interact_with_learning_agent,
                                kwargs={'agent': 'single_agent',
#                                        'env': 'envLAS',
                                        'end_time': first_experiment_end_time})

#scheduler.enter(first_experiment_start_delay, 
#                1, 
#                first_experiment_thread.start())
# Schedule second experiment
second_experiment_start_time = '005410' # 2:30pm
second_experiment_end_time = '005420'   # 4:00pm
second_experiment_start_delay = (datetime.strptime(date.today().strftime("%Y%m%d")+'-'+second_experiment_start_time, '%Y%m%d-%H%M%S') - open_time).total_seconds()
if second_experiment_start_delay < 0:
    logging.error('Start time earlier than ROM open-time!')

second_experiment_thread = Timer(interval = second_experiment_start_delay,
                                 function = interact_with_learning_agent,
                                 kwargs={'agent': 'LAS_agent_community',
#                                         'env': 'envLAS',
                                         'end_time': second_experiment_end_time})

#scheduler.enter(second_experiment_start_delay, 1, 
#                second_experiment_thread.start())


    
if __name__ == '__main__':
    
    # Run two experiments
    logging.info('Run scheduler...')
#    scheduler.run()
    first_experiment_thread.start()
    print(first_experiment_thread.is_alive())
    second_experiment_thread.start()
    print(first_experiment_thread.is_alive())
    
    
    
    
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 11 15:12:53 2018

@author: jack.lingheng.meng
"""

import matplotlib.pyplot as plt
import time
import random
from collections import deque
import numpy as np

def plot_cumulative_reward(cumulativeReward):
    line, = plt.plot(cumulativeReward)
    plt.ion()
    #plt.ylim([0,10])
    plt.show()
    plt.pause(0.0001)

#ax = plt.axes(xlim=(0, 20), ylim=(0, 1))

cumulativeReward = Ext_Mot_LASAgent._cumulativeRewardMemory
plot_cumulative_reward(Ext_Mot_LASAgent._cumulativeRewardMemory)


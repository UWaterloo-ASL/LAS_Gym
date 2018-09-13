""" 
Data structure for implementing experience replay

Author: Patrick Emami
"""
from collections import deque
import random
import numpy as np

class ReplayBuffer(object):

    def __init__(self, buffer_size, random_seed=123):
        """
        The right side of the deque contains the most recent experiences 
        """
        self.buffer_size = buffer_size
        self.buffer = deque(maxlen = buffer_size)
        random.seed(random_seed)

    def add(self, s, a, r, t, s2):
        experience = (s, a, r, t, s2)
        self.buffer.append(experience)

    def size(self):
        return len(self.buffer)

    def sample_batch(self, batch_size):
        batch = []
        if len(self.buffer) < batch_size:
            batch = random.sample(self.buffer, len(self.buffer))
        else:
            batch = random.sample(self.buffer, batch_size)

        s_old_batch = np.array([sample[0] for sample in batch])
        a_old_batch = np.array([sample[1] for sample in batch])
        r_old_batch = np.array([sample[2] for sample in batch])
        done_batch = np.array([sample[3] for sample in batch])
        s_new_batch = np.array([sample[4] for sample in batch])

        return s_old_batch, a_old_batch, r_old_batch, done_batch, s_new_batch

    def clear(self):
        self.buffer.clear()



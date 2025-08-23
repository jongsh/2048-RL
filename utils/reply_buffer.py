import random
import collections
import numpy as np


class ReplayBuffer:
    """Replay buffer for storing transitions in reinforcement learning"""

    def __init__(self, capacity, min_capacity=0):
        self.capacity = capacity
        self.min_capacity = min_capacity
        self.buffer = collections.deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def clear(self):
        self.buffer.clear()

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        if len(self.buffer) < self.min_capacity:
            return None
        transitions = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*transitions)
        return (
            np.array(states, dtype=np.int32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.int32),
            np.array(dones, dtype=np.int32),
        )

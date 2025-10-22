import random
import collections
import numpy as np


class ReplayBuffer:
    """Replay buffer for storing transitions in reinforcement learning"""

    @classmethod
    def from_data_list(cls, data_list):
        length = len(data_list)
        buffer = cls(capacity=length)
        for transition in data_list:
            buffer.add(
                transition["state"],
                transition["action"],
                transition["reward"],
                transition["next_state"],
                transition["done"],
                transition["action_mask"],
            )
        return buffer

    def __init__(self, capacity, min_capacity=0):
        self.capacity = capacity
        self.min_capacity = min_capacity
        self.buffer = collections.deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def clear(self):
        self.buffer.clear()

    def add(self, state, action, reward, next_state, done, action_mask):
        self.buffer.append((state, action, reward, next_state, done, action_mask))

    def sample(self, batch_size):
        if len(self.buffer) < self.min_capacity:
            return None
        transitions = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones, action_mask = zip(*transitions)
        return (
            np.array(states, dtype=np.int32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.int32),
            np.array(dones, dtype=np.int32),
            np.array(action_mask, dtype=np.int32),
        )

    def save(self, dir_path):
        np.savez_compressed(dir_path / "replay_buffer.npz", data=list(self.buffer))

    def load(self, dir_path):
        data = np.load(dir_path / "replay_buffer.npz", allow_pickle=True)["data"]
        self.buffer = collections.deque(data.tolist(), maxlen=self.capacity)

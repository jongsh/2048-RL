import random
import collections
import numpy as np


class SumTree:
    """SumTree data structure for prioritized experience replay"""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)  # Tree structure to store priorities
        self.data = np.zeros(capacity, dtype=object)  # Array to store transitions
        self.size = 0  # Number of transitions currently stored
        self.write = 0  # Pointer to the store position in the data array

    @property
    def total_priority(self):
        return self.tree[0]

    def clear(self):
        self.tree = np.zeros(2 * self.capacity - 1)
        self.data = np.zeros(self.capacity, dtype=object)
        self.size = 0
        self.write = 0

    def _propagate(self, idx: int, change: float):
        # Calculate the parent index and update the tree value
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx: int, score: float):
        # Use the score to traverse the tree and find the corresponding leaf index
        left, right = 2 * idx + 1, 2 * idx + 2
        if left >= len(self.tree):
            return idx
        if score <= self.tree[left]:
            return self._retrieve(left, score)
        else:
            return self._retrieve(right, score - self.tree[left])

    def update(self, idx: int, priority: float):
        # Update the priority of a transition and propagate the change up the tree
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)

    def add(self, priority: float, data: object):
        # Add a new transition with the given priority to the tree and data array
        tree_idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(tree_idx, priority)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0
        if self.size < self.capacity:
            self.size += 1

    def get(self, score: float):
        # Retrieve a transition based on the given score by traversing the tree
        tree_idx = self._retrieve(0, score)
        data_idx = tree_idx - self.capacity + 1
        return self.tree[tree_idx], self.data[data_idx], tree_idx


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


class PrioritizedReplayBuffer:
    """
    Prioritized replay buffer for storing transitions with priorities.
    This implementation uses a SumTree to efficiently sample transitions based on their priorities.
    """

    @classmethod
    def from_data_list(cls, capacity, min_capacity, alpha, data_list):
        # create a PrioritizedReplayBuffer from a list of transitions.
        buffer = cls(capacity=capacity, min_capacity=min_capacity, alpha=alpha)
        for transition in data_list:
            buffer.add(
                transition["state"],
                transition["action"],
                transition["reward"],
                transition["next_state"],
                transition["done"],
                transition["action_mask"],
            )

    def __init__(self, capacity, min_capacity=0, alpha=0.6):
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.min_capacity = min_capacity
        self.alpha = alpha  # priority exponent
        self.epsilon = 0.01  # small constant to avoid zero priority
        self.max_priority = 1.0  # initial max priority

    def __len__(self):
        return self.tree.size

    def clear(self):
        self.tree.clear()

    def add(self, state, action, reward, next_state, done, action_mask):
        transition = (state, action, reward, next_state, done, action_mask)
        self.tree.add(self.max_priority, transition)

    def sample(self, batch_size, beta=0.4):
        # sample a batch of transitions based on their priorities and calculate importance-sampling weights
        if len(self) < self.min_capacity:
            return None

        batch, priorities, idxs = [], [], []

        # divide the total priority into equal segments and sample one transition from each segment
        segment = self.tree.total_priority() / batch_size
        for i in range(batch_size):
            left, right = segment * i, segment * (i + 1)
            left, right = max(left, 0), min(right, self.tree.total_priority)
            score = random.uniform(left, right)
            priority, data, idx = self.tree.get(score)
            idxs.append(idx)
            priorities.append(priority)
            batch.append(data)

        # calculate importance-sampling weights
        sampling_probabilities = np.array(priorities) / self.tree.total_priority
        is_weights = np.power(self.tree.size * sampling_probabilities, -beta)
        is_weights /= is_weights.max()  # normalize for stability

        states, actions, rewards, next_states, dones, action_masks = zip(*batch)
        return (
            np.array(states, dtype=np.int32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.int32),
            np.array(dones, dtype=np.int32),
            np.array(action_masks, dtype=np.int32),
            np.array(is_weights, dtype=np.float32),
            np.array(idxs, dtype=np.int64),
        )

    def update_priorities(self, idxs, td_errors):
        # Update the priorities of the sampled transitions based on their TD errors
        for idx, td_error in zip(idxs, td_errors):
            priority = (abs(td_error) + self.epsilon) ** self.alpha
            self.tree.update(idx, priority)
            self.max_priority = max(self.max_priority, priority)

    def save(self, dir_path):
        # Save the replay buffer to disk
        np.savez_compressed(
            dir_path / "prioritized_replay_buffer.npz",
            tree=self.tree.tree,
            data=self.tree.data,
            size=self.tree.size,
            write=self.tree.write,
        )

    def load(self, dir_path):
        # Load the replay buffer from disk
        data = np.load(dir_path / "prioritized_replay_buffer.npz", allow_pickle=True)
        assert data["data"].shape[0] <= self.capacity, "Loaded data exceeds buffer capacity!"
        self.tree.tree = data["tree"]
        self.tree.data = data["data"]
        self.tree.size = data["size"]
        self.tree.write = data["write"]

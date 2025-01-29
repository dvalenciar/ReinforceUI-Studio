import numpy as np


class MemoryBuffer:
    def __init__(self, observation_size, action_num, hyperparameters):
        self.max_size = int(hyperparameters.get("buffer_size"))
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((self.max_size, observation_size))
        self.action = np.zeros((self.max_size, action_num))
        self.next_state = np.zeros((self.max_size, observation_size))
        self.reward = np.zeros((self.max_size,))
        self.done = np.zeros((self.max_size,))

    def add_experience(self, state, action, reward, next_state, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.done[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_experience(self, batch_size):
        batch_size = min(batch_size, self.size)

        # Sample indices for the batch
        ind = np.random.randint(0, self.size, size=batch_size)
        return (
            self.state[ind],
            self.action[ind],
            self.reward[ind],
            self.next_state[ind],
            self.done[ind],
        )

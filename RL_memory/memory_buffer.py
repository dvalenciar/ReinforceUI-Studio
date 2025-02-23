import numpy as np


class MemoryBuffer:
    def __init__(self, observation_size, action_num, hyperparameters):
        self.max_size = int(hyperparameters.get("buffer_size"))
        self.ptr = 0
        self.size = 0
        self.state = np.zeros((self.max_size, observation_size))
        self.action = np.zeros((self.max_size, action_num))
        self.reward = np.zeros((self.max_size,))
        self.next_state = np.zeros((self.max_size, observation_size))
        self.done = np.zeros((self.max_size,))
        # log_prob need for ppo
        self.log_prob = np.zeros((self.max_size,))

    def add_experience(self, state, action, reward, next_state, done, log_prob=None):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.reward[self.ptr] = reward
        self.next_state[self.ptr] = next_state
        self.done[self.ptr] = done
        if log_prob is not None:
            self.log_prob[self.ptr] = log_prob
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)


    def sample_experience(self, batch_size):
        batch_size = min(batch_size, self.size)
        # Sample indices for the batch
        # fixme use np.random.choice instead of np.random.randint  check if this is correct
        ind = np.random.randint(0, self.size, size=batch_size)
        return (
            self.state[ind],
            self.action[ind],
            self.reward[ind],
            self.next_state[ind],
            self.done[ind],
        )

    def return_flushed_memory(self):
        """
        Flushes the memory buffers and returns the experiences in order.
        This is similar to the sample_experience method, but it returns all the experiences in order
        and empties the memory buffer.
        This is particularly used on PPO to train the policy

        Returns:
            experiences : The full memory buffer in order.
        """
        experiences = (
            self.state[: self.size],
            self.action[: self.size],
            self.reward[: self.size],
            self.next_state[: self.size],
            self.done[: self.size],
            self.log_prob[: self.size],
        )

        # Reset pointers
        self.ptr = 0
        self.size = 0
       # reset buffers without reallocating memory

        self.state = np.zeros((self.max_size, self.state.shape[1]))
        self.action = np.zeros((self.max_size, self.action.shape[1]))
        self.reward = np.zeros((self.max_size,))
        self.next_state = np.zeros((self.max_size, self.next_state.shape[1]))
        self.done = np.zeros((self.max_size,))
        self.log_prob = np.zeros((self.max_size,))

        return experiences

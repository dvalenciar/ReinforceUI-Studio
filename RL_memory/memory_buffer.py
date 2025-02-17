import numpy as np


class MemoryBuffer:
    def __init__(self, observation_size, action_num, hyperparameters):
        self.max_size = int(hyperparameters.get("buffer_size"))
        self.ptr = 0
        self.size = 0
        # fixme - check if the following is correct use np.empty instead of np.zeros
        self.state = np.zeros((self.max_size, observation_size))
        self.action = np.zeros((self.max_size, action_num))
        self.next_state = np.zeros((self.max_size, observation_size))
        self.reward = np.zeros((self.max_size,))
        self.done = np.zeros((self.max_size,))
        # log need for ppo
        self.log_prob = np.zeros((self.max_size,))

    def add_experience(self, state, action, reward, next_state, done, log_prob=None):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.done[self.ptr] = done
        if log_prob is not None:
            self.log_prob[self.ptr] = log_prob

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_experience(self, batch_size):
        batch_size = min(batch_size, self.size)
        # Sample indices for the batch
        #fixme use np.random.choice instead of np.random.randint  check if this is correct
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
        This is particulary used on PPO to train the policy

        Returns:
            experiences (dict): The full memory buffer in order.
        """
        experiences = {
            "state": self.state[: self.size],
            "action": self.action[: self.size],
            "reward": self.reward[: self.size],
            "next_state": self.next_state[: self.size],
            "done": self.done[: self.size],
            "log_prob": self.log_prob[: self.size],
        }

        # fixme me changes this to np.empty instead of np.zeros if above was changed
        self.state.fill(0)
        self.action.fill(0)
        self.next_state.fill(0)
        self.reward.fill(0)
        self.done.fill(0)
        self.log_prob.fill(0)
        self.ptr = 0
        self.size = 0

        return {key: np.array(value) for key, value in experiences.items()}


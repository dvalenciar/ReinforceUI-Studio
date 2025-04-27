import numpy as np
from typing import Tuple, Optional, Union


class MemoryBuffer:
    def __init__(
        self,
        observation_size: int,
        action_num: int,
        hyperparameters: dict,
        algorithm_name: str,
    ) -> None:
        """Initialize the memory buffer with appropriate storage.

        Args:
            observation_size: Dimension of state representation
            action_num: Dimension of action space
            hyperparameters: Algorithm configuration parameters
            algorithm_name: Name of RL algorithm using this buffer
        """
        self.ptr = 0
        self.size = 0

        # Set buffer size based on algorithm
        if algorithm_name == "PPO":
            self.max_size = int(hyperparameters.get("max_steps_per_batch"))
            self.log_prob = np.zeros((self.max_size,), dtype=np.float32)
        else:
            self.max_size = int(hyperparameters.get("buffer_size"))

        # allocate memory for the buffer.
        self.state = np.zeros((self.max_size, observation_size), dtype=np.float32)

        if algorithm_name == "DQN":
            self.action = np.zeros((self.max_size,), dtype=np.int32)
        else:
            self.action = np.zeros((self.max_size, action_num), dtype=np.float32)
        self.reward = np.zeros((self.max_size,), dtype=np.float32)
        self.next_state = np.zeros((self.max_size, observation_size), dtype=np.float32)
        self.done = np.zeros((self.max_size,), dtype=np.bool_)

    def add_experience(
        self,
        state: np.ndarray,
        action: Union[np.ndarray, int],
        reward: float,
        next_state: np.ndarray,
        done: bool,
        log_prob: Optional[float] = None,
    ) -> None:
        """Store a single transition in the buffer.

        Args:
            state: Current state observation
            action: Action taken
            reward: Reward received
            next_state: Next state observation
            done: Whether episode terminated
            log_prob: Action log probability (for PPO)
        """
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.reward[self.ptr] = reward
        self.next_state[self.ptr] = next_state
        self.done[self.ptr] = done
        if log_prob is not None:
            self.log_prob[self.ptr] = log_prob

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_experience(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        """Sample a random batch of experiences.

        Args:
            batch_size: Number of transitions to sample

        Returns:
            Tuple containing batches of (states, actions, rewards, next_states, dones)
        """
        batch_size = min(batch_size, self.size)
        # Sample indices for the batch without replacement
        ind = np.random.choice(self.size, size=batch_size, replace=False)
        return (
            self.state[ind],
            self.action[ind],
            self.reward[ind],
            self.next_state[ind],
            self.done[ind],
        )

    def return_flushed_memory(self) -> Tuple[np.ndarray, ...]:
        """Return all stored experiences in order and reset buffer.

        Used primarily for on-policy algorithms like PPO.

        Returns:
            Tuple containing all experiences in buffer as
            (states, actions, rewards, next_states, dones, log_probs)
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

        return experiences

from typing import Any

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.wrappers import RescaleAction


class GymEnvironment:
    """Environment wrapper for Gymnasium environments.

    This class provides an interface to interact with environments from the
    Gymnasium library, including methods for resetting the environment,
    taking steps, rendering frames, and sampling actions.

    """

    def __init__(
        self, env_name: str, seed: int, render_mode: str = "rgb_array"
    ) -> None:
        """Initialize the GymEnvironment.

        Args:
            env_name: The name of the Gymnasium environment.
            seed: The random seed for the environment.
            render_mode: The mode for rendering frames. Defaults to "rgb_array".
        """
        self.env = gym.make(env_name, render_mode=render_mode)
        if not isinstance(self.env.action_space, spaces.Discrete):
            self.env = RescaleAction(self.env, min_action=-1, max_action=1)
        _, _ = self.env.reset(seed=seed)
        self.env.action_space.seed(seed)

    def max_action_value(self) -> float:
        """Get the maximum action value.

        Returns:
            float: The maximum value for actions.
        """
        return self.env.action_space.high[0]

    def min_action_value(self) -> float:
        """Get the minimum action value.

        Returns:
            float: The minimum value for actions.
        """
        return self.env.action_space.low[0]

    def observation_space(self) -> int:
        """Get the size of the observation space.

        Returns:
            int: The size of the observation space.
        """
        return self.env.observation_space.shape[0]

    def action_num(self) -> Any:
        """Get the number of actions.

        Returns:
            int: The number of actions.
        """
        if isinstance(self.env.action_space, spaces.Box):
            return self.env.action_space.shape[0]
        elif isinstance(self.env.action_space, spaces.Discrete):
            return self.env.action_space.n

    def sample_action(self) -> int:
        """Sample a random action.

        Returns:
            int: A randomly sampled action.
        """
        return self.env.action_space.sample()

    def reset(self) -> np.ndarray:
        """Reset the environment.

        Returns:
            np.ndarray: The initial observation after resetting the environment.
        """
        state, _ = self.env.reset()
        return state

    def step(self, action: int) -> tuple:
        """Take a step in the environment.

        Args:
            action: The action to be taken.

        Returns:
            tuple: A tuple containing the next state, reward, terminated flag, and truncated flag.
        """
        state, reward, terminated, truncated, _ = self.env.step(action)
        return state, reward, terminated, truncated

    def render_frame(self) -> np.ndarray:
        """Render a frame from the environment.

        Returns:
            np.ndarray: The rendered frame.
        """
        frame = self.env.render()
        return frame

    def close(self) -> None:
        """Close the environment."""
        self.env.close()

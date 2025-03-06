import numpy as np
import gymnasium as gym
from gymnasium import spaces


class GymEnvironment:
    def __init__(self, env_name: str, seed: int, render_mode: str = "rgb_array"):
        self.env = gym.make(env_name, render_mode=render_mode)
        _, _ = self.env.reset(seed=seed)
        self.env.action_space.seed(seed)

    def max_action_value(self):
        return self.env.action_space.high[0]

    def min_action_value(self):
        return self.env.action_space.low[0]

    def observation_space(self):
        return self.env.observation_space.shape[0]

    def action_num(self) -> int:
        if isinstance(self.env.action_space, spaces.Box):
            action_num = self.env.action_space.shape[0]
        elif isinstance(self.env.action_space, spaces.Discrete):
            action_num = self.env.action_space.n
        return action_num

    def sample_action(self) -> int:
        return self.env.action_space.sample()

    def reset(self) -> np.ndarray:
        state, _ = self.env.reset()
        return state

    def step(self, action: int) -> tuple:
        state, reward, done, truncated, _ = self.env.step(action)
        return state, reward, done, truncated

    def render_frame(self) -> np.ndarray:
        frame = self.env.render()
        return frame

    def close(self):
        self.env.close()

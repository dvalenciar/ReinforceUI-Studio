import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.wrappers import RescaleAction


class GymEnvironment:
    def __init__(self, env_name: str, seed: int, render_mode: str = "rgb_array"):
        self.env = gym.make(env_name, render_mode=render_mode)
        if not isinstance(self.env.action_space, spaces.Discrete):
            self.env = RescaleAction(self.env, min_action=-1, max_action=1)
        _, _ = self.env.reset(seed=seed)
        self.env.action_space.seed(seed)

    def max_action_value(self):
        return self.env.action_space.high[0]

    def min_action_value(self):
        return self.env.action_space.low[0]

    def observation_space(self):
        return self.env.observation_space.shape[0]

    def action_num(self):
        if isinstance(self.env.action_space, spaces.Box):
            return self.env.action_space.shape[0]
        elif isinstance(self.env.action_space, spaces.Discrete):
            return self.env.action_space.n

    def sample_action(self) -> int:
        return self.env.action_space.sample()

    def reset(self) -> np.ndarray:
        state, _ = self.env.reset()
        return state

    def step(self, action: int) -> tuple:
        state, reward, terminated, truncated, _ = self.env.step(action)
        return state, reward, terminated, truncated

    def render_frame(self) -> np.ndarray:
        frame = self.env.render()
        return frame

    def close(self):
        self.env.close()

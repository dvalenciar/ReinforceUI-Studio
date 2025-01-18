import numpy as np
import gymnasium as gym


class GymEnvironment:
    def __init__(self, env_name: str, seed: int):
        self.env = gym.make(env_name, render_mode="rgb_array")
        _, _ = self.env.reset(seed=seed)
        self.env.action_space.seed(seed)

    def max_action_value(self):
        return self.env.action_space.high[0]

    def min_action_value(self):
        return self.env.action_space.low[0]

    def observation_space(self):
        return self.env.observation_space.shape[0]

    def action_num(self) -> int:
        return self.env.action_space.shape[0]

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

from collections import deque
from functools import cached_property
import numpy as np


class ImageWrapper:
    def __init__(self, config, environment):

        self.encoder = config.get("encoder")
        self.environment = environment

        self.grey_scale = False

        self.frames_to_stack = 3
        self.frames_stacked: deque[list[np.ndarray]] = deque(
            [], maxlen=self.frames_to_stack
        )

        self.frame_width = 256
        self.frame_height = 256


    def observation_space(self):
        channels = 1 if self.grey_scale else 3
        channels *= self.frames_to_stack
        image_space = (channels, self.frame_width, self.frame_height)
        return image_space

    def grab_frame(self):
        frame = self.environment.grab_frame(height=self.frame_height, width=self.frame_width)

    @cached_property
    def action_num(self):
        return self.environment.action_num()

    @cached_property
    def sample_action(self):
        return self.environment.action_space.sample()


    def reset(self):
        pass




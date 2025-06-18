import numpy as np
from collections import deque
from functools import cached_property
from reinforceui_studio.RL_helpers.cnn_encoder import CnnEncoder


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
        self.cnn_encoder = CnnEncoder(image_size=(self.frame_width, self.frame_height))

    def max_action_value(self):
        return self.environment.max_action_value

    def min_action_value(self):
        return self.environment.min_action_value


    def observation_space(self):
        # todo this is potencially incorrect since it neeeds to return the size of the embedding here
        # todo so basically this is the embedding_dim=512 for restn and convext net is othe number
        # channels = 1 if self.grey_scale else 3
        # channels *= self.frames_to_stack
        # image_space = (channels, self.frame_width, self.frame_height)
        # return image_space
        return self.cnn_encoder.embedding_size

    @cached_property
    def action_num(self):
        return self.environment.action_num()

    def sample_action(self):
        return self.environment.sample_action()

    def reset(self):
        frame = self.environment.grab_frame(height=self.frame_height, width=self.frame_width)
        frame = np.moveaxis(frame, -1, 0)
        for _ in range(self.frames_to_stack):
            self.frames_stacked.append(frame)
        stacked_frames = np.concatenate(list(self.frames_stacked), axis=0)
        state = self.cnn_encoder.create_embedding (stacked_frames)
        return state

    def step(self, action:int):
        frame = self.environment.grab_frame(height=self.frame_height, width=self.frame_width)
        frame = np.moveaxis(frame, -1, 0)
        self.frames_stacked.append(frame)
        stacked_frames = np.concatenate(list(self.frames_stacked), axis=0)
        state = self.cnn_encoder.create_embedding(stacked_frames)
        _, reward, done, truncated = self.environment.step(action)
        return state, reward, done, truncated

    def render_frame(self):
        return self.environment.render_frame()

    def close(self):
        self.environment.close()












from reinforceui_studio.RL_helpers.cnn_encoder import CnnEncoder


class ImageWrapper:
    def __init__(self, config, environment):

        self.environment = environment
        self.cnn_encoder = CnnEncoder(model_name=config.get("encoder"))

    def max_action_value(self):
        return self.environment.max_action_value

    def min_action_value(self):
        return self.environment.min_action_value

    def observation_space(self):
        return self.cnn_encoder.fc_out_dim

    def action_num(self):
        return self.environment.action_num()

    def sample_action(self):
        return self.environment.sample_action()

    def reset(self):
        _ = self.environment.reset() # Reset the environment
        frame = self.environment.grab_frame()
        state = self.cnn_encoder.create_embedding(frame)
        return state

    def step(self, action:int):
        _, reward, done, truncated = self.environment.step(action)
        frame = self.environment.grab_frame()
        state = self.cnn_encoder.create_embedding(frame)
        return state, reward, done, truncated

    def render_frame(self):
        return self.environment.render_frame()

    def close(self):
        self.environment.close()

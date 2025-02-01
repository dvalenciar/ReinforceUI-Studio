from dm_control import suite
import numpy as np
import matplotlib.pyplot as plt


class DMControlEnvironment:
    def __init__(self, env_name: str, seed: int, render_mode: str = None):
        if env_name == "ball_in_cup_catch":
            self.domain = "ball_in_cup"
            task = "catch"
        else:
            try:
                # Split environment name into domain and task
                self.domain, task = env_name.split("_", 1)
            except ValueError:
                raise ValueError(f"Invalid environment name '{env_name}'.")

        self.env = suite.load(
            domain_name=self.domain, task_name=task, task_kwargs={"random": seed}
        )
        self.render_mode = render_mode

    def max_action_value(self):
        return self.env.action_spec().maximum[0]

    def min_action_value(self):
        return self.env.action_spec().minimum[0]

    def observation_space(self):
        observation_spec = self.env.observation_spec()
        observation_size = sum(
            np.prod(spec.shape) for spec in observation_spec.values()
        )
        return observation_size

    def action_num(self) -> int:
        return self.env.action_spec().shape[0]

    def sample_action(self) -> np.ndarray:
        return np.random.uniform(
            self.min_action_value(), self.max_action_value(), size=self.action_num()
        )

    def reset(self) -> np.ndarray:
        time_step = self.env.reset()
        observation = np.hstack(list(time_step.observation.values()))
        return observation

    def step(self, action) -> tuple:
        time_step = self.env.step(action)
        state, reward, done = (
            np.hstack(list(time_step.observation.values())),
            time_step.reward,
            time_step.last(),
        )
        # for consistency with the GymEnvironment class  truncated is set to False
        return state, reward, done, False

    def render_frame(self, height=240, width=300) -> np.ndarray:
        frame1 = self.env.physics.render(camera_id=0, height=height, width=width)
        frame2 = self.env.physics.render(camera_id=1, height=height, width=width)
        combined_frame = np.hstack((frame1, frame2))  # Combine frames horizontally

        # display the frame similar to GymEnvironment with render_mode=render_mode
        # using matplotlib to display the frame was chosen because easy, This simple display the frame
        if self.render_mode == "human":
            plt.imshow(combined_frame)
            plt.axis("off")
            plt.show(block=False)  # Non-blocking
            plt.pause(0.01)  # Adjust refresh rate
            plt.clf()  # Clear figure for the next frame
        return combined_frame

    def close(self):
        plt.close()  # Close the figure

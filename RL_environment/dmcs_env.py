from dm_control import suite
import numpy as np
import matplotlib.pyplot as plt


class DMControlEnvironment:
    """Environment wrapper for DeepMind Control Suite.

    This class provides an interface to interact with environments from the
    DeepMind Control Suite, including methods for resetting the environment,
    taking steps, rendering frames, and sampling actions.

    """

    def __init__(self, env_name: str, seed: int, render_mode: str = None) -> None:
        """Initialize the DMControlEnvironment.

        Args:
            env_name: The name of the environment in the format 'domain_task'.
            seed: The random seed for the environment.
            render_mode: The mode for rendering frames. Defaults to None.
        """
        if env_name == "ball_in_cup_catch":
            self.domain = "ball_in_cup"
            task = "catch"
        else:
            try:
                self.domain, task = env_name.split("_", 1)
            except ValueError:
                raise ValueError(f"Invalid environment name '{env_name}'.")

        self.env = suite.load(
            domain_name=self.domain,
            task_name=task,
            task_kwargs={"random": seed},
        )
        self.render_mode = render_mode

    def max_action_value(self) -> float:
        """Get the maximum action value.

        Returns:
            float: The maximum value for actions.
        """
        return self.env.action_spec().maximum[0]

    def min_action_value(self) -> float:
        """Get the minimum action value.

        Returns:
            float: The minimum value for actions.
        """
        return self.env.action_spec().minimum[0]

    def observation_space(self) -> int:
        """Get the size of the observation space.

        Returns:
            int: The size of the observation space.
        """
        observation_spec = self.env.observation_spec()
        observation_size = sum(
            np.prod(spec.shape) for spec in observation_spec.values()
        )
        return int(observation_size)

    def action_num(self) -> int:
        """Get the number of actions.

        Returns:
            int: The number of actions.
        """
        return self.env.action_spec().shape[0]

    def sample_action(self) -> np.ndarray:
        """Sample a random action.

        Returns:
            np.ndarray: A randomly sampled action.
        """
        return np.random.uniform(
            self.min_action_value(),
            self.max_action_value(),
            size=self.action_num(),
        )

    def reset(self) -> np.ndarray:
        """Reset the environment.

        Returns:
            np.ndarray: The initial observation after resetting the environment.
        """
        time_step = self.env.reset()
        observation = np.hstack(list(time_step.observation.values()))
        return observation

    def step(self, action: int) -> tuple:
        """Take a step in the environment.

        Args:
            action: The action to be taken.

        Returns:
            tuple: A tuple containing the next state, reward, done flag, and truncated flag.
        """
        time_step = self.env.step(action)
        state, reward, done = (
            np.hstack(list(time_step.observation.values())),
            time_step.reward,
            time_step.last(),
        )
        return state, reward, done, False

    def render_frame(self, height: int = 240, width: int = 300) -> np.ndarray:
        """Render a frame from the environment.

        Args:
            height: The height of the rendered frame. Defaults to 240.
            width: The width of the rendered frame. Defaults to 300.

        Returns:
            np.ndarray: The rendered frame.
        """
        frame1 = self.env.physics.render(camera_id=0, height=height, width=width)
        frame2 = self.env.physics.render(camera_id=1, height=height, width=width)
        combined_frame = np.hstack((frame1, frame2))

        if self.render_mode == "human":
            plt.imshow(combined_frame)
            plt.axis("off")
            plt.show(block=False)
            plt.pause(0.01)
            plt.clf()
        return combined_frame

    def close(self) -> None:
        """Close the environment and any open figures."""
        plt.close()

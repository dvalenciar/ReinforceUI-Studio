from typing import Any
from RL_helpers.util import set_seed


def policy_loop_test(
    env: Any,
    rl_agent: Any,
    logger: Any,
    number_test_episodes: int = 1,
    algo_name: str = None,
) -> None:
    """Test the policy of a reinforcement learning agent.

    This function tests the policy of a reinforcement learning agent
    over a specified number of episodes and records the test metrics.

    Args:
        env: The environment in which the agent is tested.
        rl_agent: The reinforcement learning agent being tested.
        logger: The logger for recording test metrics.
        number_test_episodes: The number of test episodes. Defaults to 1.
        algo_name: The name of the algorithm used by the agent. Defaults to None.
    """
    rl_agent.load_models(filename="model", filepath=f"{logger.log_dir}/models_log")
    logger.start_video_record(env.render_frame())
    for episode in range(number_test_episodes):
        state = env.reset()
        done = False
        truncated = False
        episode_reward = 0
        while not done and not truncated:
            if algo_name == "PPO":
                action, _ = rl_agent.select_action_from_policy(state)
            elif algo_name == "DQN":
                action = rl_agent.select_action_from_policy(state)
            else:
                action = rl_agent.select_action_from_policy(state, evaluation=True)
            state, reward, done, truncated = env.step(action)
            episode_reward += reward
            logger.record_video_frame(env.render_frame())
    logger.end_video_record()


def policy_from_model_load_test(config_data: dict, models_log_path: str) -> None:
    """Test the policy of a loaded model.

    This function tests the policy of a reinforcement learning agent
    loaded from a model file over a single episode.

    Args:
        config_data: The configuration data for the environment and agent.
        models_log_path: The path to the directory containing the model files.
    """
    from RL_loops.training_policy_loop import (
        import_algorithm_instance,
        create_environment_instance,
    )

    algorithm_name = config_data.get("Algorithm")
    set_seed(int(config_data.get("Shared Parameters").get("Seed")))

    env_data = {
        "Seed": int(config_data.get("Shared Parameters").get("Seed")),
        "selected_platform": config_data.get("selected_platform"),
        "selected_environment": config_data.get("selected_environment"),
    }

    algorithm = import_algorithm_instance(algorithm_name)
    env = create_environment_instance(env_data, render_mode="human")
    rl_agent = algorithm(
        env.observation_space(),
        env.action_num(),
        config_data.get("Hyperparameters"),
    )
    rl_agent.load_models(filename="model", filepath=models_log_path)
    for episode in range(1):
        state = env.reset()
        done = False
        truncated = False
        episode_reward = 0
        while not done and not truncated:
            if algorithm_name == "PPO":
                action, _ = rl_agent.select_action_from_policy(state)
            elif algorithm_name == "DQN":
                action = rl_agent.select_action_from_policy(state)
            else:
                action = rl_agent.select_action_from_policy(state, evaluation=True)
            state, reward, done, truncated = env.step(action)
            episode_reward += reward
            env.render_frame()
    env.close()

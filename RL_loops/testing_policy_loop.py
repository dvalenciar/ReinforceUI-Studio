import importlib

from RL_helpers.util import denormalize_action, set_seed
from RL_environment.dmcs_env import DMControlEnvironment
from RL_environment.gym_env import GymEnvironment


def policy_loop_test(env, rl_agent, logger, number_test_episodes=1):
    rl_agent.load_models(filename="model", filepath=f"{logger.log_dir}/models_log")
    frame = env.render_frame()
    logger.start_video_record(frame)
    for episode in range(number_test_episodes):
        state = env.reset()
        done = False
        truncated = False
        episode_reward = 0
        while not done and not truncated:
            action = rl_agent.select_action_from_policy(state, evaluation=True)
            action_env = denormalize_action(
                action, env.max_action_value(), env.min_action_value()
            )
            next_state, reward, done, truncated = env.step(action_env)
            state = next_state
            episode_reward += reward
            frame = env.render_frame()
            logger.record_video_frame(frame)
    logger.end_video_record()


def policy_from_model_load_test(config_data, models_log_path):
    selected_platform = config_data.get("selected_platform")
    selected_environment = config_data.get("selected_environment")
    algorithm = config_data.get("Algorithm")
    seed = int(config_data.get("Seed"))
    set_seed(seed)
    hyperparameters = config_data.get("Hyperparameters")
    # Create environment instance
    if selected_platform == "Gymnasium" or selected_platform == "MuJoCo":
        env = GymEnvironment(selected_environment, seed=seed, render_mode="human")
    elif selected_platform == "DMCS":
        env = DMControlEnvironment(selected_environment, seed=seed, render_mode="human")
    else:
        raise ValueError(f"Unsupported platform: {selected_platform}")

    algorithm_module = importlib.import_module(f"RL_algorithms.{algorithm}")
    algorithm_class = getattr(algorithm_module, algorithm)
    rl_agent = algorithm_class(
        env.observation_space(), env.action_num(), hyperparameters
    )

    # Test the policy
    rl_agent.load_models(filename="model", filepath=models_log_path)
    for episode in range(1):
        state = env.reset()
        done = False
        truncated = False
        episode_reward = 0
        while not done and not truncated:
            action = rl_agent.select_action_from_policy(state, evaluation=True)
            action_env = denormalize_action(
                action, env.max_action_value(), env.min_action_value()
            )
            next_state, reward, done, truncated = env.step(action_env)
            state = next_state
            episode_reward += reward
            env.render_frame()
    env.close()

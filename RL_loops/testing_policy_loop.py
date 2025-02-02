from RL_helpers.util import denormalize_action, set_seed


def policy_loop_test(env, rl_agent, logger, number_test_episodes=1):
    rl_agent.load_models(filename="model", filepath=f"{logger.log_dir}/models_log")
    logger.start_video_record(env.render_frame())
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
            logger.record_video_frame(env.render_frame())
    logger.end_video_record()


def policy_from_model_load_test(config_data, models_log_path):
    from RL_loops.training_policy_loop import (
        import_algorithm_instance,
        create_environment_instance,
    )

    set_seed(int(config_data.get("Seed")))
    algorithm = import_algorithm_instance(config_data)
    env = create_environment_instance(config_data, render_mode="human")
    rl_agent = algorithm(
        env.observation_space(), env.action_num(), config_data.get("Hyperparameters")
    )
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

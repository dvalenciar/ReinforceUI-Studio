from RL_helpers.util import denormalize_action


def test_policy_loop(env, rl_agent, logger, number_test_episodes=1):
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

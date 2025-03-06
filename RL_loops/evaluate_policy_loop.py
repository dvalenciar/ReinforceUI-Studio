from RL_helpers.util import denormalize_action


def evaluate_policy_loop(
    env, rl_agent, number_eval_episodes, logger, total_steps, algo_name=None
):
    total_reward = 0
    df_log = None
    for episode in range(number_eval_episodes):
        state = env.reset()
        episode_timesteps = 0
        episode_reward = 0
        done = False
        truncated = False
        while not done and not truncated:
            episode_timesteps += 1
            if algo_name == "PPO":
                action, _ = rl_agent.select_action_from_policy(state)
                action_env = denormalize_action(action, env.max_action_value(), env.min_action_value())
            elif algo_name == "DQN":
                action_env = rl_agent.select_action_from_policy(state)
            else:
                action = rl_agent.select_action_from_policy(state, evaluation=True)
            if algo_name != "DQN":
                action_env = denormalize_action(action, env.max_action_value(), env.min_action_value())

            next_state, reward, done, truncated = env.step(action_env)
            episode_reward += reward
            state = next_state

        total_reward += episode_reward
        average_reward = total_reward / (episode + 1)
        df_log = logger.log_evaluation(
            episode + 1,
            episode_reward,
            episode_timesteps,
            total_steps + 1,
            average_reward,
        )
    return df_log

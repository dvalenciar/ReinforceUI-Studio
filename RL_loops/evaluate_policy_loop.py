

def evaluate_policy_loop(
    env, rl_agent, number_eval_episodes, logger, total_steps, algo_name=None
):

    df_log = None
    total_reward_env = 0
    episode_timestep_env = 0
    episode_reward_env = 0
    done = False
    truncated = False
    state = env.reset()

    for episode in range(number_eval_episodes):
        while not done and not truncated:
            episode_timestep_env += 1
            if algo_name == "PPO":
                action, _ = rl_agent.select_action_from_policy(state)
            elif algo_name == "DQN":
                action = rl_agent.select_action_from_policy(state)
            else:
                action = rl_agent.select_action_from_policy(state, evaluation=True)

            state, reward, done, truncated = env.step(action)
            episode_reward_env += reward

            if done or truncated:
                total_reward_env += episode_reward_env
                average_reward = total_reward_env / (episode + 1)
                df_log = logger.log_evaluation(
                    episode + 1,
                    episode_reward_env,
                    episode_timestep_env,
                    total_steps + 1,
                    average_reward,
                )
                # Reset the environment
                state = env.reset()
                episode_timestep_env = 0
                episode_reward_env = 0
    return df_log

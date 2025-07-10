from typing import Any
from pandas import DataFrame


def evaluate_policy_loop(
    env: Any,
    rl_agent: Any,
    number_eval_episodes: int,
    logger: Any,
    total_steps: int,
    alg_name: str = None,
) -> DataFrame:
    """Evaluate the policy of a reinforcement learning agent.

    This function evaluates the policy of a reinforcement learning agent
    over a specified number of episodes, logs the evaluation metrics, and
    returns the evaluation logs.

    Args:
        env: The environment in which the agent is evaluated.
        rl_agent: The reinforcement learning agent being evaluated.
        number_eval_episodes: The number of evaluation episodes.
        logger: The logger for recording evaluation metrics.
        total_steps: The total number of steps taken so far.
        alg_name: The name of the algorithm used by the agent. Defaults to None.

    Returns:
        pd.DataFrame: DataFrame containing the evaluation logs.
    """
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
            if alg_name == "PPO":
                action, _ = rl_agent.select_action_from_policy(state)
            elif alg_name == "DQN":
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

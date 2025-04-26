import time
import importlib
import random
from typing import Any, Callable
from RL_memory.memory_buffer import MemoryBuffer
from RL_environment.gym_env import GymEnvironment
from RL_environment.dmcs_env import DMControlEnvironment
from RL_helpers.util import set_seed
from RL_helpers.record_logger import RecordLogger
from RL_loops.evaluate_policy_loop import evaluate_policy_loop
from RL_loops.testing_policy_loop import policy_loop_test


def import_algorithm_instance(algorithm_name: str) -> Any:
    """Import the algorithm instance.

    Args:
        algorithm_name: The name of the algorithm to import.

    Returns:
        Any: The imported algorithm class.
    """
    algorithm_module = importlib.import_module(f"RL_algorithms.{algorithm_name}")
    algorithm_class = getattr(algorithm_module, algorithm_name)
    return algorithm_class


def create_environment_instance(
    config_data: dict,
    render_mode: str = "rgb_array",
    evaluation_env: bool = False,
) -> Any:
    """Create an environment instance.

    Args:
        config_data: The configuration data for the environment.
        render_mode: The mode for rendering frames. Defaults to "rgb_array".
        evaluation_env: Whether the environment is for evaluation. Defaults to False.

    Returns:
        Any: The created environment instance.
    """
    platform_name = config_data.get("selected_platform")
    env_name = config_data.get("selected_environment")
    seed = (
        int(config_data.get("Seed"))
        if not evaluation_env
        else (int(config_data.get("Seed")) + 1)
    )

    if platform_name == "Gymnasium" or platform_name == "MuJoCo":
        environment = GymEnvironment(env_name, seed, render_mode)
    elif platform_name == "DMCS":
        environment = DMControlEnvironment(env_name, seed, render_mode)
    else:
        raise ValueError(f"Unsupported platform: {platform_name}")
    return environment


def training_loop(  # noqa: C901
    config_data: dict,
    training_window: Any,
    log_folder_path: Any,
    algorithm_name: str,
    display_name: str,
    is_running: Callable,
) -> None:
    """Run the training loop for the reinforcement learning agent.

    Args:
        config_data: The configuration data for the training.
        training_window: The training window for updating progress.
        log_folder_path: The path to the log folder.
        algorithm_name: The name of the algorithm being used.
        display_name: The display name for the algorithm.
        is_running: A callable that returns True if training should continue.
    """
    set_seed(int(config_data.get("Seed")))
    algorithm = import_algorithm_instance(algorithm_name)

    env = create_environment_instance(
        config_data, render_mode="rgb_array", evaluation_env=False
    )
    env_evaluation = create_environment_instance(
        config_data, render_mode="rgb_array", evaluation_env=True
    )

    rl_agent = algorithm(
        env.observation_space(),
        env.action_num(),
        config_data.get("Hyperparameters"),
    )
    memory = MemoryBuffer(
        env.observation_space(),
        env.action_num(),
        config_data.get("Hyperparameters"),
        algorithm_name,
    )

    logger = RecordLogger(log_folder_path, rl_agent)

    steps_training = int(config_data.get("Training Steps", 1000000))
    evaluation_interval = int(config_data.get("Evaluation Interval", 1000))
    log_interval = int(config_data.get("Log Interval", 1000))
    number_eval_episodes = int(config_data.get("Evaluation Episodes", 10))

    episode_timesteps = 0
    episode_num = 0
    episode_reward = 0
    total_episode_time = 0
    episode_start_time = time.time()
    state = env.reset()

    is_ppo = algorithm_name == "PPO"
    is_dqn = algorithm_name == "DQN"

    if is_ppo:
        max_steps_per_batch = int(
            config_data.get("Hyperparameters").get("max_steps_per_batch")
        )
    elif is_dqn:
        exploration_rate = 1
        epsilon_min = float(config_data.get("Hyperparameters").get("epsilon_min"))
        epsilon_decay = float(config_data.get("Hyperparameters").get("epsilon_decay"))
        G = int(config_data.get("G Value", 1))  # noqa: N806
        batch_size = int(config_data.get("Batch Size", 32))
        steps_exploration = int(config_data.get("Exploration Steps", 1000))
    else:
        G = int(config_data.get("G Value", 1))  # noqa: N806
        batch_size = int(config_data.get("Batch Size", 32))
        steps_exploration = int(config_data.get("Exploration Steps", 1000))

    training_completed = True
    for total_step_counter in range(steps_training):
        if not is_running():  # Check the running state using the callable
            print("Training loop interrupted. Exiting...")
            training_completed = False
            break

        progress = (total_step_counter + 1) / steps_training * 100
        episode_timesteps += 1

        # Select action
        if is_ppo:
            action, log_prob = rl_agent.select_action_from_policy(state)
        if is_dqn:
            if total_step_counter < steps_exploration:
                action = env.sample_action()
            else:
                exploration_rate *= epsilon_decay
                exploration_rate = max(epsilon_min, exploration_rate)
                if random.random() < exploration_rate:
                    action = env.sample_action()
                else:
                    action = rl_agent.select_action_from_policy(state)
        if not is_ppo and not is_dqn:
            if total_step_counter < steps_exploration:
                action = env.sample_action()
            else:
                action = rl_agent.select_action_from_policy(state)

        # Take a step in the environment
        next_state, reward, done, truncated = env.step(action)

        # Store experience in memory
        if is_ppo:
            memory.add_experience(state, action, reward, next_state, done, log_prob)
        else:
            memory.add_experience(state, action, reward, next_state, done)

        state = next_state
        episode_reward += reward

        # Train the policy
        if is_ppo and (total_step_counter + 1) % max_steps_per_batch == 0:
            rl_agent.train_policy(memory)

        elif is_dqn and total_step_counter > batch_size:
            for _ in range(G):
                rl_agent.train_policy(memory, batch_size)

        elif not is_ppo and not is_dqn and total_step_counter >= steps_exploration:
            for _ in range(G):
                rl_agent.train_policy(memory, batch_size)

        # Handle episode completion
        if done or truncated:
            episode_time = time.time() - episode_start_time
            total_episode_time += episode_time
            average_episode_time = total_episode_time / (episode_num + 1)
            remaining_episodes = (
                steps_training - total_step_counter - 1
            ) // episode_timesteps
            estimated_time_remaining = average_episode_time * remaining_episodes
            episode_time_str = time.strftime(
                "%H:%M:%S", time.gmtime(max(0, estimated_time_remaining))
            )

            training_window.update_algo_signal.emit(
                display_name, "Time Remaining", episode_time_str
            )
            training_window.update_algo_signal.emit(
                display_name, "Episode Number", episode_num + 1
            )
            training_window.update_algo_signal.emit(
                display_name, "Episode Reward", round(episode_reward, 3)
            )
            training_window.update_algo_signal.emit(
                display_name, "Episode Steps", episode_timesteps
            )

            df_log_train = logger.log_training(
                episode=episode_num + 1,
                episode_reward=episode_reward,
                episode_steps=episode_timesteps,
                total_timesteps=total_step_counter + 1,
                duration=episode_time,
            )

            training_window.update_plot_signal.emit(
                display_name, df_log_train, "training"
            )

            # Reset the environment
            state = env.reset()
            episode_timesteps = 0
            episode_num += 1
            episode_reward = 0
            episode_start_time = time.time()

        # Evaluate the policy
        if (total_step_counter + 1) % evaluation_interval == 0:
            df_log_evaluation = evaluate_policy_loop(
                env_evaluation,
                rl_agent,
                number_eval_episodes,
                logger,
                total_step_counter,
                algorithm_name,
            )
            df_grouped = df_log_evaluation.groupby(
                "Total Timesteps", as_index=False
            ).last()

            training_window.update_plot_signal.emit(
                display_name, df_grouped, "evaluation"
            )

        # Update the training window
        training_window.update_algo_signal.emit(display_name, "Progress", int(progress))
        training_window.update_algo_signal.emit(
            display_name, "Total Steps", total_step_counter + 1
        )

        # Save checkpoint based on log interval
        if (total_step_counter + 1) % log_interval == 0:
            logger.save_logs(plot_flag=False)

    # Finalize training
    logger.save_logs(plot_flag=True)
    policy_loop_test(env, rl_agent, logger, algo_name=algorithm_name)

    training_window.training_completed_signal.emit(display_name, training_completed)

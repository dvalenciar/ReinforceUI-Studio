import time
import importlib
from RL_memory.memory_buffer import MemoryBuffer
from RL_environment.gym_env import GymEnvironment
from RL_environment.dmcs_env import DMControlEnvironment
from RL_helpers.util import normalize_action, denormalize_action, set_seed
from RL_helpers.record_logger import RecordLogger
from RL_loops.evaluate_policy_loop import evaluate_policy_loop
from RL_loops.testing_policy_loop import policy_loop_test


def import_algorithm_instance(config_data):
    algorithm_name = config_data.get("Algorithm")
    algorithm_module = importlib.import_module(f"RL_algorithms.{algorithm_name}")
    algorithm_class = getattr(algorithm_module, algorithm_name)
    return algorithm_class


def create_environment_instance(config_data, render_mode="rgb_array"):
    platform_name = config_data.get("selected_platform")
    env_name = config_data.get("selected_environment")
    seed = int(config_data.get("Seed"))

    if platform_name == "Gymnasium" or platform_name == "MuJoCo":
        environment = GymEnvironment(env_name, seed, render_mode)
    elif platform_name == "DMCS":
        environment = DMControlEnvironment(env_name, seed, render_mode)
    else:
        raise ValueError(f"Unsupported platform: {platform_name}")
    return environment


def training_loop(config_data, training_window, log_folder_path, is_running):
    set_seed(int(config_data.get("Seed")))
    algorithm = import_algorithm_instance(config_data)
    env = create_environment_instance(config_data)
    rl_agent = algorithm(
        env.observation_space(), env.action_num(), config_data.get("Hyperparameters")
    )
    memory = MemoryBuffer(
        env.observation_space(), env.action_num(), config_data.get("Hyperparameters")
    )
    logger = RecordLogger(log_folder_path, rl_agent)

    steps_training = int(config_data.get("Training Steps", 1000000))
    steps_exploration = int(config_data.get("Exploration Steps", 1000))
    G = int(config_data.get("G Value", 1))
    batch_size = int(config_data.get("Batch Size", 32))
    evaluation_interval = int(config_data.get("Evaluation Interval", 1000))
    log_interval = int(config_data.get("Log Interval", 1000))
    number_eval_episodes = int(config_data.get("Evaluation Episodes", 10))

    episode_timesteps = 0
    episode_num = 0
    episode_reward = 0
    total_episode_time = 0
    episode_start_time = time.time()
    state = env.reset()

    if config_data.get("Algorithm") == "PPO":
        max_steps_per_batch = int(config_data.get("Hyperparameters").get("max_steps_per_batch"))

        training_completed = True
        for total_step_counter in range(steps_training):
            if not is_running():  # Check the running state using the callable
                print("Training loop interrupted. Exiting...")
                training_completed = False
                break

            progress = (total_step_counter + 1) / steps_training * 100
            episode_timesteps += 1

            action, log_prob = rl_agent.select_action_from_policy(state)
            action_env = denormalize_action(action, env.max_action_value(), env.min_action_value())

            next_state, reward, done, truncated = env.step(action_env)
            episode_reward += reward

            memory.add_experience(state, action, reward, next_state, done, log_prob)
            state = next_state

            if (total_step_counter + 1) %  max_steps_per_batch == 0:
                for _ in range(G):
                    rl_agent.train_policy(memory)

            if (total_step_counter + 1) % evaluation_interval == 0:
                df_log_evaluation = evaluate_policy_loop(env, rl_agent, number_eval_episodes, logger, total_step_counter, "PPO")
                df_grouped = df_log_evaluation.groupby("Total Timesteps", as_index=False).last()
                training_window.update_plot_eval(df_grouped)

            training_window.update_progress_signal.emit(int(progress))
            training_window.update_step_signal.emit(total_step_counter + 1)

            if done or truncated:
                # calculate the estimated time remaining
                episode_time = time.time() - episode_start_time
                total_episode_time += episode_time
                average_episode_time = total_episode_time / (episode_num + 1)
                remaining_episodes = (steps_training - total_step_counter - 1) // episode_timesteps
                estimated_time_remaining = average_episode_time * remaining_episodes
                episode_time_str = time.strftime(
                    "%H:%M:%S", time.gmtime(max(0, estimated_time_remaining))
                )
                training_window.update_time_remaining_signal.emit(episode_time_str)

                training_window.update_episode_signal.emit(episode_num + 1)
                training_window.update_reward_signal.emit(round(episode_reward, 3))
                training_window.update_episode_steps_signal.emit(episode_timesteps)

                df_log_train = logger.log_training(
                    episode_num + 1,
                    episode_reward,
                    episode_timesteps,
                    total_step_counter + 1,
                    episode_time,
                )
                training_window.update_plot(df_log_train)

                # Save checkpoint based on log interval
                if (total_step_counter + 1) % log_interval == 0:
                    logger.save_checkpoint()

                # reset the environment
                state = env.reset()
                episode_timesteps = 0
                episode_num += 1
                episode_reward = 0
                episode_start_time = time.time()

        logger.save_logs()
        policy_loop_test(env, rl_agent, logger, algo_name="PPO")
        training_window.training_completed_signal.emit(training_completed)

    else:
        training_completed = True
        for total_step_counter in range(steps_training):
            if not is_running():  # Check the running state using the callable
                print("Training loop interrupted. Exiting...")
                training_completed = False
                break

            progress = (total_step_counter + 1) / steps_training * 100
            episode_timesteps += 1

            if total_step_counter < steps_exploration:
                action_env = env.sample_action()
                action = normalize_action(
                    action_env, env.max_action_value(), env.min_action_value()
                )
            else:
                action = rl_agent.select_action_from_policy(state)
                action_env = denormalize_action(
                    action, env.max_action_value(), env.min_action_value()
                )

            next_state, reward, done, truncated = env.step(action_env)
            episode_reward += reward

            memory.add_experience(state, action, reward, next_state, done)
            state = next_state

            if total_step_counter >= steps_exploration:
                for _ in range(G):
                    rl_agent.train_policy(memory, batch_size)

            if (total_step_counter + 1) % evaluation_interval == 0:
                df_log_evaluation = evaluate_policy_loop(
                    env, rl_agent, number_eval_episodes, logger, total_step_counter
                )
                df_grouped = df_log_evaluation.groupby(
                    "Total Timesteps", as_index=False
                ).last()
                training_window.update_plot_eval(df_grouped)

            training_window.update_progress_signal.emit(int(progress))
            training_window.update_step_signal.emit(total_step_counter + 1)

            if done or truncated:
                # calculate the estimated time remaining
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
                training_window.update_time_remaining_signal.emit(episode_time_str)

                training_window.update_episode_signal.emit(episode_num + 1)
                training_window.update_reward_signal.emit(round(episode_reward, 3))
                training_window.update_episode_steps_signal.emit(episode_timesteps)

                df_log_train = logger.log_training(
                    episode_num + 1,
                    episode_reward,
                    episode_timesteps,
                    total_step_counter + 1,
                    episode_time,
                )
                training_window.update_plot(df_log_train)

                # Save checkpoint based on log interval
                if (total_step_counter + 1) % log_interval == 0:
                    logger.save_checkpoint()

                # reset the environment
                state = env.reset()
                episode_timesteps = 0
                episode_num += 1
                episode_reward = 0
                episode_start_time = time.time()

        logger.save_logs()
        policy_loop_test(env, rl_agent, logger)
        training_window.training_completed_signal.emit(training_completed)

import os
import numpy as np
import cv2
import pandas as pd


from RL_helpers.plotters import plot_logs


class RecordLogger:
    def __init__(self, log_dir: str, rl_agent) -> None:  # noqa: ANN001
        """Initialize the RecordLogger.

        Args:
            log_dir: Directory to save logs and videos.
            rl_agent: The reinforcement learning agent being logged, this come from the algorithm.
        """
        self.logs_training = []
        self.logs_evaluation = []
        self.rl_agent = rl_agent
        self.log_dir = log_dir
        self.video_writer = None

        self.data_log_dir = os.path.join(log_dir, "data_log")
        self.model_log_dir = os.path.join(log_dir, "models_log")

        os.makedirs(self.data_log_dir, exist_ok=True)
        os.makedirs(self.model_log_dir, exist_ok=True)

    def log_training(
        self,
        episode: int,
        episode_reward: float,
        episode_steps: int,
        total_timesteps: int,
        duration: float,
    ) -> pd.DataFrame:
        """Log training metrics.

        Args:
            episode: The episode number.
            episode_reward: The reward obtained in the episode.
            episode_steps: The number of steps taken in the episode.
            total_timesteps: The total number of timesteps taken so far.
            duration: The duration of the episode.

        Returns:
            pd.DataFrame: DataFrame containing the training logs.
        """
        self.logs_training.append(
            {
                "Episode Number": episode,
                "Episode Reward": episode_reward,
                "Episode Steps": episode_steps,
                "Total Timesteps": total_timesteps,
                "Episode_Duration": duration,
            }
        )
        return pd.DataFrame(self.logs_training)

    def log_evaluation(
        self,
        episode: int,
        episode_reward: float,
        episode_steps: int,
        total_timesteps: int,
        average_reward: float,
    ) -> pd.DataFrame:
        """Log evaluation metrics.

        Args:
            episode: The episode number.
            episode_reward: The reward obtained in the episode.
            episode_steps: The number of steps taken in the episode.
            total_timesteps: The total number of timesteps taken so far.
            average_reward: The average reward obtained.

        Returns:
            pd.DataFrame: DataFrame containing the evaluation logs.
        """
        self.logs_evaluation.append(
            {
                "Episode Number": episode,
                "Episode Reward": episode_reward,
                "Episode Steps": episode_steps,
                "Total Timesteps": total_timesteps,
                "Average Reward": average_reward,
            }
        )
        return pd.DataFrame(self.logs_evaluation)

    @staticmethod
    def _save_csv(logs: list, filename: str) -> None:
        """Save logs to a CSV file.

        Args:
            logs: List of logs to be saved.
            filename: The name of the CSV file.
        """
        df = pd.DataFrame(logs)
        df.to_csv(filename, index=False)

    def save_logs(self, plot_flag: bool = False) -> None:
        """Save training and evaluation logs to CSV files and plot them."""
        self._save_csv(
            self.logs_training,
            os.path.join(self.data_log_dir, "training_log.csv"),
        )

        self._save_csv(
            self.logs_evaluation,
            os.path.join(self.data_log_dir, "evaluation_log.csv"),
        )

        self.rl_agent.save_models(filename="model", filepath=self.model_log_dir)

        if plot_flag:
            plot_logs(
                self.logs_training,
                "Total Timesteps",
                "Episode Reward",
                "Training Curve",
                "Steps",
                "Episode Reward",
                os.path.join(self.data_log_dir, "training_log.png"),
            )

            plot_logs(
                self.logs_evaluation,
                "Total Timesteps",
                "Average Reward",
                "Evaluation Curve",
                "Steps",
                "Average Reward",
                os.path.join(self.data_log_dir, "evaluation_log.png"),
            )

    def start_video_record(self, frame: np.ndarray) -> None:
        """Start recording a video.

        Args:
            frame: The first frame of the video to determine the video dimensions.
        """
        frame_height, frame_width, _ = frame.shape
        video_filename = os.path.join(self.log_dir, "video_tested_final_policy.mp4")
        self.video_writer = cv2.VideoWriter(
            video_filename,
            cv2.VideoWriter_fourcc(*"mp4v"),
            30,
            (frame_width, frame_height),
        )

    def record_video_frame(self, frame: np.ndarray) -> None:
        """Record a frame to the video.

        Args:
            frame: The frame to be recorded.
        """
        if self.video_writer:
            self.video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    def end_video_record(self) -> None:
        """End the video recording and release the video writer."""
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
            print("Video recording completed.")

import os
from typing import Any
from PyQt5.QtCore import QThread
from reinforceui_studio.RL_loops.training_policy_loop import training_loop


class TrainingThread(QThread):
    def __init__(
        self, training_window: Any, config_data: dict, log_folder: str
    ) -> None:
        """Thread for running the RL training loop in the background.

        Args:
            training_window (Any): Reference to the training window UI.
            config_data (dict): Configuration data for the training session.
            log_folder (str): Path to the folder for logging outputs.
        """
        super().__init__()
        self.config_data = config_data
        self.algorithm_name = config_data["Algorithm"]
        self.display_name = config_data["UniqueName"]
        self.training_window = training_window
        self.log_folder = os.path.join(log_folder, self.display_name)
        self._is_running = True

    def run(self) -> None:
        """Executes the training loop in a separate thread."""
        print(f"[{self.algorithm_name}] Training thread started")
        training_loop(
            config_data=self.config_data,
            training_window=self.training_window,
            log_folder_path=self.log_folder,
            algorithm_name=self.algorithm_name,
            display_name=self.display_name,
            is_running=lambda: self._is_running,
        )

    def stop(self) -> None:
        """Signals the thread to stop running."""
        print(f"[{self.algorithm_name}] Training thread stopped")
        self._is_running = False

import os
from PyQt5.QtCore import QThread
from reinforceui_studio.RL_loops.training_policy_loop import training_loop


class TrainingThread(QThread):
    def __init__(self, training_window, config_data, log_folder):
        super().__init__()
        self.config_data = config_data
        self.algorithm_name = config_data["Algorithm"]
        self.display_name = config_data["UniqueName"]
        self.training_window = training_window
        self.log_folder = os.path.join(log_folder, self.display_name)

        self._is_running = True

    def run(self):
        print(f"[{self.algorithm_name}] Training thread started")
        training_loop(
            config_data=self.config_data,
            training_window=self.training_window,
            log_folder_path=self.log_folder,
            algorithm_name=self.algorithm_name,
            display_name=self.display_name,
            is_running=lambda: self._is_running,
        )

    def stop(self):
        print(f"[{self.algorithm_name}] Training thread stopped")
        self._is_running = False

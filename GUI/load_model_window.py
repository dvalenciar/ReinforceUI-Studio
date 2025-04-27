import json
import os

from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import (
    QLabel,
    QVBoxLayout,
    QHBoxLayout,
    QFileDialog,
    QSpacerItem,
    QSizePolicy,
    QWidget,
)
from PyQt5.QtCore import Qt
from GUI.ui_base_window import BaseWindow
from GUI.ui_utils import create_button
from GUI.ui_styles import Styles
from RL_loops.testing_policy_loop import policy_from_model_load_test


class LoadConfigWindow(BaseWindow):
    def __init__(self, main_window, user_selections) -> None:  # noqa
        """Initialize the load model window.

        Args:
            main_window: Callback to return to main window
            user_selections: Dictionary containing user selections
        """
        super().__init__("Load Pre-trained Model", 900, 400)

        self.main_window = main_window
        self.user_selections = user_selections
        self.test_policy_button = None

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout()

        # Top layout with Back button
        top_layout = QHBoxLayout()
        back_button = create_button(
            self,
            "Back",
            width=120,
            height=50,
            icon=QIcon("media_resources/icons/back.svg"),
        )
        back_button.clicked.connect(self.back_main_window)
        top_layout.addWidget(back_button, alignment=Qt.AlignLeft)
        top_layout.addItem(
            QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)
        )
        main_layout.addLayout(top_layout)

        # Title and instructions
        welcome_load_screen_message = QLabel(
            "Please Load a Pre-trained Model Directory.",
            self,
        )
        welcome_load_screen_message.setWordWrap(True)
        welcome_load_screen_message.setAlignment(Qt.AlignCenter)
        welcome_load_screen_message.setStyleSheet(Styles.WELCOME_LABEL)
        main_layout.addWidget(welcome_load_screen_message)

        note_message = QLabel(
            "Note: The directory should contain the model files and the model configuration file. \n"
            "Ideally, this directory should be created by the ReinforceUI Studio to avoid any errors.",
            self,
        )
        note_message.setWordWrap(True)
        note_message.setAlignment(Qt.AlignCenter)
        note_message.setStyleSheet(Styles.TEXT_LABEL)
        main_layout.addWidget(note_message)

        # Status labels
        self.config_status = QLabel("Config.json: ❌", self)
        self.models_log_status = QLabel("models_log: ❌", self)
        self.info_display = QLabel("", self)
        self.config_status.setStyleSheet(Styles.TEXT_LABEL)
        self.models_log_status.setStyleSheet(Styles.TEXT_LABEL)
        self.info_display.setStyleSheet(Styles.TEXT_LABEL)
        main_layout.addWidget(self.config_status)
        main_layout.addWidget(self.models_log_status)
        main_layout.addWidget(self.info_display)

        # Button layout
        self.button_layout = QHBoxLayout()
        self.button_layout.setSpacing(20)
        load_button = create_button(self, "Load Directory", width=250, height=50)
        load_button.clicked.connect(self.load_directory)

        self.button_layout.addWidget(load_button)
        self.button_layout.setContentsMargins(100, 20, 100, 20)
        self.button_layout.setAlignment(Qt.AlignCenter)
        main_layout.addLayout(self.button_layout)

        main_widget.setLayout(main_layout)

    def load_directory(self) -> None:
        """Open file dialog to select model directory and verify its contents."""
        directory = QFileDialog.getExistingDirectory(self, "Select Directory", "")
        if directory:
            config_path = os.path.join(directory, "config.json")
            models_log_path = os.path.join(directory, "models_log")

            config_exists = os.path.exists(config_path)
            models_log_exists = os.path.exists(models_log_path)

            self._update_status_labels(config_exists, models_log_exists)

            if config_exists and models_log_exists:
                self._load_config_and_display(config_path, models_log_path)
            else:
                self.info_display.setText(
                    "Error: config.json or models_log directory is missing!"
                )
                self.info_display.setStyleSheet("color: red; font-size: 16px;")

    def _update_status_labels(
        self, config_exists: bool, models_log_exists: bool
    ) -> None:
        """Update status labels based on file existence.

        Args:
            config_exists: Boolean whether config file exists
            models_log_exists: Boolean whether models_log directory exists
        """
        self.config_status.setText(f"Config.json: {'✔️' if config_exists else '❌'}")
        self.models_log_status.setText(
            f"models_log: {'✔️' if models_log_exists else '❌'}"
        )
        self.config_status.setStyleSheet(
            "color: #2E7D32; font-size: 16px;  /* Darker Green */"
            if config_exists
            else "color: #D32F2F; font-size: 16px;"
        )
        self.models_log_status.setStyleSheet(
            "color: #2E7D32; font-size: 16px;"
            if models_log_exists
            else "color: #D32F2F; font-size: 16px;"
        )

    def _load_config_and_display(self, config_path: str, models_log_path: str) -> None:
        """Load config file and display model info.

        Args:
            config_path: Path to the config.json file
            models_log_path: Path to the models_log directory
        """
        with open(config_path, "r") as file:
            config_data = json.load(file)
            selected_platform = config_data.get("selected_platform", "N/A")
            selected_environment = config_data.get("selected_environment", "N/A")
            algorithm = config_data.get("Algorithm", "N/A")

            self.info_display.setText(
                f"Platform: {selected_platform}\n"
                f"Environment: {selected_environment}\n"
                f"Algorithm: {algorithm}"
            )
            self.info_display.setAlignment(Qt.AlignCenter)
            self.info_display.setStyleSheet("color: #2E7D32; font-size: 16px;")

            # Create Test Policy button if it doesn't exist
            if not self.test_policy_button:
                self.test_policy_button = create_button(
                    self, "Test Policy", width=250, height=50
                )
                self.test_policy_button.clicked.connect(
                    lambda: policy_from_model_load_test(config_data, models_log_path)
                )
                self.button_layout.addWidget(self.test_policy_button)

    def back_main_window(self) -> None:
        """Return to the main window."""
        self.close()
        self.main_window()


if __name__ == "__main__":
    import sys
    from PyQt5.QtWidgets import QApplication

    app = QApplication(sys.argv)
    window = LoadConfigWindow(lambda: None, {})
    window.show()
    sys.exit(app.exec_())

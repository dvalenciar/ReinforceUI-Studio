from PyQt5.QtWidgets import (
    QDialog,
    QDesktopWidget,
    QLabel,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QFileDialog,
    QSpacerItem,
    QSizePolicy,
)
from PyQt5.QtCore import Qt
import json
import os
from GUI.ui_base_window import BaseWindow
from GUI.ui_utils import create_button
from GUI.ui_styles import Styles
from RL_loops.testing_policy_loop import policy_from_model_load_test


class LoadConfigWindow(QDialog):
    def __init__(self, main_window, user_selections):
        super().__init__()

        self.main_window = main_window
        self.user_selections = user_selections
        self.setWindowTitle("Load Pre-trained Model")
        self.setFixedSize(900, 400)
        self.setStyleSheet("background-color: #121212;")
        self.center()

        main_layout = QVBoxLayout()

        # Top layout with Back button
        top_layout = QHBoxLayout()
        back_button = QPushButton("Back", self)
        back_button.setFixedSize(100, 35)
        self.apply_button_style(back_button)
        back_button.clicked.connect(self.back_main_window)
        top_layout.addWidget(back_button, alignment=Qt.AlignLeft)
        top_layout.addItem(
            QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)
        )
        main_layout.addLayout(top_layout)

        welcome_load_screen_message = QLabel(
            "Please Load a Pre-trained Model Directory.",
            self,
        )
        welcome_load_screen_message.setWordWrap(True)
        welcome_load_screen_message.setAlignment(Qt.AlignCenter)
        welcome_load_screen_message.setStyleSheet(
            "color: #E0E0E0; font-size: 18px; font-weight: bold;"
        )
        main_layout.addWidget(welcome_load_screen_message)

        note_message = QLabel(
            "Note: This Directory should contain the model files and the model configuration file. \n"
            "Ideally, this directory should be created by the ReinforceUI Studio to avoid any errors.",
            self,
        )
        note_message.setWordWrap(True)
        note_message.setAlignment(Qt.AlignCenter)
        note_message.setStyleSheet("color: #E0E0E0; font-size: 15px;")
        main_layout.addWidget(note_message)

        self.config_status = QLabel("Config.json: ❌", self)
        self.models_log_status = QLabel("models_log: ❌", self)
        self.info_display = QLabel("", self)
        self.config_status.setStyleSheet("color: #E0E0E0; font-size: 15px;")
        self.models_log_status.setStyleSheet(
            "color: #E0E0E0; font-size: 15px;"
        )
        self.info_display.setStyleSheet("color: #E0E0E0; font-size: 15px;")
        main_layout.addWidget(self.config_status)
        main_layout.addWidget(self.models_log_status)
        main_layout.addWidget(self.info_display)

        self.button_layout = QHBoxLayout()
        self.button_layout.setSpacing(20)
        load_button = self.create_button("Load Directory", 250, 50)
        self.apply_button_style(load_button)

        self.button_layout.addWidget(load_button)
        self.button_layout.setContentsMargins(100, 20, 100, 20)
        self.button_layout.setAlignment(Qt.AlignCenter)
        main_layout.addLayout(self.button_layout)

        self.setLayout(main_layout)

        load_button.clicked.connect(self.load_directory)

    def create_button(self, text, width, height, icon=None):
        button = QPushButton(text, self)
        button.setFixedSize(width, height)
        if icon:
            button.setIcon(icon)
            button.setStyleSheet("background-color: transparent;")
        return button

    @staticmethod
    def apply_button_style(button):
        button.setStyleSheet(Styles.BUTTON)

    def center(self):
        screen_geometry = QDesktopWidget().availableGeometry().center()
        frame_geometry = self.frameGeometry()
        frame_geometry.moveCenter(screen_geometry)
        self.move(frame_geometry.topLeft())

    def load_directory(self):
        directory = QFileDialog.getExistingDirectory(
            self, "Select Directory", ""
        )
        if directory:
            config_path = os.path.join(directory, "config.json")
            models_log_path = os.path.join(directory, "models_log")

            config_exists = os.path.exists(config_path)
            models_log_exists = os.path.exists(models_log_path)

            self.config_status.setText(
                f"Config.json: {'✔️' if config_exists else '❌'}"
            )
            self.models_log_status.setText(
                f"models_log: {'✔️' if models_log_exists else '❌'}"
            )
            self.config_status.setStyleSheet(
                "color: green; font-size: 16px;"
                if config_exists
                else "color: red; font-size: 16px;"
            )
            self.models_log_status.setStyleSheet(
                "color: green; font-size: 16px;"
                if models_log_exists
                else "color: red; font-size: 16px;"
            )

            if config_exists and models_log_exists:
                with open(config_path, "r") as file:
                    config_data = json.load(file)
                    selected_platform = config_data.get(
                        "selected_platform", "N/A"
                    )
                    selected_environment = config_data.get(
                        "selected_environment", "N/A"
                    )
                    algorithm = config_data.get("Algorithm", "N/A")
                    self.info_display.setText(
                        f"Platform: {selected_platform}\nEnvironment: {selected_environment}\nAlgorithm: {algorithm}"
                    )
                    self.info_display.setAlignment(Qt.AlignCenter)
                    self.info_display.setStyleSheet(
                        "color: #00FF00; font-size: 16px;"
                    )

                    # Create Test Policy button
                    test_policy_button = self.create_button(
                        "Test Policy", 250, 50
                    )
                    self.apply_button_style(test_policy_button)
                    self.button_layout.addWidget(test_policy_button)
                    test_policy_button.clicked.connect(
                        lambda: policy_from_model_load_test(
                            config_data, models_log_path
                        )
                    )
            else:
                self.info_display.setText(
                    "Error: config.json or models_log directory is missing!"
                )
                self.info_display.setStyleSheet("color: red; font-size: 16px;")

    def back_main_window(self):
        self.close()
        self.main_window()


if __name__ == "__main__":
    import sys
    from PyQt5.QtWidgets import QApplication

    app = QApplication(sys.argv)
    window = LoadConfigWindow(None, {})
    window.show()
    sys.exit(app.exec_())

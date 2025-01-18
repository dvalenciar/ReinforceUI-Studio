import yaml
from PyQt5.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QLabel,
    QComboBox,
    QPushButton,
    QHBoxLayout,
    QSpacerItem,
    QSizePolicy,
)
from PyQt5.QtCore import Qt
from GUI.select_algorithm_window import SelectAlgorithmWindow


class SelectEnvironmentWindow(QDialog):
    def __init__(self, platform_window, selected_platform, user_selections):
        super().__init__()

        self.platform_window = platform_window
        self.selected_platform = selected_platform
        self.user_selections = user_selections

        self.setWindowTitle(f"Select Environment")
        self.setFixedSize(500, 300)
        self.setStyleSheet("background-color: black;")

        layout = QVBoxLayout()
        button_layout = QHBoxLayout()

        back_button = QPushButton("Back", self)
        back_button.setFixedSize(100, 35)
        self.apply_button_style(back_button)
        back_button.clicked.connect(self.open_platform_selection)
        button_layout.addWidget(back_button)

        button_layout.addItem(
            QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)
        )

        next_button = QPushButton("Next", self)
        next_button.setFixedSize(100, 35)
        self.apply_button_style(next_button)
        next_button.clicked.connect(self.confirm_selection)
        button_layout.addWidget(next_button)

        layout.addLayout(button_layout)

        welcome_label = QLabel(
            f"Please select the environment for {selected_platform}", self
        )
        welcome_label.setAlignment(Qt.AlignCenter)
        welcome_label.setStyleSheet(
            "color: yellow; font-size: 16px; font-weight: bold;"
        )
        layout.addWidget(welcome_label)

        environments = self.load_environments(selected_platform)

        self.env_combo = QComboBox(self)
        self.env_combo.addItems(environments)
        self.env_combo.setStyleSheet(
            """
            QComboBox {
                background-color: #444444;
                color: white;
                font-size: 16px;
                padding: 0px;
                border: 1px solid white;
            }
       
        """
        )
        layout.addWidget(self.env_combo)
        self.env_combo.setFixedHeight(35)
        self.setLayout(layout)

    def load_environments(self, platform):
        try:
            with open("config/config_platform.yaml", "r") as file:
                config = yaml.safe_load(file)
                platforms = config.get("platforms", {})
                return platforms.get(platform, {}).get("environments", [])
        except FileNotFoundError:
            return []

    def open_platform_selection(self):
        self.close()
        self.platform_window()

    def confirm_selection(self):
        self.close()
        selected_env = self.env_combo.currentText()
        self.user_selections["selected_environment"] = selected_env

        self.select_alg_window = SelectAlgorithmWindow(
            self.show, selected_env, self.user_selections
        )
        self.select_alg_window.show()

    @staticmethod
    def apply_button_style(button):
        button.setStyleSheet(
            """
            QPushButton { 
            background-color: #444444; 
            color: white; 
            font-size: 16px; 
            padding: 10px 20px; 
            border-radius: 10px; 
            border: 1px solid white; 
            }
            QPushButton:hover { background-color: #555555; }
        """
        )

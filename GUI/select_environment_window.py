import yaml
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import (
    QVBoxLayout,
    QLabel,
    QComboBox,
    QHBoxLayout,
    QSpacerItem,
    QSizePolicy,
    QWidget,
)
from PyQt5.QtCore import Qt

from GUI.ui_base_window import BaseWindow
from GUI.ui_utils import create_button
from GUI.ui_styles import Styles
from GUI.training_window import TrainingWindow


class SelectEnvironmentWindow(BaseWindow):
    def __init__(self, platform_window, user_selections) -> None:  # noqa
        super().__init__("Select Environment", 900, 300)

        self.platform_window = platform_window
        self.user_selections = user_selections
        self.algorithm_selected = user_selections["Algorithms"]
        self.selected_platform = user_selections["selected_platform"]
        self.select_alg_window = None

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout()

        # Navigation buttons
        button_layout = QHBoxLayout()
        back_button = create_button(
            self,
            "Back",
            width=120,
            height=50,
            icon=QIcon("media_resources/icons/back.svg"),
        )
        back_button.clicked.connect(self.open_platform_selection)
        button_layout.addWidget(back_button)

        button_layout.addItem(
            QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)
        )

        next_button = create_button(self, "Next", width=120, height=50)
        next_button.clicked.connect(self.confirm_selection)
        button_layout.addWidget(next_button)

        layout.addLayout(button_layout)

        # Title label
        welcome_label = QLabel(
            f"Please select the environment for {self.selected_platform}", self
        )
        welcome_label.setAlignment(Qt.AlignCenter)
        welcome_label.setStyleSheet(Styles.WELCOME_LABEL)
        layout.addWidget(welcome_label)

        # Environment selection dropdown
        environments = self.load_environments(self.selected_platform)
        self.env_combo = QComboBox(self)
        self.env_combo.addItems(environments)
        self.env_combo.setStyleSheet(Styles.COMBO_BOX)
        layout.addWidget(self.env_combo)

        main_widget.setLayout(layout)

    def load_environments(self, platform: str) -> list:
        """Load available environments based on selected platform and algorithm.

        Args:
            platform: The selected platform name

        Returns:
            List of environment names
        """
        try:
            with open("config/config_platform.yaml", "r") as file:
                config = yaml.safe_load(file)
                platforms = config.get("platforms", {})
                if len(self.algorithm_selected) == 1:
                    if self.algorithm_selected[0]["Algorithm"] == "DQN":
                        return platforms.get(platform, {}).get(
                            "discrete_environments", []
                        )
                return platforms.get(platform, {}).get("environments", [])
        except FileNotFoundError:
            return []

    def open_platform_selection(self) -> None:
        """Return to platform selection screen."""
        self.close()
        self.platform_window()

    def confirm_selection(self) -> None:
        """Confirm environment selection and proceed to training window."""
        selected_env = self.env_combo.currentText()
        self.user_selections["selected_environment"] = selected_env
        self.close()

        self.select_alg_window = TrainingWindow(self.show, self.user_selections)
        self.select_alg_window.show()

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import (
    QLabel,
    QVBoxLayout,
    QHBoxLayout,
    QWidget,
)

from GUI.ui_base_window import BaseWindow
from GUI.ui_utils import create_button
from GUI.ui_styles import Styles
from GUI.select_algorithm_window import SelectAlgorithmWindow
from GUI.load_model_window import LoadConfigWindow


class WelcomeWindow(BaseWindow):
    def __init__(self) -> None:
        """Initialize the WelcomeWindow class."""
        super().__init__("RL Configuration Guide")
        self.load_config_window = None
        self.platform_config_window = None
        self.user_selections = {"setup_choice": ""}

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout()

        welcome_label = QLabel(
            "Welcome to the ReinforceUI Studio!! \n"
            "Easily configure RL environments, select algorithms, and monitor training. \n"
            " \n Press select one of the following options to get started:\n",
            self,
        )
        welcome_label.setWordWrap(True)
        welcome_label.setAlignment(Qt.AlignCenter)
        welcome_label.setStyleSheet(Styles.WELCOME_LABEL)
        main_layout.addWidget(welcome_label)

        # Button layout
        button_layout = QHBoxLayout()
        button_layout.setSpacing(25)

        manual_button = create_button(
            self,
            "Start Training Configuration",
            icon=QIcon("media_resources/icon_custom_config.png"),
        )

        load_button = create_button(
            self,
            "Load Pre-trained Model",
            icon=QIcon("media_resources/icon_config.png"),
        )

        button_layout.addWidget(manual_button)
        button_layout.addWidget(load_button)
        button_layout.setContentsMargins(100, 20, 100, 20)
        button_layout.setAlignment(Qt.AlignCenter)
        main_layout.addLayout(button_layout)

        main_widget.setLayout(main_layout)

        manual_button.clicked.connect(self.open_manual_configuration)
        load_button.clicked.connect(self.open_manual_configuration)

    def open_manual_configuration(self) -> None:
        """Open the manual configuration window."""
        if self.sender().text() == "Load Pre-trained Model":
            self.user_selections["setup_choice"] = "load_model"
            self.close()
            self.load_config_window = LoadConfigWindow(
                self.show, self.user_selections
            )
            self.load_config_window.show()
        else:
            self.user_selections["setup_choice"] = "train_model"
            self.close()
            self.platform_config_window = SelectAlgorithmWindow(
                self.show, self.user_selections
            )
            self.platform_config_window.show()

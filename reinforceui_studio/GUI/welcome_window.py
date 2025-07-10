from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import (
    QLabel,
    QVBoxLayout,
    QHBoxLayout,
    QWidget,
)

from reinforceui_studio.GUI.ui_base_window import BaseWindow
from reinforceui_studio.GUI.ui_utils import create_button, get_icon_path
from reinforceui_studio.GUI.ui_styles import Styles
from reinforceui_studio.GUI.select_algorithm_window import (
    SelectAlgorithmWindow,
)
from reinforceui_studio.GUI.select_multiple_algorithm_window import (
    SelectMultipleAlgorithmWindow,
)
from reinforceui_studio.GUI.load_model_window import LoadConfigWindow


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
            " \n Press select one of the following options to get started\n",
            self,
        )
        welcome_label.setWordWrap(True)
        welcome_label.setAlignment(Qt.AlignCenter)
        welcome_label.setStyleSheet(Styles.WELCOME_LABEL)
        main_layout.addWidget(welcome_label)

        # Button layout
        button_layout = QHBoxLayout()
        button_layout.setSpacing(10)

        manual_button = create_button(
            self,
            "Single Model Training",
            icon=QIcon(get_icon_path("single_icon.svg")),
        )

        comparative_button = create_button(
            self,
            "Multiple Models Training ",
            icon=QIcon(get_icon_path("comparative_icon.svg")),
        )

        load_button = create_button(
            self,
            "Load Pre-trained Model",
            icon=QIcon(get_icon_path("load_icon.svg")),
        )

        button_layout.addWidget(manual_button)
        button_layout.addWidget(comparative_button)
        button_layout.addWidget(load_button)
        button_layout.setContentsMargins(20, 20, 20, 20)
        button_layout.setAlignment(Qt.AlignCenter)
        main_layout.addLayout(button_layout)

        main_widget.setLayout(main_layout)

        manual_button.clicked.connect(self.open_single_manual_configuration)
        comparative_button.clicked.connect(self.open_comparative_configuration)
        load_button.clicked.connect(self.load_manual_configuration)

    def load_manual_configuration(self) -> None:
        """Open the load configuration window."""
        self.user_selections["setup_choice"] = "load_model"
        self.close()
        self.load_config_window = LoadConfigWindow(self.show, self.user_selections)
        self.load_config_window.show()

    def open_single_manual_configuration(self) -> None:
        """Open single manual configuration window."""
        self.user_selections["setup_choice"] = "single_train_model"
        self.close()
        self.platform_config_window = SelectAlgorithmWindow(
            self.show, self.user_selections
        )
        self.platform_config_window.show()

    def open_comparative_configuration(self) -> None:
        """Open comparative configuration window."""
        self.user_selections["setup_choice"] = "compare_model"
        self.close()
        self.platform_config_window = SelectMultipleAlgorithmWindow(
            self.show, self.user_selections
        )
        self.platform_config_window.show()

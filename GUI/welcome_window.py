from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import (
    QMainWindow,
    QPushButton,
    QLabel,
    QVBoxLayout,
    QHBoxLayout,
    QWidget,
    QDesktopWidget,
)
from PyQt5.QtCore import Qt
from GUI.select_algorithm_window import SelectAlgorithmWindow
from GUI.load_model_window import LoadConfigWindow


class WelcomeWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.load_config_window = None
        self.platform_config_window = None
        self.user_selections = {"setup_choice": ""}
        self.setWindowTitle("RL Configuration Guide")
        self.setFixedSize(900, 230)
        self.setStyleSheet("background-color: #121212;")
        self.center()

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
        welcome_label.setStyleSheet(
            "color: #E0E0E0; font-size: 18px; font-weight: bold;"
        )
        main_layout.addWidget(welcome_label)

        button_layout = QHBoxLayout()
        button_layout.setSpacing(25)  # Increase the separation between buttons
        manual_button = self.create_button(
            "Start Training Configuration",
            250,
            50,
            QIcon("media_resources/icon_custom_config.png"),
        )
        load_button = self.create_button(
            "Load Pre-trained Model", 250, 50, QIcon("media_resources/icon_config.png")
        )

        button_layout.addWidget(manual_button)
        button_layout.addWidget(load_button)
        button_layout.setContentsMargins(100, 20, 100, 20)
        button_layout.setAlignment(Qt.AlignCenter)
        main_layout.addLayout(button_layout)

        main_widget.setLayout(main_layout)

        manual_button.clicked.connect(self.open_manual_configuration)
        load_button.clicked.connect(self.open_manual_configuration)

    def create_button(self, text, width, height, icon=None):
        button = QPushButton(text, self)
        button.setFixedSize(width, height)
        if icon:
            button.setIcon(icon)
            button.setStyleSheet("background-color: transparent;")
        button.setStyleSheet(
            """
            QPushButton {
                background-color: #444444;
                color: white;
                font-size: 15px;
                padding: 10px 20px;
                border-radius: 10px;
                border: 1px solid white;
            }
            QPushButton:hover {
                background-color: #555555;
            }
        """
        )
        return button

    def open_manual_configuration(self):
        if self.sender().text() == "Load Pre-trained Model":
            self.user_selections["setup_choice"] = "load_model"
            self.close()
            self.load_config_window = LoadConfigWindow(self.show, self.user_selections)
            self.load_config_window.show()
        else:
            self.user_selections["setup_choice"] = "train_model"
            self.close()
            self.platform_config_window = SelectAlgorithmWindow(
                self.show, self.user_selections
            )
            self.platform_config_window.show()

    def center(self):
        screen_geometry = QDesktopWidget().availableGeometry().center()
        frame_geometry = self.frameGeometry()
        frame_geometry.moveCenter(screen_geometry)
        self.move(frame_geometry.topLeft())

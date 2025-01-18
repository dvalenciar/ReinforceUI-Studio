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

from GUI.select_platform_window import PlatformConfigWindow


class WelcomeWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.manual_config_window = None
        self.user_selections = {"setup_choice": ""}
        self.setWindowTitle("RL Configuration Guide")
        self.setFixedSize(700, 200)
        self.center()

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout()

        welcome_label = QLabel(
            "Welcome to the ReinforceUI Studio!! \n"
            "Easily configure RL environments, select algorithms, and monitor training. \n",
            self,
        )

        welcome_label.setWordWrap(True)
        welcome_label.setAlignment(Qt.AlignCenter)
        welcome_label.setStyleSheet(
            "color: yellow; font-size: 18px; font-weight: bold;"
        )
        main_layout.addWidget(welcome_label)

        button_layout = QHBoxLayout()
        manual_button = self.create_button("Start Configuration", 250, 50)
        # config_button = self.create_button("", 50, 50, QIcon('media_resources/icon_config.png'))

        button_layout.addWidget(manual_button)
        button_layout.setContentsMargins(100, 20, 100, 20)
        button_layout.setAlignment(Qt.AlignCenter)
        main_layout.addLayout(button_layout)

        main_widget.setLayout(main_layout)
        self.setStyleSheet("background-color: black;")

        manual_button.clicked.connect(self.open_manual_configuration)

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
                font-size: 16px;
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
        self.user_selections["setup_choice"] = "manual"
        self.close()
        self.manual_config_window = PlatformConfigWindow(
            self.show, self.user_selections
        )
        self.manual_config_window.show()

    def center(self):
        screen_geometry = QDesktopWidget().availableGeometry().center()
        frame_geometry = self.frameGeometry()
        frame_geometry.moveCenter(screen_geometry)
        self.move(frame_geometry.topLeft())

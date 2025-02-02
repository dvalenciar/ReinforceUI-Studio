from PyQt5.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QLabel,
    QPushButton,
    QHBoxLayout,
    QSpacerItem,
    QSizePolicy,
    QMessageBox,
)
from PyQt5.QtGui import QMovie, QPixmap
from PyQt5.QtCore import Qt
from GUI.select_environment_window import SelectEnvironmentWindow


class PlatformConfigWindow(QDialog):
    def __init__(self, main_window, user_selections):
        super().__init__()

        self.main_window = main_window
        self.user_selections = user_selections
        self.setWindowTitle("RL Training Platform Selection")
        self.setFixedSize(1000, 430)
        self.setStyleSheet("background-color: #121212;")

        main_layout = QVBoxLayout(self)
        buttons_layout = QHBoxLayout()

        back_button = QPushButton("Back", self)
        back_button.setFixedSize(100, 35)
        self.apply_button_style(back_button)
        back_button.clicked.connect(self.open_main_window)
        buttons_layout.addWidget(back_button, alignment=Qt.AlignLeft)

        buttons_layout.addItem(
            QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)
        )

        next_button = QPushButton("Next", self)
        next_button.setFixedSize(100, 35)
        self.apply_button_style(next_button)
        next_button.clicked.connect(self.open_select_environment)
        buttons_layout.addWidget(next_button, alignment=Qt.AlignRight)

        main_layout.addLayout(buttons_layout)

        welcome_label = QLabel("Select the RL Platform.", self)
        welcome_label.setAlignment(Qt.AlignCenter)
        welcome_label.setStyleSheet(
            "color: #E0E0E0; font-size: 18px; font-weight: bold;"
        )
        main_layout.addWidget(welcome_label)

        platforms_layout = QHBoxLayout()
        spacer = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)
        platforms_layout.addItem(spacer)

        platforms = [
            {
                "name": "Gymnasium",
                "gif": "media_resources/pendulum.gif",
                "is_gif": True,
            },
            {"name": "DMCS", "gif": "media_resources/cheetah_run.gif", "is_gif": True},
            {"name": "MuJoCo", "gif": "media_resources/half_cheetah", "is_gif": True},
        ]

        self.selected_button = None

        for platform in platforms:
            v_layout = QVBoxLayout()
            gif_label = QLabel(self)
            if platform["is_gif"]:
                movie = QMovie(platform["gif"])
                gif_label.setMovie(movie)
                movie.start()
            else:
                pixmap = QPixmap(platform["gif"])
                gif_label.setPixmap(pixmap)
                gif_label.setScaledContents(True)

            platform_button = QPushButton(platform["name"], self)
            platform_button.setFixedSize(150, 50)
            self.apply_button_style(platform_button)
            platform_button.clicked.connect(
                lambda checked, b=platform_button: self.handle_button_click(b)
            )

            v_layout.addWidget(gif_label, alignment=Qt.AlignCenter)
            v_layout.addWidget(platform_button, alignment=Qt.AlignCenter)
            platforms_layout.addLayout(v_layout)

        platforms_layout.addItem(spacer)
        main_layout.addLayout(platforms_layout)

    def handle_button_click(self, button):
        if self.selected_button:
            self.apply_button_style(self.selected_button)
        self.selected_button = button
        self.apply_selected_button_style(button)

    def open_main_window(self):
        self.close()
        self.main_window()

    def open_select_environment(self):
        if self.selected_button:
            selected_platform = self.selected_button.text()
            self.user_selections["selected_platform"] = selected_platform
            self.close()
            self.select_env_window = SelectEnvironmentWindow(
                self.show, selected_platform, self.user_selections
            )
            self.select_env_window.show()
        else:
            msg_box = QMessageBox(self)
            msg_box.setIcon(QMessageBox.Warning)
            msg_box.setWindowTitle("Selection Required")
            msg_box.setText("Please select a platform before proceeding.")
            msg_box.setStyleSheet(
                """
                QMessageBox { background-color: black; color: white; }
                QMessageBox QLabel { color: white; }
                QMessageBox QPushButton { background-color: #444444; color: white; font-size: 14px; padding: 5px 15px; border-radius: 5px; border: 1px solid white; }
                QMessageBox QPushButton:hover { background-color: #555555; }
            """
            )
            msg_box.setStandardButtons(QMessageBox.Ok)
            msg_box.exec_()

    @staticmethod
    def apply_button_style(button):
        button.setStyleSheet(
            """
            QPushButton { background-color: #444444; color: white; font-size: 14px; padding: 7px 15px; border-radius: 10px; border: 1px solid white; }
            QPushButton:hover { background-color: #555555; }
        """
        )

    @staticmethod
    def apply_selected_button_style(button):
        button.setStyleSheet(
            """
            QPushButton { background-color: #2a9d8f; color: white; font-size: 16px; padding: 10px 20px; border-radius: 10px; border: 2px solid white; }
        """
        )

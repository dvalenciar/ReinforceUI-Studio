from PyQt5.QtWidgets import (
    QVBoxLayout,
    QLabel,
    QHBoxLayout,
    QSpacerItem,
    QSizePolicy,
    QMessageBox,
    QWidget,
)
from PyQt5.QtGui import QMovie, QPixmap, QIcon
from PyQt5.QtCore import Qt

from GUI.ui_base_window import BaseWindow
from GUI.ui_utils import create_button
from GUI.ui_styles import Styles
from GUI.select_environment_window import SelectEnvironmentWindow


class PlatformConfigWindow(BaseWindow):
    def __init__(self, algorithm_window, user_selections) -> None:  # noqa
        super().__init__("RL Training Platform Selection", 900, 400)

        self.algorithm_window = algorithm_window
        self.user_selections = user_selections
        self.algorithm_selected = user_selections["Algorithms"]
        self.selected_button = None
        self.select_env_window = None

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout()

        # Navigation buttons
        buttons_layout = QHBoxLayout()

        back_button = create_button(
            self,
            "Back",
            width=120,
            height=50,
            icon=QIcon("media_resources/icons/back.svg"),
        )
        back_button.clicked.connect(self.open_algorithm_window)
        buttons_layout.addWidget(back_button, alignment=Qt.AlignLeft)

        buttons_layout.addItem(
            QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)
        )

        next_button = create_button(self, "Next", width=100, height=50)
        next_button.clicked.connect(self.open_select_environment)
        buttons_layout.addWidget(next_button, alignment=Qt.AlignRight)

        main_layout.addLayout(buttons_layout)

        # Title label
        welcome_label = QLabel("Select the RL Platform.", self)
        welcome_label.setAlignment(Qt.AlignCenter)
        welcome_label.setStyleSheet(Styles.WELCOME_LABEL)
        main_layout.addWidget(welcome_label)

        # Platforms layout
        platforms_layout = QHBoxLayout()
        spacer = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)
        platforms_layout.addItem(spacer)

        # Define available platforms
        platforms = [
            {
                "name": "Gymnasium",
                "gif": "media_resources/pendulum.gif",
                "is_gif": True,
            },
        ]

        if len(self.algorithm_selected) == 1:
            selected_algo_name = self.algorithm_selected[0]["Algorithm"]
            if selected_algo_name != "DQN":
                platforms.insert(
                    1,
                    {
                        "name": "DMCS",
                        "gif": "media_resources/cheetah_run.gif",
                        "is_gif": True,
                    },
                )
                platforms.insert(
                    2,
                    {
                        "name": "MuJoCo",
                        "gif": "media_resources/half_cheetah.gif",
                        "is_gif": True,
                    },
                )
        else:
            platforms.insert(
                1,
                {
                    "name": "DMCS",
                    "gif": "media_resources/cheetah_run.gif",
                    "is_gif": True,
                },
            )
            platforms.insert(
                2,
                {
                    "name": "MuJoCo",
                    "gif": "media_resources/half_cheetah.gif",
                    "is_gif": True,
                },
            )

        # Create platform buttons
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

            platform_button = create_button(
                self, platform["name"], width=150, height=50
            )
            platform_button.clicked.connect(
                lambda checked, b=platform_button: self.handle_button_click(b)
            )

            v_layout.addWidget(gif_label, alignment=Qt.AlignCenter)
            v_layout.addWidget(platform_button, alignment=Qt.AlignCenter)
            platforms_layout.addLayout(v_layout)

        platforms_layout.addItem(spacer)
        main_layout.addLayout(platforms_layout)
        main_widget.setLayout(main_layout)

    def handle_button_click(self, button) -> None:  # noqa
        """Handle platform button selection.

        Args:
            button: Selected platform button

        """
        if self.selected_button:
            self.selected_button.setStyleSheet(Styles.BUTTON)
        self.selected_button = button
        button.setStyleSheet(Styles.SELECTED_BUTTON)

    def open_algorithm_window(self) -> None:
        """Return to algorithm selection screen."""
        self.close()
        self.algorithm_window()

    def open_select_environment(self) -> None:
        """Proceed to environment selection screen."""
        if self.selected_button:
            selected_platform = self.selected_button.text()
            self.user_selections["selected_platform"] = selected_platform
            self.close()
            self.select_env_window = SelectEnvironmentWindow(
                self.show, self.user_selections
            )
            self.select_env_window.show()
        else:
            self._show_selection_required_warning()

    def _show_selection_required_warning(self) -> None:
        """Display warning when no platform is selected."""
        msg_box = QMessageBox(self)
        msg_box.setIcon(QMessageBox.Warning)
        msg_box.setWindowTitle("Selection Required")
        msg_box.setText("Please select a platform before proceeding.")
        msg_box.setStyleSheet(Styles.MESSAGE_BOX)
        msg_box.setStandardButtons(QMessageBox.Ok)
        msg_box.exec_()

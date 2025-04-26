from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import (
    QVBoxLayout,
    QLabel,
    QComboBox,
    QHBoxLayout,
    QSpacerItem,
    QSizePolicy,
    QMessageBox,
    QWidget,
)
from PyQt5.QtCore import Qt
import yaml

from GUI.ui_base_window import BaseWindow
from GUI.ui_utils import create_button
from GUI.ui_styles import Styles
from GUI.select_hyperparameters_window import SelectHyperWindow
from GUI.select_platform_window import PlatformConfigWindow


class SelectAlgorithmWindow(BaseWindow):
    def __init__(self, welcome_window, user_selections) -> None:  # noqa
        """Initialize the SelectAlgorithmWindow class"""
        super().__init__("Select Algorithm", 900, 300)

        self.welcome_window = welcome_window
        self.user_selections = user_selections
        self.selected_algorithm = None
        self.custom_window = None
        self.platform_window = None
        self.use_default_hyperparameters = None
        self.custom_hyperparameters = {}

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout()

        # Navigation buttons (Back/Next)
        button_layout = QHBoxLayout()

        back_button = create_button(
            self,
            "Back",
            width=120,
            height=50,
            icon=QIcon("media_resources/icons/back.svg"),
        )
        back_button.clicked.connect(self.open_welcome_window)
        button_layout.addWidget(back_button)

        button_layout.addItem(
            QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)
        )

        next_button = create_button(
            self,
            "Next",
            width=120,
            height=50,
        )
        next_button.clicked.connect(self.confirm_selection)
        button_layout.addWidget(next_button)

        layout.addLayout(button_layout)

        # Title label
        welcome_label = QLabel("Select Algorithm", self)
        welcome_label.setAlignment(Qt.AlignCenter)
        welcome_label.setStyleSheet(Styles.WELCOME_LABEL)
        layout.addWidget(welcome_label)

        # Algorithm selection dropdown
        algorithms = self.load_algorithms()
        self.algo_combo = QComboBox(self)
        self.algo_combo.addItems(algorithms)
        self.algo_combo.setStyleSheet(Styles.COMBO_BOX)
        layout.addWidget(self.algo_combo)
        self.algo_combo.setFixedHeight(35)

        # Hyperparameter selection section
        hyperparam_label = QLabel(
            "Would you like to use the default hyperparameters?", self
        )
        hyperparam_label.setAlignment(Qt.AlignCenter)
        hyperparam_label.setStyleSheet(Styles.TEXT_LABEL)
        layout.addWidget(hyperparam_label)

        button_layout_hyperparams = QHBoxLayout()

        self.yes_button = create_button(self, "Yes", width=270, height=50)
        self.yes_button.clicked.connect(self.use_default_hyperparams)
        button_layout_hyperparams.addWidget(self.yes_button)

        self.custom_button = create_button(
            self,
            "Custom",
            width=270,
            height=50,
            icon=QIcon("media_resources/icons/config.svg"),
        )
        self.custom_button.clicked.connect(self.open_custom_hyperparams_window)
        button_layout_hyperparams.addWidget(self.custom_button)

        layout.addLayout(button_layout_hyperparams)
        main_widget.setLayout(layout)

    def load_algorithms(self) -> list:
        """Load algorithm choices from configuration file."""
        try:
            with open("config/config_algorithm.yaml", "r") as file:
                config = yaml.safe_load(file)
                return [algo["name"] for algo in config.get("algorithms", [])]
        except FileNotFoundError:
            return []

    def use_default_hyperparams(self) -> None:
        """Handle user choosing default hyperparameters."""
        self.use_default_hyperparameters = True
        self.set_active_button(self.yes_button, self.custom_button)

    def open_custom_hyperparams_window(self) -> None:
        """Open window for custom hyperparameter configuration."""
        self.use_default_hyperparameters = False
        selected_algorithm = self.algo_combo.currentText()
        self.set_active_button(self.custom_button, self.yes_button)
        self.custom_window = SelectHyperWindow(
            selected_algorithm, self.set_custom_hyperparameters
        )
        self.custom_window.show()

    def set_custom_hyperparameters(self, hyperparameters: dict) -> None:
        """Store custom hyperparameters after user selection.

        Args:
            hyperparameters (dict): User-selected hyperparameters
        """
        # self.user_selections["Hyperparameters"] = hyperparameters
        self.custom_hyperparameters = hyperparameters

    def open_welcome_window(self) -> None:
        """Return to initial welcome screen."""
        self.close()
        self.welcome_window()

    def confirm_selection(self) -> None:
        """Process user selection and advance to next screen."""
        if self.use_default_hyperparameters is None:
            self._show_selection_required_warning()
            return

        # Set algorithm selection in user_selections
        selection = []
        selected_algo = self.algo_combo.currentText()

        # If using default hyperparameters, load them from config
        if self.use_default_hyperparameters:
            try:
                with open("config/config_algorithm.yaml", "r") as file:
                    config = yaml.safe_load(file)
                    algorithms = config.get("algorithms", [])
                    for algo in algorithms:
                        if algo["name"] == selected_algo:
                            hyperparams = algo.get("hyperparameters", {})
                            break
            except FileNotFoundError:
                hyperparams = {}
        else:
            hyperparams = self.custom_hyperparameters

        selection.append(
            {
                "Algorithm": selected_algo,
                "Hyperparameters": hyperparams,
            }
        )

        self.user_selections["Algorithms"] = selection
        self.close()
        self.platform_window = PlatformConfigWindow(self.show, self.user_selections)
        self.platform_window.show()

    def _show_selection_required_warning(self) -> None:
        """Display warning dialog when hyperparameter choice is required."""
        msg_box = QMessageBox(self)
        msg_box.setIcon(QMessageBox.Warning)
        msg_box.setWindowTitle("Selection Required")
        msg_box.setText(
            "Please select an option for hyperparameters before proceeding."
        )
        msg_box.setStyleSheet(Styles.MESSAGE_BOX)
        msg_box.setStandardButtons(QMessageBox.Ok)
        msg_box.exec_()

    # todo this is redundant with the one in update_button_styles in training_window.py, remove it or make it a common function
    def set_active_button(self, active_button, inactive_button) -> None:  # noqa
        """Visually highlight the selected option button."""
        active_button.setStyleSheet(Styles.SELECTED_BUTTON)
        inactive_button.setStyleSheet(Styles.BUTTON)

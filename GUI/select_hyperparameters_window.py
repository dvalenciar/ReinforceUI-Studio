import yaml
from PyQt5.QtWidgets import (
    QVBoxLayout,
    QLabel,
    QLineEdit,
    QHBoxLayout,
    QSpacerItem,
    QSizePolicy,
    QMessageBox,
    QWidget,
)
from PyQt5.QtCore import Qt

from GUI.ui_base_window import BaseWindow
from GUI.ui_utils import create_button
from GUI.ui_styles import Styles


class SelectHyperWindow(BaseWindow):
    def __init__(self, selected_algorithm, callback) -> None:  # noqa
        super().__init__(f"Hyperparameters for {selected_algorithm}", 400, 800)

        self.selected_algorithm = selected_algorithm
        self.callback = callback
        self.hyperparameters = {}
        self.default_hyperparameters = {}
        self.hyperparam_fields = {}

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout()

        # Title label
        title_label = QLabel(f"Custom Hyperparameters for {selected_algorithm}", self)
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet(Styles.WELCOME_LABEL)
        layout.addWidget(title_label)

        # Load hyperparameters
        self.load_hyperparameters(selected_algorithm)

        # Create input fields for each hyperparameter
        for param_name, default_value in self.default_hyperparameters.items():
            param_label = QLabel(param_name, self)
            param_label.setStyleSheet(Styles.TEXT_LABEL)
            layout.addWidget(param_label)

            param_input = QLineEdit(str(default_value), self)
            param_input.setStyleSheet(Styles.LINE_EDIT)
            layout.addWidget(param_input)
            self.hyperparam_fields[param_name] = param_input

        # Action buttons
        button_layout = QHBoxLayout()

        reset_button = create_button(self, "Reset", width=120, height=50)
        reset_button.clicked.connect(self.reset_to_defaults)
        button_layout.addWidget(reset_button)

        button_layout.addItem(
            QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)
        )

        confirm_button = create_button(self, "Confirm", width=120, height=50)
        confirm_button.clicked.connect(self.confirm_changes)
        button_layout.addWidget(confirm_button)

        layout.addLayout(button_layout)
        main_widget.setLayout(layout)

    def load_hyperparameters(self, algorithm_name: str) -> None:
        """Load the default hyperparameters for the selected algorithm.

        Args:
            algorithm_name: Name of the selected algorithm

        """
        try:
            with open("config/config_algorithm.yaml", "r") as file:
                config = yaml.safe_load(file)
                algorithms = config.get("algorithms", [])
                for algo in algorithms:
                    if algo["name"] == algorithm_name:
                        self.default_hyperparameters = algo.get("hyperparameters", {})
                        self.hyperparameters = self.default_hyperparameters.copy()
                        break
        except FileNotFoundError:
            print("Algorithm config file not found.")
            self.default_hyperparameters = {}

    def confirm_changes(self) -> None:
        """Confirm and save the hyperparameter changes."""
        confirm_dialog = QMessageBox(self)
        confirm_dialog.setIcon(QMessageBox.Warning)
        confirm_dialog.setWindowTitle("Confirm Changes")
        confirm_dialog.setText("Are you sure you want these hyperparameters?")
        confirm_dialog.setStyleSheet(Styles.MESSAGE_BOX)
        confirm_dialog.setStandardButtons(QMessageBox.Yes | QMessageBox.No)

        if confirm_dialog.exec_() == QMessageBox.Yes:
            for param_name, input_field in self.hyperparam_fields.items():
                self.hyperparameters[param_name] = input_field.text()
            self.callback(self.hyperparameters)
            self.close()
        else:
            print("Hyperparameter changes were not confirmed.")

    def reset_to_defaults(self) -> None:
        """Reset all hyperparameter inputs to default values."""
        for param_name, default_value in self.default_hyperparameters.items():
            input_field = self.hyperparam_fields[param_name]
            input_field.setText(str(default_value))

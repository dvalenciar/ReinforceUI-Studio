import yaml
from PyQt5.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QHBoxLayout,
    QSpacerItem,
    QSizePolicy,
    QMessageBox,
)
from PyQt5.QtCore import Qt


class SelectHyperWindow(QDialog):
    def __init__(self, selected_algorithm, callback):
        super().__init__()
        self.selected_algorithm = selected_algorithm
        self.callback = callback
        self.hyperparameters = {}
        self.default_hyperparameters = {}
        self.hyperparam_fields = {}

        self.setWindowTitle(f"Hyperparameters for {selected_algorithm}")
        self.setStyleSheet("background-color: #121212;")

        layout = QVBoxLayout()

        title_label = QLabel(f"Custom Hyperparameters for {selected_algorithm}", self)
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("color: #E0E0E0; font-size: 16px; font-weight: bold;")
        layout.addWidget(title_label)

        self.load_hyperparameters(selected_algorithm)

        for param_name, default_value in self.default_hyperparameters.items():
            param_label = QLabel(param_name, self)
            param_label.setStyleSheet("color: white; font-size: 14px;")
            layout.addWidget(param_label)

            param_input = QLineEdit(str(default_value), self)
            param_input.setStyleSheet(
                """
                QLineEdit {
                    background-color: #444444;
                    color: white;
                    font-size: 16px;
                    padding: 5px;
                    border: 1px solid white;
                }
            """
            )
            layout.addWidget(param_input)
            self.hyperparam_fields[param_name] = param_input

        button_layout = QHBoxLayout()

        reset_button = QPushButton("Reset", self)
        reset_button.setFixedSize(120, 40)
        self.apply_button_style(reset_button)
        reset_button.clicked.connect(self.reset_to_defaults)
        button_layout.addWidget(reset_button)

        button_layout.addItem(
            QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)
        )

        confirm_button = QPushButton("Confirm", self)
        confirm_button.setFixedSize(120, 40)
        self.apply_button_style(confirm_button)
        confirm_button.clicked.connect(self.confirm_changes)
        button_layout.addWidget(confirm_button)

        layout.addLayout(button_layout)
        self.setLayout(layout)

    def load_hyperparameters(self, algorithm_name):
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

    def confirm_changes(self):
        confirm_dialog = QMessageBox(self)
        confirm_dialog.setIcon(QMessageBox.Warning)
        confirm_dialog.setWindowTitle("Confirm Changes")
        confirm_dialog.setText("Are you sure you want these hyperparameters?")
        confirm_dialog.setStyleSheet(
            """
            QMessageBox {
                background-color: black;
                color: white;
            }
            QMessageBox QLabel {
                color: white;
            }
            QMessageBox QPushButton {
                background-color: #444444;
                color: white;
                font-size: 14px;
                padding: 5px 15px;
                border-radius: 5px;
                border: 1px solid white;
            }
            QMessageBox QPushButton:hover {
                background-color: #555555;
            }
        """
        )
        confirm_dialog.setStandardButtons(QMessageBox.Yes | QMessageBox.No)

        if confirm_dialog.exec_() == QMessageBox.Yes:
            for param_name, input_field in self.hyperparam_fields.items():
                self.hyperparameters[param_name] = input_field.text()

            print(
                f"Confirmed hyperparameters for {self.selected_algorithm}: {self.hyperparameters}"
            )
            self.callback(self.hyperparameters)
            self.close()
        else:
            print("Hyperparameter changes were not confirmed.")

    def reset_to_defaults(self):
        for param_name, default_value in self.default_hyperparameters.items():
            input_field = self.hyperparam_fields[param_name]
            input_field.setText(str(default_value))

        print(f"Reset hyperparameters for {self.selected_algorithm} to defaults.")

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
            QPushButton:hover {
                background-color: #555555;
            }
        """
        )

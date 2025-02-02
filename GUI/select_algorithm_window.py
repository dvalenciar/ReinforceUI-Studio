from PyQt5.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QLabel,
    QComboBox,
    QPushButton,
    QHBoxLayout,
    QSpacerItem,
    QSizePolicy,
    QMessageBox,
)
from PyQt5.QtCore import Qt
import yaml

from GUI.select_hyper_window import SelectHyperWindow
from GUI.training_window import TrainingWindow


class SelectAlgorithmWindow(QDialog):
    def __init__(self, environment_window, selected_env, user_selections):
        super().__init__()

        self.selected_env = selected_env
        self.environment_window = environment_window
        self.user_selections = user_selections

        self.selected_algorithm = None
        self.use_default_hyperparameters = None

        self.setWindowTitle("Select Algorithm")
        self.setFixedSize(500, 300)
        self.setStyleSheet("background-color: #121212;")

        layout = QVBoxLayout()
        button_layout = QHBoxLayout()

        back_button = QPushButton("Back", self)
        back_button.setFixedSize(100, 35)
        self.apply_button_style(back_button)
        back_button.clicked.connect(self.open_environment_selection)
        button_layout.addWidget(back_button)

        button_layout.addItem(
            QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)
        )

        next_button = QPushButton("Next", self)
        next_button.setFixedSize(100, 35)
        self.apply_button_style(next_button)
        next_button.clicked.connect(self.confirm_selection)
        button_layout.addWidget(next_button)

        layout.addLayout(button_layout)

        welcome_label = QLabel("Select Algorithm", self)
        welcome_label.setAlignment(Qt.AlignCenter)
        welcome_label.setStyleSheet(
            "color: #E0E0E0; font-size: 16px; font-weight: bold;"
        )
        layout.addWidget(welcome_label)

        algorithms = self.load_algorithms()
        self.algo_combo = QComboBox(self)
        self.algo_combo.addItems(algorithms)
        self.algo_combo.setStyleSheet(
            """
            QComboBox {
                background-color: #444444;
                color: white;
                font-size: 16px;
                padding: 0px;
                border: 1px solid white;
            }
        """
        )
        layout.addWidget(self.algo_combo)
        self.algo_combo.setFixedHeight(35)

        hyperparam_label = QLabel(
            "Would you like to use the default hyperparameters?", self
        )
        hyperparam_label.setAlignment(Qt.AlignCenter)
        hyperparam_label.setStyleSheet("color: white; font-size: 14px;")
        layout.addWidget(hyperparam_label)

        button_layout_hyperparams = QHBoxLayout()

        self.yes_button = QPushButton("Yes", self)
        self.yes_button.setFixedSize(120, 40)
        self.apply_button_style(self.yes_button)
        self.yes_button.clicked.connect(self.use_default_hyperparams)
        button_layout_hyperparams.addWidget(self.yes_button)

        self.custom_button = QPushButton("Custom", self)
        self.custom_button.setFixedSize(120, 40)
        self.apply_button_style(self.custom_button)
        self.custom_button.clicked.connect(self.open_custom_hyperparams_window)
        button_layout_hyperparams.addWidget(self.custom_button)

        layout.addLayout(button_layout_hyperparams)
        self.setLayout(layout)

    def load_algorithms(self):
        try:
            with open("config/config_algorithm.yaml", "r") as file:
                config = yaml.safe_load(file)
                return [algo["name"] for algo in config.get("algorithms", [])]
        except FileNotFoundError:
            return []

    def use_default_hyperparams(self):
        self.use_default_hyperparameters = True
        self.set_active_button(self.yes_button, self.custom_button)

    def open_custom_hyperparams_window(self):
        self.use_default_hyperparameters = False
        selected_algorithm = self.algo_combo.currentText()
        self.set_active_button(self.custom_button, self.yes_button)
        self.custom_window = SelectHyperWindow(
            selected_algorithm, self.set_custom_hyperparameters
        )
        self.custom_window.show()

    def set_custom_hyperparameters(self, hyperparameters):
        self.user_selections["Hyperparameters"] = hyperparameters

    def open_environment_selection(self):
        self.close()
        self.environment_window()

    def confirm_selection(self):
        if self.use_default_hyperparameters is None:
            msg_box = QMessageBox(self)
            msg_box.setIcon(QMessageBox.Warning)
            msg_box.setWindowTitle("Selection Required")
            msg_box.setText(
                "Please select an option for hyperparameters before proceeding."
            )
            msg_box.setStyleSheet(
                """
                QMessageBox { background-color: black; color: white; }
                QMessageBox QLabel { color: white; }
                QMessageBox QPushButton {
                        background-color: #444444; 
                        color: white; font-size: 14px; 
                        padding: 5px 15px; 
                        border-radius: 5px; 
                        border: 1px solid white; 
                        }
                QMessageBox QPushButton:hover { background-color: #555555; }
            """
            )
            msg_box.setStandardButtons(QMessageBox.Ok)
            msg_box.exec_()

        elif self.use_default_hyperparameters:
            selected_algo = self.algo_combo.currentText()
            try:
                with open("config/config_algorithm.yaml", "r") as file:
                    config = yaml.safe_load(file)
                    algorithms = config.get("algorithms", [])
                    for algo in algorithms:
                        if algo["name"] == selected_algo:
                            self.user_selections["Hyperparameters"] = algo.get(
                                "hyperparameters", {}
                            )
                            break
            except FileNotFoundError:
                self.user_selections["Hyperparameters"] = {}
            self.user_selections["Algorithm"] = selected_algo
            self.close()
            self.train_window = TrainingWindow(self.show, self.user_selections)
            self.train_window.show()
        else:
            self.user_selections["Algorithm"] = self.algo_combo.currentText()
            self.close()
            self.train_window = TrainingWindow(self.show, self.user_selections)
            self.train_window.show()

    def set_active_button(self, active_button, inactive_button):
        self.apply_selected_button_style(active_button)
        self.apply_button_style(inactive_button)

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

    @staticmethod
    def apply_selected_button_style(button):
        button.setStyleSheet(
            """
            QPushButton {
                background-color: #2a9d8f;
                color: white;
                font-size: 16px;
                padding: 10px 20px;
                border-radius: 10px;
                border: 2px solid white;
            }
        """
        )

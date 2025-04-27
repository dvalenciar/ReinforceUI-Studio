from typing import Callable, Dict, List, Union
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
    QScrollArea,
)
from PyQt5.QtCore import Qt
import yaml

from GUI.ui_base_window import BaseWindow
from GUI.ui_utils import create_button
from GUI.ui_styles import Styles
from GUI.select_hyperparameters_window import SelectHyperWindow
from GUI.select_platform_window import PlatformConfigWindow


class SelectMultipleAlgorithmWindow(BaseWindow):
    def __init__(
        self,
        welcome_window: Callable[[], None],
        user_selections: Dict[str, Union[str, List[Dict[str, Union[str, Dict]]]]],
    ) -> None:
        """Initializes the SelectMultipleAlgorithmWindow.

        Args:
            welcome_window (Callable[[], None]): Function to open the welcome window.
            user_selections (Dict[str, Union[str, List[Dict[str, Union[str, Dict]]]]]): Dictionary to store user selections.
        """
        super().__init__("Select Algorithms", 900, 500)

        self.welcome_window = welcome_window
        self.user_selections = user_selections
        self.selection_rows = []
        self.custom_hyperparams = {}
        self.available_algorithms = self.load_algorithms()

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        self.main_layout = QVBoxLayout()

        # Navigation buttons
        nav_layout = QHBoxLayout()
        back_button = create_button(
            self,
            "Back",
            width=120,
            height=50,
            icon=QIcon("media_resources/icons/back.svg"),
        )
        back_button.clicked.connect(self.open_welcome_window)
        nav_layout.addWidget(back_button)
        nav_layout.addItem(
            QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)
        )
        next_button = create_button(self, "Next", width=120, height=50)
        next_button.clicked.connect(self.confirm_selection)
        nav_layout.addWidget(next_button)
        self.main_layout.addLayout(nav_layout)

        # Title
        title = QLabel("Select Algorithms to Compare", self)
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet(Styles.WELCOME_LABEL)
        self.main_layout.addWidget(title)

        # Scroll area for dynamic content
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        self.scroll_widget = QWidget()
        self.algo_selection_layout = QVBoxLayout()

        self.scroll_widget.setLayout(self.algo_selection_layout)
        scroll_area.setWidget(self.scroll_widget)
        self.main_layout.addWidget(scroll_area)

        # Add default two selections
        self.add_algorithm_selection()
        self.add_algorithm_selection()

        # Add/Remove controls
        controls_layout = QHBoxLayout()
        add_button = create_button(self, "Add Algorithm", width=200, height=50)
        add_button.clicked.connect(self.add_algorithm_selection)
        controls_layout.addWidget(add_button)
        self.main_layout.addLayout(controls_layout)

        main_widget.setLayout(self.main_layout)

    def add_algorithm_selection(self) -> None:
        """Adds a new algorithm selection row to the UI."""
        row_widget = QWidget()
        row_layout = QVBoxLayout()

        # ComboBox
        algo_combo = QComboBox()
        algo_combo.addItems(self.available_algorithms)
        algo_combo.setStyleSheet(Styles.COMBO_BOX)
        algo_combo.setFixedHeight(40)
        row_layout.addWidget(algo_combo)

        # Label
        hyperparam_label = QLabel("Use default hyperparameters?", self)
        hyperparam_label.setAlignment(Qt.AlignCenter)
        hyperparam_label.setStyleSheet(Styles.TEXT_LABEL)
        row_layout.addWidget(hyperparam_label)

        # Yes/Custom buttons row
        button_row = QHBoxLayout()

        yes_button = create_button(self, "Yes", width=270, height=40)
        custom_button = create_button(
            self,
            "Custom",
            width=270,
            height=40,
            icon=QIcon("media_resources/icons/config.svg"),
        )

        yes_button.setStyleSheet(Styles.SELECTED_BUTTON)
        custom_button.setStyleSheet(Styles.BUTTON)

        yes_button.clicked.connect(
            lambda _, b=yes_button, c=custom_button: self.set_hyper_choice(
                row_widget, True, b, c
            )
        )
        custom_button.clicked.connect(
            lambda _, b=custom_button, c=yes_button, combo=algo_combo: self.open_custom_hyperparams(
                combo.currentText(), row_widget, b, c
            )
        )

        button_row.addWidget(yes_button)
        button_row.addWidget(custom_button)

        # Add remove button if more than 1 selection
        if len(self.selection_rows) >= 2:
            remove_button = create_button(self, "Remove", width=96, height=40)
            remove_button.clicked.connect(
                lambda: self.remove_algorithm_selection(row_widget)
            )
            button_row.addWidget(remove_button)

        row_layout.addLayout(button_row)
        row_widget.setLayout(row_layout)
        self.algo_selection_layout.addWidget(row_widget)

        self.selection_rows.append(
            {
                "widget": row_widget,
                "combo": algo_combo,
                "yes_button": yes_button,
                "custom_button": custom_button,
                "use_default": True,
            }
        )

        self.scroll_widget.adjustSize()
        self.adjustSize()
        self.resize(self.sizeHint())

    def remove_algorithm_selection(self, row_widget: QWidget) -> None:
        """Removes an algorithm selection row from the UI.

        Args:
            row_widget (QWidget): The widget representing the row to be removed.
        """
        for row in self.selection_rows:
            if row["widget"] == row_widget:
                self.algo_selection_layout.removeWidget(row_widget)
                row_widget.deleteLater()
                self.selection_rows.remove(row)
                break

        self.scroll_widget.adjustSize()
        self.adjustSize()
        self.resize(self.sizeHint())

    def set_hyper_choice(
        self,
        row_widget: QWidget,
        use_default: bool,
        yes_button: QWidget,
        custom_button: QWidget,
    ) -> None:
        """Sets the hyperparameter choice for a specific algorithm selection.

        Args:
            row_widget (QWidget): The widget representing the row.
            use_default (bool): Whether to use default hyperparameters.
            yes_button (QWidget): The "Yes" button widget.
            custom_button (QWidget): The "Custom" button widget.
        """
        yes_button.setStyleSheet(
            Styles.SELECTED_BUTTON if use_default else Styles.BUTTON
        )
        custom_button.setStyleSheet(
            Styles.SELECTED_BUTTON if not use_default else Styles.BUTTON
        )

        for row in self.selection_rows:
            if row["widget"] == row_widget:
                row["use_default"] = use_default
                break

    def open_custom_hyperparams(
        self,
        algorithm: str,
        row_widget: QWidget,
        custom_button: QWidget,
        yes_button: QWidget,
    ) -> None:
        """Opens the custom hyperparameters window for a specific algorithm.

        Args:
            algorithm (str): The name of the algorithm.
            row_widget (QWidget): The widget representing the row.
            custom_button (QWidget): The "Custom" button widget.
            yes_button (QWidget): The "Yes" button widget.
        """
        custom_button.setStyleSheet(Styles.SELECTED_BUTTON)
        yes_button.setStyleSheet(Styles.BUTTON)

        for row in self.selection_rows:
            if row["widget"] == row_widget:
                row["use_default"] = False
                break

        self.custom_window = SelectHyperWindow(
            algorithm,
            lambda params: self.save_custom_params(row_widget, params),
        )
        self.custom_window.show()

    def save_custom_params(
        self, row_widget: QWidget, hyperparameters: Dict[str, Union[str, int, float]]
    ) -> None:
        """Saves custom hyperparameters for a specific algorithm.

        Args:
            row_widget (QWidget): The widget representing the row.
            hyperparameters (Dict[str, Union[str, int, float]]): The custom hyperparameters.
        """
        for i, row in enumerate(self.selection_rows):
            if row["widget"] == row_widget:
                self.custom_hyperparams[i] = hyperparameters

    def confirm_selection(self) -> None:
        """Confirms the algorithm selections and proceeds to the next window."""
        selections = []
        for i, row in enumerate(self.selection_rows):
            algo_name = row["combo"].currentText()
            hyperparams = {}

            if row["use_default"]:
                try:
                    with open("config/config_algorithm.yaml", "r") as file:
                        config = yaml.safe_load(file)
                        for algo in config.get("algorithms", []):
                            if algo["name"] == algo_name:
                                hyperparams = algo.get("hyperparameters", {})
                                break
                except FileNotFoundError:
                    hyperparams = {}
            else:
                hyperparams = self.custom_hyperparams.get(i, {})

            selections.append(
                {
                    "Algorithm": algo_name,
                    "Hyperparameters": hyperparams,
                }
            )

        if not selections:
            self._show_selection_required_warning()
            return

        self.user_selections["Algorithms"] = selections
        self.close()
        platform_window = PlatformConfigWindow(self.show, self.user_selections)
        platform_window.show()

    def _show_selection_required_warning(self) -> None:
        """Displays a warning message if no algorithm is selected."""
        msg_box = QMessageBox(self)
        msg_box.setIcon(QMessageBox.Warning)
        msg_box.setWindowTitle("Selection Required")
        msg_box.setText("Please select at least one algorithm before proceeding.")
        msg_box.setStyleSheet(Styles.MESSAGE_BOX)
        msg_box.setStandardButtons(QMessageBox.Ok)
        msg_box.exec_()

    def open_welcome_window(self) -> None:
        """Closes the current window and opens the welcome window."""
        self.close()
        self.welcome_window()

    @staticmethod
    def load_algorithms() -> List[str]:
        """Loads the list of available algorithms from the configuration file.

        Returns:
            List[str]: A list of algorithm names.
        """
        try:
            with open("config/config_algorithm.yaml", "r") as file:
                config = yaml.safe_load(file)
                # Filter out "DQN" from the loaded algorithm names
                return [
                    algo["name"]
                    for algo in config.get("algorithms", [])
                    if algo["name"] != "DQN"
                ]
        except FileNotFoundError:
            return []

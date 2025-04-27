import os
import json
from datetime import datetime
from PyQt5.QtWidgets import (
    QStackedWidget,
    QLabel,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QProgressBar,
    QWidget,
    QLineEdit,
    QMessageBox,
    QFrame,
    QSpacerItem,
    QTabWidget,
)
from PyQt5.QtCore import Qt, QUrl, pyqtSignal
from PyQt5.QtGui import QDesktopServices, QIcon
from GUI.ui_utils import PlotCanvas, TrainingThread, make_unique_names
from GUI.ui_base_window import BaseWindow
from GUI.ui_styles import Styles
from GUI.ui_utils import create_button, create_activation_button
from RL_helpers.plotters import plot_comparison


class TrainingWindow(BaseWindow):
    update_algo_signal = pyqtSignal(str, str, object)
    update_plot_signal = pyqtSignal(str, object, str)
    training_completed_signal = pyqtSignal(str, bool)

    def __init__(self, previous_window, previous_selections) -> None:  # noqa
        """Initialize the TrainingWindow class"""
        super().__init__("Training Configuration Window", 1300, 890)

        # handle the possibility of having the same algorithms with different hyperparameters
        make_unique_names(previous_selections["Algorithms"])

        self.main_folder_name = None
        self.selected_button = None
        self.training_start = None

        self.previous_window = previous_window
        self.previous_selections = previous_selections

        self.algo_info = {}
        self.training_threads = []
        self.training_plot_data_by_algo = {}
        self.evaluation_plot_data_by_algo = {}  # For evaluation curves
        self.completed_algorithms = set()
        self.total_algorithms = len(self.previous_selections.get("Algorithms", []))

        self.default_values = {
            "Training Steps": "1000000",
            "Exploration Steps": "1000",
            "Batch Size": "64",
            "G Value": "1",
            "Evaluation Interval": "1000",
            "Evaluation Episodes": "10",
            "Log Interval": "1000",
            "Seed": "0",
        }

        self.connect_signals()
        self.init_ui()

    def connect_signals(self) -> None:
        """Connect signals to their respective slots"""
        signals = [
            (self.update_algo_signal, self.update_algorithm_labels),
            (self.update_plot_signal, self.update_plot),
            (self.training_completed_signal, self.update_confirmation),
        ]
        for signal, slot in signals:
            signal.connect(slot)

    def init_ui(self) -> None:
        """Initialize the UI of the TrainingWindow"""
        main_layout = QVBoxLayout()
        container = QWidget()
        container.setLayout(main_layout)

        main_layout.addLayout(self.create_back_button_layout())

        summary_level = QLabel("ReinforceUI-Studio", self)
        summary_level.setStyleSheet(Styles.BIG_TITLE_LABEL)
        summary_level.setAlignment(Qt.AlignCenter)

        main_layout.addWidget(summary_level)
        main_layout.addLayout(self.create_summary_layout())
        main_layout.addWidget(self.create_separator())

        middle_layout = QHBoxLayout()
        middle_layout.addLayout(self.create_left_layout())
        middle_layout.addWidget(self.create_separator(vertical=True))
        middle_layout.addLayout(self.create_right_layout())
        main_layout.addLayout(middle_layout)

        main_layout.addWidget(self.create_bottom_tab_layout())

        open_log_file_button = create_button(
            self, "Open Log Folder", width=200, height=40
        )
        open_log_file_button.clicked.connect(self.open_log_file)
        main_layout.addWidget(open_log_file_button, alignment=Qt.AlignRight)

        self.setCentralWidget(container)
        self.show_training_curve()
        self.adjust_for_ppo()

    def create_back_button_layout(self) -> QHBoxLayout:
        button_layout = QHBoxLayout()
        back_button = create_button(
            self,
            "Back",
            width=120,
            height=50,
            icon=QIcon("media_resources/icons/back.svg"),
        )
        back_button.clicked.connect(self.back_to_selection)
        button_layout.addWidget(back_button, alignment=Qt.AlignLeft)
        return button_layout

    def create_left_layout(self) -> QVBoxLayout:
        layout = QVBoxLayout()
        label = QLabel("Training Parameters", self)
        label.setStyleSheet(Styles.SUBTITLE_LABEL)
        label.setAlignment(Qt.AlignCenter)
        layout.addWidget(label)

        self.input_layout = QGridLayout()
        self.training_inputs = self.create_input_fields()
        layout.addLayout(self.input_layout)
        layout.addItem(QSpacerItem(20, 180))
        layout.addLayout(self.create_start_stop_button_layout())
        layout.addWidget(self.create_separator())
        return layout

    def create_right_layout(self) -> QVBoxLayout:
        layout = QVBoxLayout()
        subtitle_label = QLabel("Training/Evaluation Curves", self)
        subtitle_label.setStyleSheet(Styles.SUBTITLE_LABEL)
        layout.addWidget(subtitle_label, alignment=Qt.AlignCenter)

        self.plot_stack = QStackedWidget()
        self.training_figure = PlotCanvas()
        self.evaluation_figure = PlotCanvas()
        self.plot_stack.addWidget(self.training_figure)
        self.plot_stack.addWidget(self.evaluation_figure)

        layout.addWidget(self.plot_stack)
        layout.addLayout(self.create_selection_plot_layout())
        layout.addWidget(self.create_separator())
        return layout

    def create_bottom_tab_layout(self) -> QWidget:
        container = QWidget()
        layout = QHBoxLayout(container)  # Set layout on container directly

        tab_widget = QTabWidget(self)
        for algo_dict in self.previous_selections.get("Algorithms", []):
            algo_name = algo_dict.get("UniqueName")  # Use unique display name
            tab = self.create_algorithm_tab(algo_name)
            tab_widget.addTab(tab, algo_name)

        layout.addWidget(tab_widget)
        container.setLayout(layout)
        return container

    def create_algorithm_tab(self, algo_name: str) -> QWidget:
        widget = QWidget()
        labels_layout = QHBoxLayout()
        vertical_layout = QVBoxLayout()

        labels = {
            "Time Remaining": QLabel("Time Remaining: N/A", self),
            "Total Steps": QLabel("Total Steps: 0", self),
            "Episode Number": QLabel("Episode Number: 0", self),
            "Episode Reward": QLabel("Episode Reward: 0", self),
            "Episode Steps": QLabel("Episode Steps: 0", self),
        }
        for label in labels.values():
            label.setStyleSheet(Styles.TEXT_LABEL)
            labels_layout.addWidget(label)

        vertical_layout.addLayout(labels_layout)  # Add the label row

        # Create progress bar
        progress_bar = QProgressBar(self)
        progress_bar.setStyleSheet(Styles.PROGRESS_BAR)
        progress_bar.setFixedHeight(30)
        progress_bar.setValue(0)

        vertical_layout.addWidget(progress_bar)

        # Store references
        self.algo_info[algo_name] = {
            "labels": labels,
            "progress_bar": progress_bar,
        }
        widget.setLayout(vertical_layout)
        return widget

    def create_selection_plot_layout(self) -> QHBoxLayout:
        button_layout = QHBoxLayout()
        self.view_training_button = create_button(
            self, "View Training Curve", width=350, height=40
        )
        self.view_training_button.clicked.connect(self.show_training_curve)
        button_layout.addWidget(self.view_training_button)

        self.view_evaluation_button = create_button(
            self, "View Evaluation Curve", width=350, height=40
        )
        self.view_evaluation_button.clicked.connect(self.show_evaluation_curve)
        button_layout.addWidget(self.view_evaluation_button)
        return button_layout

    def create_start_stop_button_layout(self) -> QHBoxLayout:
        layout = QHBoxLayout()
        start_button = create_activation_button(
            self, "Start", width=160, height=35, start_button=True
        )
        start_button.clicked.connect(self.start_training)
        layout.addWidget(start_button)

        stop_button = create_activation_button(
            self, "Stop", width=160, height=35, start_button=False
        )
        stop_button.clicked.connect(self.stop_training)
        layout.addWidget(stop_button)
        return layout

    def create_input_fields(self) -> dict:
        inputs = {label: QLineEdit(self) for label in self.default_values}
        row, col = 0, 0
        for label, widget in inputs.items():
            text_label = QLabel(label, self)
            text_label.setStyleSheet(Styles.TEXT_LABEL)
            self.input_layout.addWidget(text_label, row, col)
            widget.setText(self.default_values.get(label, ""))
            widget.setStyleSheet(Styles.LINE_EDIT)

            self.input_layout.addWidget(widget, row + 1, col)
            widget.returnPressed.connect(self.lock_inputs)
            col += 1
            if col >= 2:
                col = 0
                row += 2
        return inputs

    def create_summary_layout(self):
        layout = QHBoxLayout()
        display_names = {
            "selected_environment": "Environment",
            "selected_platform": "Platform",
        }
        for key, value in self.previous_selections.items():
            if key == "Algorithms":
                # Join algorithm names with commas
                algo_names = ", ".join([algo["Algorithm"] for algo in value])
                label = QLabel(f"Algorithm(s): {algo_names}", self)
                label.setStyleSheet(Styles.TEXT_LABEL)
                layout.addWidget(label, alignment=Qt.AlignCenter)
            elif key in display_names:
                label = QLabel(f"{display_names[key]}: {value}", self)
                label.setStyleSheet(Styles.TEXT_LABEL)
                layout.addWidget(label, alignment=Qt.AlignCenter)

        view_hyper_button = create_button(
            self, "View Hyperparameters", width=215, height=40
        )
        view_hyper_button.clicked.connect(self.show_summary_hyperparameters)
        layout.addWidget(view_hyper_button, alignment=Qt.AlignRight)
        return layout

    def show_training_completed_message(self, completion_flag) -> None:
        msg_box = QMessageBox(self)
        msg_box.setIcon(
            QMessageBox.Information if completion_flag else QMessageBox.Warning
        )
        msg_box.setWindowTitle(
            "Training Completed" if completion_flag else "Training Interrupted"
        )
        msg_box.setText(
            "The training process has been successfully completed."
            if completion_flag
            else "The training process has been interrupted."
        )
        msg_box.setStyleSheet(Styles.MESSAGE_BOX)
        see_log_button = msg_box.addButton("See log folder", QMessageBox.AcceptRole)
        msg_box.exec_()

        if msg_box.clickedButton() == see_log_button:
            self.open_log_file()
        self.reset_training_window()

    def update_confirmation(self, algo_name, status_flag):
        self.completed_algorithms.add(algo_name)
        if len(self.completed_algorithms) == self.total_algorithms:
            self.generate_comparison_plot()
            self.show_training_completed_message(status_flag)

    def update_plot(self, algo_name, data_plot, plot_type: str):
        if plot_type == "training":
            self.training_plot_data_by_algo[algo_name] = data_plot
            self.training_figure.plot_data(
                data_plot=self.training_plot_data_by_algo,
                title="Training Curve",
                y_label="Episode Reward",
            )
        elif plot_type == "evaluation":
            self.evaluation_plot_data_by_algo[algo_name] = data_plot
            self.evaluation_figure.plot_data(
                data_plot=self.evaluation_plot_data_by_algo,
                title="Evaluation Curve",
                y_label="Average Reward",
            )

    def update_algorithm_labels(self, algo_name: str, key: str, value):
        if algo_name not in self.algo_info:
            return  # Ignore updates for unknown algorithms
        if key == "Time Remaining":
            self.algo_info[algo_name]["labels"]["Time Remaining"].setText(
                f"Time Remaining: {value}"
            )
        elif key == "Total Steps":
            self.algo_info[algo_name]["labels"]["Total Steps"].setText(
                f"Total Steps: {value}"
            )
        elif key == "Episode Number":
            self.algo_info[algo_name]["labels"]["Episode Number"].setText(
                f"Episode Number: {value}"
            )
        elif key == "Episode Reward":
            self.algo_info[algo_name]["labels"]["Episode Reward"].setText(
                f"Episode Reward: {value}"
            )
        elif key == "Episode Steps":
            self.algo_info[algo_name]["labels"]["Episode Steps"].setText(
                f"Episode Steps: {value}"
            )
        elif key == "Progress":
            self.algo_info[algo_name]["progress_bar"].setValue(value)

    def show_training_curve(self):
        self.plot_stack.setCurrentWidget(self.training_figure)
        self.update_button_styles(
            self.view_training_button, self.view_evaluation_button
        )

    def show_evaluation_curve(self):
        self.plot_stack.setCurrentWidget(self.evaluation_figure)
        self.update_button_styles(
            self.view_evaluation_button, self.view_training_button
        )

    def show_summary_hyperparameters(self):
        lines = []

        algorithms = self.previous_selections.get("Algorithms", [])
        for algo in algorithms:
            algo_name = algo.get("Algorithm", "Unknown Algorithm")
            hyperparams = algo.get("Hyperparameters", {})
            lines.append(f"{algo_name}:\n")
            for param, value in hyperparams.items():
                lines.append(f"  {param}: {value}")
            lines.append("")  # Empty line between algorithms

        selections = "\n".join(lines)
        self.show_message_box("Hyperparameters", selections, QMessageBox.Information)

    def lock_inputs(self):
        for widget in self.training_inputs.values():
            widget.setReadOnly(True)

    def start_training(self):
        if self.training_start:
            return

        if not self.all_inputs_filled():
            self.show_message_box(
                "Input Error",
                "Please fill in all fields before starting training.",
                QMessageBox.Warning,
            )
            return

        if self.show_confirmation(
            "Confirm Training", "The training will start. Are you sure?"
        ):
            self.training_start = True
            self.lock_inputs()

            algorithms = self.previous_selections.get("Algorithms", [])
            algo_names = [entry.get("Algorithm") for entry in algorithms]

            self.create_log_folder(algo_names=algo_names)

            shared_training_params = {
                label: widget.text() for label, widget in self.training_inputs.items()
            }

            per_algorithm_configs = []
            for algo_entry in algorithms:
                config = {
                    "Algorithm": algo_entry.get("Algorithm"),
                    "UniqueName": algo_entry.get("UniqueName"),
                    "Hyperparameters": algo_entry.get("Hyperparameters", {}),
                    **shared_training_params,
                    "selected_platform": self.previous_selections.get(
                        "selected_platform"
                    ),
                    "selected_environment": self.previous_selections.get(
                        "selected_environment"
                    ),
                    "setup_choice": self.previous_selections.get("setup_choice"),
                }
                per_algorithm_configs.append(config)

            for config_data in per_algorithm_configs:
                thread = TrainingThread(self, config_data, self.main_folder_name)
                self.training_threads.append(thread)
                thread.start()

    def create_log_folder(self, algo_names: list[str]) -> None:
        home_dir = os.path.expanduser("~")
        timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M")
        algo_str = "_".join(algo_names)
        self.main_folder_name = os.path.join(
            home_dir, f"training_log_{algo_str}_{timestamp}"
        )
        os.makedirs(self.main_folder_name, exist_ok=True)

        # Shared/global training parameters
        training_params = {
            label: widget.text() for label, widget in self.training_inputs.items()
        }

        # Save the global config file
        main_config = {**self.previous_selections, **training_params}
        with open(
            os.path.join(self.main_folder_name, "session_config.json"), "w"
        ) as config_file:
            json.dump(main_config, config_file, indent=4)

        # Save per-algorithm configs
        algorithms = self.previous_selections.get("Algorithms", [])

        for algo_entry in algorithms:
            algo_name_display = algo_entry.get("UniqueName")

            algo_folder = os.path.join(self.main_folder_name, algo_name_display)
            os.makedirs(algo_folder, exist_ok=True)

            algo_config = {
                "Shared Parameters": training_params,
                "Algorithm": algo_entry.get("Algorithm"),
                "Hyperparameters": algo_entry.get("Hyperparameters", {}),
                "selected_platform": self.previous_selections.get("selected_platform"),
                "selected_environment": self.previous_selections.get(
                    "selected_environment"
                ),
            }

            with open(os.path.join(algo_folder, "config.json"), "w") as algo_file:
                json.dump(algo_config, algo_file, indent=4)

    def stop_training(self):
        if self.training_start and self.show_confirmation(
            "Stop Training", "Are you sure you want to stop the training?"
        ):
            self.training_start = False

            # Stop and wait for all threads
            for thread in self.training_threads:
                thread.stop()
                thread.wait()

            # Re-enable UI input fields
            for widget in self.training_inputs.values():
                widget.setReadOnly(False)

    def back_to_selection(self):
        if self.training_start:
            self.show_message_box(
                "Training in Progress",
                "Please stop the training before going back.",
                QMessageBox.Warning,
            )
            return
        self.close()
        self.previous_window()

    def all_inputs_filled(self):
        algorithms = self.previous_selections.get("Algorithms", [])
        is_single_ppo = len(algorithms) == 1 and algorithms[0].get("Algorithm") == "PPO"
        for label, widget in self.training_inputs.items():
            if is_single_ppo and label in [
                "Exploration Steps",
                "Batch Size",
                "G Value",
            ]:
                continue
            if widget.text().strip() == "":
                return False
        return True

    def open_log_file(self):
        if not self.main_folder_name:
            self.show_message_box(
                "Log Folder",
                "Log folder does not exist. Please Start training first.",
                QMessageBox.Warning,
            )
        else:
            QDesktopServices.openUrl(QUrl.fromLocalFile(self.main_folder_name))

    def show_message_box(self, title, text, icon):
        msg_box = QMessageBox(self)
        msg_box.setIcon(icon)
        msg_box.setWindowTitle(title)
        msg_box.setText(text)
        msg_box.setStyleSheet(Styles.MESSAGE_BOX)
        msg_box.setStandardButtons(QMessageBox.Ok)
        msg_box.exec_()

    def show_confirmation(self, title, text):
        confirm_msg = QMessageBox(self)
        confirm_msg.setIcon(QMessageBox.Warning)
        confirm_msg.setWindowTitle(title)
        confirm_msg.setText(text)
        confirm_msg.setStyleSheet(Styles.MESSAGE_BOX)
        confirm_msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        return confirm_msg.exec_() == QMessageBox.Yes

    def generate_comparison_plot(self):
        algo_log_dirs = {}
        for algo_dict in self.previous_selections.get("Algorithms", []):
            unique_name = algo_dict.get("UniqueName", algo_dict["Algorithm"])
            algo_folder = os.path.join(self.main_folder_name, unique_name)
            algo_log_dirs[unique_name] = algo_folder

        output_file = os.path.join(self.main_folder_name, "final_plot.png")

        if len(algo_log_dirs) == 1:
            plot_title = "Final Evaluation Curve"
        else:
            plot_title = "Evaluation Comparison"

        plot_comparison(algo_log_dirs, output_file, title=plot_title)

    def reset_training_window(self):
        self.main_folder_name = None
        self.training_threads = []

        for field, widget in self.training_inputs.items():
            widget.setText(self.default_values.get(field, ""))
            widget.setReadOnly(False)

        # Reset all algorithm-specific UI (labels and progress bars)
        for algo_data in self.algo_info.values():
            for label in algo_data["labels"].values():
                label.setText(label.text().split(":")[0] + ": 0")
            algo_data["progress_bar"].setValue(0)

        self.training_figure.clear_data()
        self.evaluation_figure.clear_data()

        self.adjust_for_ppo()
        self.training_start = False
        self.completed_algorithms = set()
        self.training_plot_data_by_algo.clear()
        self.evaluation_plot_data_by_algo.clear()

    def adjust_for_ppo(self):
        algorithms = self.previous_selections.get("Algorithms", [])
        if len(algorithms) == 1 and algorithms[0].get("Algorithm") == "PPO":
            for field in ["Exploration Steps", "Batch Size", "G Value"]:
                self.training_inputs[field].setText("")
                self.training_inputs[field].setReadOnly(True)

    @staticmethod
    def update_button_styles(active_button, inactive_button):
        active_button.setStyleSheet(Styles.SELECTED_BUTTON)
        inactive_button.setStyleSheet(Styles.BUTTON)

    @staticmethod
    def create_separator(vertical=False) -> QFrame:
        separator = QFrame()
        separator.setFrameShape(QFrame.VLine if vertical else QFrame.HLine)
        separator.setStyleSheet(Styles.SEPARATOR_LINE)
        return separator

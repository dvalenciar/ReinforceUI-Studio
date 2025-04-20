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
)
from PyQt5.QtCore import Qt, QUrl, pyqtSignal
from PyQt5.QtGui import QDesktopServices, QIcon
from GUI.ui_utils import PlotCanvas, TrainingThread
from GUI.ui_base_window import BaseWindow
from GUI.ui_styles import Styles
from GUI.ui_utils import create_button, create_activation_button


class TrainingWindow(BaseWindow):
    update_plot_signal = pyqtSignal(object)
    update_plot_eval_signal = pyqtSignal(object)
    update_progress_signal = pyqtSignal(int)
    update_step_signal = pyqtSignal(int)
    update_reward_signal = pyqtSignal(float)
    update_episode_signal = pyqtSignal(int)
    update_time_remaining_signal = pyqtSignal(str)
    update_episode_steps_signal = pyqtSignal(int)
    training_completed_signal = pyqtSignal(bool)

    def __init__(self, previous_window, previous_selections) -> None:  # noqa
        """Initialize the TrainingWindow class"""
        super().__init__("Training Configuration Window", 1300, 870)

        self.folder_name = None
        self.selected_button = None
        self.training_start = None
        self.previous_window = previous_window
        self.previous_selections = previous_selections

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
        self.init_ui()
        self.connect_signals()

    def connect_signals(self) -> None:
        """Connect signals to their respective slots"""
        signals = [
            (self.update_progress_signal, self.update_progress_bar),
            (self.update_step_signal, self.update_step_label),
            (self.update_reward_signal, self.update_reward_label),
            (self.update_episode_signal, self.update_episode_label),
            (self.update_time_remaining_signal, self.update_time_remaining),
            (self.update_episode_steps_signal, self.update_episode_steps),
            (self.update_plot_signal, self.update_plot),
            (self.update_plot_eval_signal, self.update_plot_eval),
            (
                self.training_completed_signal,
                self.show_training_completed_message,
            ),
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

        main_layout.addLayout(self.create_bottom_layout())
        main_layout.addWidget(self.create_progress_bar())

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
        back_button = create_button(self, "Back", width=120, height=50, icon=QIcon("media_resources/icons/back.svg"))
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

    def create_bottom_layout(self) -> QHBoxLayout:
        layout = QHBoxLayout()
        self.info_labels = self.create_info_labels()
        for label in self.info_labels.values():
            layout.addWidget(label)
        return layout

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
        # start_button = create_activation_button(
        #     self, "Start", width=160, height=35, start_button=True
        # )
        start_button = create_button(self, "Start", width=160, height=35) # todo, check if user this or previous


        start_button.clicked.connect(self.start_training)
        layout.addWidget(start_button)
        stop_button = create_activation_button(
            self, "Stop", width=160, height=35, start_button=False
        )
        stop_button.clicked.connect(self.stop_training)
        layout.addWidget(stop_button)
        return layout

    def create_separator(self, vertical=False) -> QFrame:
        separator = QFrame()
        separator.setFrameShape(QFrame.VLine if vertical else QFrame.HLine)
        separator.setStyleSheet(Styles.SEPARATOR_LINE)
        return separator

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

    def create_info_labels(self) -> dict:
        labels = {
            "Time Remaining": QLabel("Time Remaining: N/A", self),
            "Total Steps": QLabel("Total Steps: 0", self),
            "Episode Number": QLabel("Episode Number: 0", self),
            "Episode Reward": QLabel("Episode Reward: 0", self),
            "Episode Steps": QLabel("Episode Steps: 0", self),
        }
        for label in labels.values():
            label.setStyleSheet(Styles.TEXT_LABEL)
        return labels

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

    def create_progress_bar(self) -> QProgressBar:
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setStyleSheet(Styles.PROGRESS_BAR)
        self.progress_bar.setFixedHeight(30)
        self.progress_bar.setValue(0)
        return self.progress_bar

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
        see_log_button = msg_box.addButton(
            "See log folder", QMessageBox.AcceptRole
        )
        msg_box.exec_()
        if msg_box.clickedButton() == see_log_button:
            self.open_log_file()
        self.reset_training_window()

    def update_plot_eval(self, data_plot):
        self.evaluation_figure.plot_data(
            data_plot, "Evaluation Curve", "Average Reward"
        )

    def update_plot(self, data_plot):
        self.training_figure.plot_data(
            data_plot, "Training Curve", "Episode Reward"
        )

    def update_episode_steps(self, steps):
        self.info_labels["Episode Steps"].setText(f"Episode Steps: {steps}")

    def update_time_remaining(self, time_remaining):
        self.info_labels["Time Remaining"].setText(
            f"Time Remaining: {time_remaining}"
        )

    def update_episode_label(self, episode):
        self.info_labels["Episode Number"].setText(
            f"Episode Number: {episode}"
        )

    def update_reward_label(self, reward):
        self.info_labels["Episode Reward"].setText(f"Episode Reward: {reward}")

    def update_step_label(self, step):
        self.info_labels["Total Steps"].setText(f"Total Steps: {step}")

    def update_progress_bar(self, value):
        self.progress_bar.setValue(value)

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
        self.show_message_box(
            "Hyperparameters", selections, QMessageBox.Information
        )

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
            self.create_log_folder()
            training_params = {
                label: widget.text()
                for label, widget in self.training_inputs.items()
            }
            config_data = {**self.previous_selections, **training_params}
            self.training_thread = TrainingThread(
                self, config_data, self.folder_name
            )
            self.training_thread.start()

    def create_log_folder(self):
        home_dir = os.path.expanduser("~")
        timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M")
        self.folder_name = os.path.join(home_dir, f"training_log_{timestamp}")
        os.makedirs(self.folder_name, exist_ok=True)
        training_params = {
            label: widget.text()
            for label, widget in self.training_inputs.items()
        }
        config_data = {**self.previous_selections, **training_params}
        with open(
            os.path.join(self.folder_name, "config.json"), "w"
        ) as config_file:
            json.dump(config_data, config_file, indent=4)

    def stop_training(self):
        if self.training_start and self.show_confirmation(
            "Stop Training", "Are you sure you want to stop the training?"
        ):
            self.training_start = False
            self.training_thread.stop()
            self.training_thread.wait()
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
        for label, widget in self.training_inputs.items():
            if self.previous_selections.get(
                "Algorithm"
            ) == "PPO" and label in [
                "Exploration Steps",
                "Batch Size",
                "G Value",
            ]:
                continue
            if widget.text().strip() == "":
                return False
        return True

    def open_log_file(self):
        if not self.folder_name:
            self.show_message_box(
                "Log Folder",
                "Log folder does not exist. Please Start training first.",
                QMessageBox.Warning,
            )
        else:
            QDesktopServices.openUrl(QUrl.fromLocalFile(self.folder_name))

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

    def reset_training_window(self):
        self.folder_name = None
        for field, widget in self.training_inputs.items():
            widget.setText(self.default_values.get(field, ""))
        self.progress_bar.setValue(0)
        self.training_figure.clear_data()
        self.evaluation_figure.clear_data()
        for label in self.info_labels.values():
            label.setText(label.text().split(":")[0] + ": 0")
        for widget in self.training_inputs.values():
            widget.setReadOnly(False)
        self.adjust_for_ppo()
        self.training_start = False

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

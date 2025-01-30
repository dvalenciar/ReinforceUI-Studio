import os
import json
from datetime import datetime
from PyQt5.QtWidgets import QStackedWidget
from PyQt5.QtWidgets import (
    QMainWindow,
    QLabel,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QPushButton,
    QProgressBar,
    QWidget,
    QLineEdit,
    QMessageBox,
    QFrame,
    QDesktopWidget,
    QSpacerItem,
)
from PyQt5.QtCore import Qt, QUrl, pyqtSignal, QThread
from PyQt5.QtGui import QDesktopServices
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from RL_loops.training_policy_loop import training_loop


class TrainingWindow(QMainWindow):
    update_plot_signal = pyqtSignal(object)
    update_plot_eval_signal = pyqtSignal(object)
    update_progress_signal = pyqtSignal(int)
    update_step_signal = pyqtSignal(int)
    update_reward_signal = pyqtSignal(float)
    update_episode_signal = pyqtSignal(int)
    update_time_remaining_signal = pyqtSignal(str)
    update_episode_steps_signal = pyqtSignal(int)
    training_completed_signal = pyqtSignal(bool)

    def __init__(self, algorithm_window, previous_selections):
        super().__init__()
        self.folder_name = None
        self.training_start = None
        self.algorithm_window = algorithm_window
        self.previous_selections = previous_selections
        self.init_ui()
        self.connect_signals()

    def connect_signals(self):
        self.update_progress_signal.connect(self.update_progress_bar)
        self.update_step_signal.connect(self.update_step_label)
        self.update_reward_signal.connect(self.update_reward_label)
        self.update_episode_signal.connect(self.update_episode_label)
        self.update_time_remaining_signal.connect(self.update_time_remaining)
        self.update_episode_steps_signal.connect(self.update_episode_steps)
        self.update_plot_signal.connect(self.update_plot)
        self.update_plot_eval_signal.connect(self.update_plot_eval)
        self.training_completed_signal.connect(self.show_training_completed_message)

    def show_training_completed_message(self, completion_flag):
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
        msg_box.setStyleSheet(self.get_message_box_style())

        # Add custom button
        see_log_button = msg_box.addButton("See log folder", QMessageBox.AcceptRole)
        msg_box.exec_()

        if msg_box.clickedButton() == see_log_button:
            self.open_log_file()
        self.reset_training_window()

    def update_plot_eval(self, data_plot):
        self.evaluation_figure.plot_data(
            data_plot, "Evaluation Curve", "Average Reward"
        )

    def update_plot(self, data_plot):
        self.training_figure.plot_data(data_plot, "Training Curve", "Episode Reward")

    def update_episode_steps(self, steps):
        self.info_labels["Episode Steps"].setText(f"Episode Steps: {steps}")

    def update_time_remaining(self, time_remaining):
        self.info_labels["Time Remaining"].setText(f"Time Remaining: {time_remaining}")

    def update_episode_label(self, episode):
        self.info_labels["Episode Number"].setText(f"Episode Number: {episode}")

    def update_reward_label(self, reward):
        self.info_labels["Episode Reward"].setText(f"Episode Reward: {reward}")

    def update_step_label(self, step):
        self.info_labels["Total Steps"].setText(f"Total Steps: {step}")

    def update_progress_bar(self, value):
        self.progress_bar.setValue(value)

    def init_ui(self):
        self.setWindowTitle("Training Configuration Window")
        self.setFixedSize(1100, 700)
        self.setStyleSheet("background-color: black;")
        self.center()

        main_layout = QVBoxLayout()
        container = QWidget()
        container.setLayout(main_layout)

        top_layout = QHBoxLayout()
        top_layout.addWidget(self.create_button("Back", self.back_to_selection))
        top_layout.addStretch()
        main_layout.addLayout(top_layout)

        main_layout.addWidget(
            self.create_label("Summary of Selections", "yellow", 16, True),
            alignment=Qt.AlignLeft,
        )
        main_layout.addLayout(self.create_summary_layout())
        main_layout.addWidget(self.create_separator())

        middle_layout = QHBoxLayout()
        self.left_layout = QVBoxLayout()
        self.left_layout.addWidget(
            self.create_label("Training parameters", "yellow", 16, True),
            alignment=Qt.AlignLeft,
        )
        self.input_layout = QGridLayout()
        self.training_inputs = self.create_input_fields()
        self.left_layout.addLayout(self.input_layout)
        self.left_layout.addItem(QSpacerItem(20, 180))
        self.left_layout.addLayout(self.create_button_layout())
        self.left_layout.addWidget(self.create_separator())
        middle_layout.addLayout(self.left_layout)
        middle_layout.addWidget(self.create_separator(vertical=True))

        right_layout = QVBoxLayout()
        right_layout.addWidget(
            self.create_label("Training/Evaluation Curves", "yellow", 18, True)
        )

        # Create QStackedWidget to hold both plots
        self.plot_stack = QStackedWidget()
        self.training_figure = MatplotlibCanvas()
        self.evaluation_figure = MatplotlibCanvas()

        self.plot_stack.addWidget(self.training_figure)
        self.plot_stack.addWidget(self.evaluation_figure)
        right_layout.addWidget(self.plot_stack)

        arrow_layout = QHBoxLayout()
        self.view_training_button = self.create_button(
            "View Training Curve", self.show_training_curve
        )
        self.view_evaluation_button = self.create_button(
            "View Evaluation Curve", self.show_evaluation_curve
        )
        arrow_layout.addWidget(self.view_training_button)
        arrow_layout.addWidget(self.view_evaluation_button)
        right_layout.addLayout(arrow_layout)
        middle_layout.addLayout(right_layout)
        main_layout.addLayout(middle_layout)

        bottom_layout = QHBoxLayout()
        self.info_labels = self.create_info_labels()
        for label in self.info_labels.values():
            bottom_layout.addWidget(label)
        main_layout.addLayout(bottom_layout)

        self.progress_bar = QProgressBar(self)
        self.progress_bar.setStyleSheet(
            "QProgressBar { background-color: #444444; color: white; border: 2px solid white; border-radius: 5px; text-align: center; } QProgressBar::chunk { background-color: #2a9d8f; width: 20px; }"
        )
        self.progress_bar.setFixedHeight(30)
        self.progress_bar.setValue(0)
        main_layout.addWidget(self.progress_bar)

        main_layout.addWidget(
            self.create_button("View Log Folder", self.open_log_file),
            alignment=Qt.AlignLeft,
        )
        self.setCentralWidget(container)

        # Set the training curve as the default view
        self.show_training_curve()

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

    def create_button(
        self,
        text,
        callback=None,
        width=None,
        height=None,
        style="background-color: #444444;",
    ):
        button = QPushButton(text, self)
        button.setStyleSheet(
            f"QPushButton {{ {style} color: white; font-size: 14px; padding: 5px 15px; border-radius: 5px; border: 1px solid white; }} QPushButton:hover {{ background-color: #555555; }}"
        )
        if width and height:
            button.setFixedSize(width, height)
        if callback:
            button.clicked.connect(callback)
        return button

    def create_separator(self, vertical=False):
        separator = QFrame()
        separator.setFrameShape(QFrame.VLine if vertical else QFrame.HLine)
        separator.setStyleSheet("color: white; border: 1px solid white;")
        return separator

    def create_input_fields(self):
        inputs = {
            "Training Steps": QLineEdit(self),
            "Exploration Steps": QLineEdit(self),
            "Batch Size": QLineEdit(self),
            "G Value": QLineEdit(self),
            "Evaluation Interval": QLineEdit(self),
            "Evaluation Episodes": QLineEdit(self),
            "Log Interval": QLineEdit(self),
            "Seed": QLineEdit(self),
        }
        default_values = {
            "Training Steps": "10000",
            "Exploration Steps": "1000",
            "Batch Size": "32",
            "G Value": "1",
            "Evaluation Interval": "1000",
            "Evaluation Episodes": "10",
            "Log Interval": "1000",
            "Seed": "0",
        }
        row, col = 0, 0
        for label, widget in inputs.items():
            self.input_layout.addWidget(self.create_label(label, "white", 14), row, col)
            widget.setText(default_values.get(label, ""))
            widget.setStyleSheet(
                "QLineEdit { background-color: #444444; color: white; font-size: 14px; padding: 5px; border: 1px solid white; }"
            )
            self.input_layout.addWidget(widget, row + 1, col)
            widget.returnPressed.connect(self.lock_inputs)
            col += 1
            if col >= 2:
                col = 0
                row += 2
        return inputs

    def create_info_labels(self):
        labels = {
            "Time Remaining": QLabel("Time Remaining: N/A", self),
            "Total Steps": QLabel("Total Steps: 0", self),
            "Episode Number": QLabel("Episode Number: 0", self),
            "Episode Reward": QLabel("Episode Reward: 0", self),
            "Episode Steps": QLabel("Episode Steps: 0", self),
        }
        for label in labels.values():
            label.setStyleSheet("color: white; font-size: 14px;")
        return labels

    def create_label(self, text, color, size, bold=False):
        label = QLabel(text, self)
        label.setStyleSheet(
            f"color: {color}; font-size: {size}px; {'font-weight: bold;' if bold else ''}"
        )
        return label

    def create_summary_layout(self):
        summary_layout = QHBoxLayout()
        display_names = {
            "selected_platform": "Platform",
            "selected_environment": "Environment",
            "Algorithm": "Algorithm",
        }
        for key, value in self.previous_selections.items():
            if key in display_names:
                summary_layout.addWidget(
                    self.create_label(f"{display_names[key]}: {value}", "white", 14),
                    alignment=Qt.AlignLeft,
                )
        summary_layout.addWidget(
            self.create_button(
                "View Hyperparameters", self.show_summary_hyperparameters, 200, 30
            )
        )
        return summary_layout

    def create_button_layout(self):
        button_layout = QHBoxLayout()
        button_layout.addWidget(
            self.create_button(
                "Start", self.start_training, style="background-color: green;"
            )
        )
        button_layout.addWidget(
            self.create_button(
                "Stop", self.stop_training, style="background-color: red;"
            )
        )
        return button_layout

    def show_summary_hyperparameters(self):
        relevant_keys = ["Hyperparameters"]
        lines = [
            f"{sub_key}: {sub_value}\n"
            for key, values in self.previous_selections.items()
            if key in relevant_keys
            for sub_key, sub_value in values.items()
        ]
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
            self.create_log_folder()
            training_params = {
                label: widget.text() for label, widget in self.training_inputs.items()
            }
            config_data = {**self.previous_selections, **training_params}
            self.training_thread = TrainingThread(self, config_data, self.folder_name)
            self.training_thread.start()

    def create_log_folder(self):
        home_dir = os.path.expanduser("~")
        timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M")
        self.folder_name = os.path.join(home_dir, f"training_log_{timestamp}")
        os.makedirs(self.folder_name, exist_ok=True)
        training_params = {
            label: widget.text() for label, widget in self.training_inputs.items()
        }
        config_data = {**self.previous_selections, **training_params}
        with open(os.path.join(self.folder_name, "config.json"), "w") as config_file:
            json.dump(config_data, config_file, indent=4)

    def stop_training(self):
        if self.training_start and self.show_confirmation(
            "Stop Training", "Are you sure you want to stop the training?"
        ):
            self.training_start = False
            self.training_thread.stop()
            self.training_thread.wait()  # Wait for the thread to finish
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
        self.algorithm_window()

    def all_inputs_filled(self):
        return all(
            widget.text().strip() != "" for widget in self.training_inputs.values()
        )

    def center(self):
        screen_geometry = QDesktopWidget().availableGeometry().center()
        frame_geometry = self.frameGeometry()
        frame_geometry.moveCenter(screen_geometry)
        self.move(frame_geometry.topLeft())

    def open_log_file(self):
        # if hasattr(self, 'folder_name') and os.path.exists(self.folder_name):
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
        msg_box.setStyleSheet(self.get_message_box_style())
        msg_box.setStandardButtons(QMessageBox.Ok)
        msg_box.exec_()

    def show_confirmation(self, title, text):
        confirm_msg = QMessageBox(self)
        confirm_msg.setIcon(QMessageBox.Warning)
        confirm_msg.setWindowTitle(title)
        confirm_msg.setText(text)
        confirm_msg.setStyleSheet(self.get_message_box_style())
        confirm_msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        return confirm_msg.exec_() == QMessageBox.Yes

    def reset_training_window(self):
        # Reset all input fields to their default values
        self.folder_name = None

        default_values = {
            "Training Steps": "10000",
            "Exploration Steps": "1000",
            "Batch Size": "32",
            "G Value": "1",
            "Evaluation Interval": "1000",
            "Evaluation Episodes": "10",
            "Log Interval": "1000",
            "Seed": "0",
        }
        for field, widget in self.training_inputs.items():
            widget.setText(default_values.get(field, ""))

        # Reset progress bar
        self.progress_bar.setValue(0)

        # Clear plots
        self.training_figure.clear_data()
        self.evaluation_figure.clear_data()

        # Reset info labels
        self.info_labels["Time Remaining"].setText("Time Remaining: N/A")
        self.info_labels["Total Steps"].setText("Total Steps: 0")
        self.info_labels["Episode Number"].setText("Episode Number: 0")
        self.info_labels["Episode Reward"].setText("Episode Reward: 0")
        self.info_labels["Episode Steps"].setText("Episode Steps: 0")

        # Unlock input fields
        for widget in self.training_inputs.values():
            widget.setReadOnly(False)

        # Reset training start flag
        self.training_start = False

    @staticmethod
    def get_message_box_style():
        return (
            "QMessageBox { background-color: black; color: white; } "
            "QMessageBox QLabel { color: white; } "
            "QMessageBox QPushButton { background-color: #444444; color: white; font-size: 14px; padding: 5px 15px; border-radius: 5px; border: 1px solid white; } "
            "QMessageBox QPushButton:hover { background-color: #555555; }"
        )

    @staticmethod
    def update_button_styles(active_button, inactive_button):
        active_button.setStyleSheet(
            """
            QPushButton { background-color: #2a9d8f; color: white; font-size: 14px; padding: 5px 15px; border-radius: 5px; border: 1px solid white; }
        """
        )
        inactive_button.setStyleSheet(
            """
            QPushButton { background-color: #444444; color: white; font-size: 14px; padding: 5px 15px; border-radius: 5px; border: 1px solid white; }
            QPushButton:hover { background-color: #555555; }
        """
        )


class TrainingThread(QThread):
    def __init__(self, training_window, config_data, log_folder):
        super().__init__()
        self.config_data = config_data
        self.training_window = training_window
        self.log_folder = log_folder
        self._is_running = True

    def run(self):
        print("Training thread started")
        training_loop(
            self.config_data,
            self.training_window,
            self.log_folder,
            is_running=lambda: self._is_running,
        )

    def stop(self):
        self._is_running = False


class MatplotlibCanvas(FigureCanvas):
    def __init__(self):
        # Create a Matplotlib figure
        self.figure = Figure(facecolor="black")
        super().__init__(self.figure)

        # Create an axis for the plot
        self.ax = self.figure.add_subplot(111, facecolor="black", frameon=False)
        self.clear_data()

    def plot_data(self, data_plot, title, y_label):
        self.ax.clear()
        self.ax.set_title(title, color="white", fontsize=12)
        self.ax.set_xlabel("Steps", color="white", fontsize=11, labelpad=1)
        self.ax.set_ylabel(y_label, color="white", fontsize=11, labelpad=1)
        self.ax.tick_params(axis="x", colors="white", labelsize=10)
        self.ax.tick_params(axis="y", colors="white", labelsize=10)
        self.ax.grid(True, color="gray", linestyle="--", linewidth=0.5)
        self.ax.plot(
            data_plot["Total Timesteps"],
            data_plot[y_label],
            color="#32CD32",
            linewidth=2,
        )
        self.draw()

    def clear_data(self):
        self.ax.clear()
        self.ax.grid(False)
        self.ax.tick_params(axis="x", colors="black", labelsize=10)
        self.ax.tick_params(axis="y", colors="black", labelsize=10)
        self.ax.text(
            0.5,
            0.5,
            "Reward Curves will be displayed here soon",
            ha="center",
            va="center",
            fontsize=12,
            color="gray",
        )
        self.draw()

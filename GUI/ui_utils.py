from PyQt5.QtWidgets import QPushButton
from GUI.ui_styles import Styles
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from RL_loops.training_policy_loop import training_loop
from PyQt5.QtCore import QThread


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


class PlotCanvas(FigureCanvasQTAgg):
    """A canvas for plotting training progress."""

    def __init__(self, width: int = 10, height: int = 4, dpi: int = 100):
        """Initialize the plot canvas.

        Args:
            width: Figure width in inches
            height: Figure height in inches
            dpi: Resolution in dots per inch
        """
        self.figure = Figure(figsize=(width, height), dpi=dpi)
        super().__init__(self.figure)
        self.figure.set_facecolor("black")
        self.ax = self.figure.add_subplot(
            111, facecolor="black", frameon=False
        )
        self.clear_data()

    def plot_data(self, data_plot: dict, title: str, y_label: str) -> None:
        """Plot training data.

        Args:
            data_plot: Dictionary containing data to plot
            title: Plot title
            y_label: Y-axis label
        """
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

    def clear_data(self) -> None:
        """Clear plot and display placeholder text."""
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


def create_button(
    parent, text=" ", width=270, height=50, icon=None, transparent=False
) -> QPushButton:
    """Create a standardized button with consistent styling.

    Args:
        parent: Parent widget
        text: Button text
        width: Button width
        height: Button height
        icon: Optional QIcon
        transparent: Whether to use transparent styling

    Returns:
        QPushButton: Styled button
    """
    button = QPushButton(text, parent)
    button.setFixedSize(width, height)

    if icon:
        button.setIcon(icon)

    button.setStyleSheet(
        Styles.TRANSPARENT_BUTTON if transparent else Styles.BUTTON
    )
    return button

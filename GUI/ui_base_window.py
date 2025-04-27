from PyQt5.QtWidgets import QMainWindow, QDesktopWidget
from GUI.ui_styles import Styles


class BaseWindow(QMainWindow):
    """Base window class with common functionality for all windows."""

    def __init__(self, title: str, width: int = 900, height: int = 230) -> None:
        """Initialize the BaseWindow class.

        Args:
            title (str): The title of the window.
            width (int): The width of the window.
            height (int): The height of the window.
        """
        super().__init__()
        self.setWindowTitle(title)
        self.setFixedSize(width, height)
        self.setStyleSheet(Styles.MAIN_BACKGROUND)
        self.center()

    def center(self) -> None:
        """Center the window on the screen."""
        screen_geometry = QDesktopWidget().availableGeometry().center()
        frame_geometry = self.frameGeometry()
        frame_geometry.moveCenter(screen_geometry)
        self.move(frame_geometry.topLeft())

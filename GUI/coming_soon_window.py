from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from PyQt5.QtCore import Qt


class ComingSoonWindow(QMainWindow):
    def __init__(self, main_window_callback):
        super().__init__()
        self.setWindowTitle("Feature - Coming Soon")
        self.setFixedSize(400, 400)
        self.main_window_callback = main_window_callback

        central_widget = QWidget(self)
        layout = QVBoxLayout(central_widget)
        layout.setAlignment(Qt.AlignCenter)

        label = QLabel(" Feature - Coming Soon!", self)
        label.setAlignment(Qt.AlignCenter)
        layout.addWidget(label)

        back_button = QPushButton("Back to Main Window", self)
        back_button.clicked.connect(self.return_to_main_window)
        layout.addWidget(back_button)

        self.setCentralWidget(central_widget)
        self.center()

        self.setStyleSheet(
            """
            QMainWindow { background-color: black; }
            QLabel { color: yellow; font-size: 16px; font-weight: bold; }
            QPushButton {
                background-color: #444444;
                color: white;
                font-size: 16px;
                padding: 10px 20px;
                border-radius: 10px;
                border: 1px solid white;
            }
            QPushButton:hover { background-color: #555555; }
        """
        )

    def return_to_main_window(self):
        self.main_window_callback()
        self.close()

    def center(self):
        screen_geometry = QApplication.desktop().availableGeometry().center()
        frame_geometry = self.frameGeometry()
        frame_geometry.moveCenter(screen_geometry)
        self.move(frame_geometry.topLeft())


if "__main__" == __name__:
    app = QApplication([])
    window = ComingSoonWindow(app.quit)
    window.show()
    app.exec_()

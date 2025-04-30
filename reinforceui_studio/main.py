import sys
from PyQt5.QtWidgets import QApplication
from reinforceui_studio.GUI.welcome_window import WelcomeWindow


def main() -> None:
    """Run the main function."""
    app = QApplication(sys.argv)
    window = WelcomeWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

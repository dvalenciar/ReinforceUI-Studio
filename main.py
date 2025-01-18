import sys
from PyQt5.QtWidgets import QApplication
from GUI.welcome_window import WelcomeWindow


def main():
    app = QApplication(sys.argv)
    window = WelcomeWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

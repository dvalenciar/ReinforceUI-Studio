import pytest
from PyQt5.QtWidgets import QApplication
from GUI.welcome_window import WelcomeWindow


@pytest.fixture
def app():
    app = QApplication([])
    yield app
    app.quit()


def test_welcome_window(app):
    window = WelcomeWindow()
    assert window is not None
    assert window.isVisible() == False

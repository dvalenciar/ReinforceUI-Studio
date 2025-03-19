class Styles:
    """Central location for all styling in the application."""

    MAIN_BACKGROUND = "background-color: #121212;"

    WELCOME_LABEL = """
        color: #black;
        font-size: 18px;
        font-weight: bold;
        padding: 10px;
        
    """

    BUTTON = """
        QPushButton {
            background-color: #E3F2FD;
            color: #0D47A1;
            font-size: 15px;
            padding: 10px 20px;
            border-radius: 20px;
            border: 1.2px solid #1976D2;
          
        }
        QPushButton:hover {
            background-color: #1976D2;
            color: white;
        }
    """

    ACTIVE_BUTTON = """
        QPushButton { 
            background-color: #2a9d8f; 
            color: white; 
            font-size: 14px; 
            padding: 5px 15px; 
            border-radius: 5px; 
            border: 1px solid white; 
        }
    """

    INACTIVE_BUTTON = """
        QPushButton { 
            background-color: #444444; 
            color: white; font-size: 14px; 
            padding: 5px 15px; 
            border-radius: 5px; 
            border: 1px solid white; 
        }
        QPushButton:hover { 
            background-color: #555555; 
        }
    """

    TRANSPARENT_BUTTON = """
        QPushButton {
            background-color: transparent;
            color: white;
            font-size: 15px;
            padding: 10px 20px;
            border-radius: 10px;
            border: 1px solid white;
        }
        QPushButton:hover {
            background-color: rgba(255, 255, 255, 0.1);
        }
    """

    SELECTED_BUTTON = """
        QPushButton {
            background-color: #2a9d8f;
            color: white;
            font-size: 16px;
            padding: 10px 20px;
            border-radius: 10px;
            border: 2px solid white;
        }
    """

    COMBO_BOX = """
        QComboBox {
            background-color: #444444;
            color: white;
            font-size: 16px;
            padding: 0px;
            border: 1px solid white;
        }
    """

    MESSAGE_BOX = """
        QMessageBox { background-color: black; color: white; }
        QMessageBox QLabel { color: white; }
        QMessageBox QPushButton {
            background-color: #444444;
            color: white; font-size: 14px;
            padding: 5px 15px;
            border-radius: 5px;
            border: 1px solid white;
        }
        QMessageBox QPushButton:hover { background-color: #555555; }
    """

    LINE_EDIT = """
        QLineEdit {
            background-color: #444444;
            color: white;
            font-size: 16px;
            padding: 5px;
            border: 1px solid white;
        }
    """

    PROGRESS_BAR = """
        QProgressBar {
            background-color: #444444;
            color: white;
            border: 2px solid white;
            border-radius: 5px;
            text-align: center;
        }
        QProgressBar::chunk { background-color: #2a9d8f; width: 20px; }
    """

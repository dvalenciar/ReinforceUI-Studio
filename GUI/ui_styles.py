class Styles:
    """Central location for all styling in the application."""

    MAIN_BACKGROUND = "background: linear-gradient(to bottom, #EAF2F8, #D6EAF8);"

    WELCOME_LABEL = """
        color: #333333;
        font-size: 20px;
        font-weight: bold;
        padding: 10px;    
    """

    SUBTITLE_LABEL = """
        color: #333333;
        font-size: 18px;
        font-weight: bold;
        padding: 10px;
    """

    TEXT_LABEL = """
        color: #555555;
        font-size: 16px;
        padding: 1px;
    """

    SEPARATOR_LINE = """
        color: #1976D2;
        border: 1px solid #1976D2;
    """


    BUTTON = """
        QPushButton {
            background-color: #E3F2FD; /* Light Blue */
            color: #0D47A1; /* Dark Blue Text */
            font-size: 15px;
            padding: 10px 20px;
            border-radius: 17px;
            border: 1.1px solid #1976D2;
        }
        QPushButton:hover {
            background-color: #1976D2; /* Medium Blue */
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
            background-color: #1976D2; /* Medium Blue - Matches Hover Effect */
            color: white;
            font-size: 15px;
            padding: 10px 20px;
            border-radius: 17px;
            border: 2px solid #0D47A1; /* Darker Blue Border */
        }

        QPushButton:hover {
            background-color: #1565C0; /* Slightly Darker Blue */
        }
    """


    COMBO_BOX = """
        QComboBox {
            background-color: linear-gradient(to bottom, #EAF2F8, #D6EAF8);
            color: #0D47A1; /* Dark Blue Text - Matches Button */
            font-size: 16px;
            padding: 5px;
            border-radius: 10px;
            border: 1px solid #1976D2; /* Matches Button Border */
        }

        QComboBox:hover {
            background-color: #BBDEFB; /* Softer Blue on Hover */
        }
        
        QComboBox QAbstractItemView {
            background-color: white;
            color: #0D47A1;
            border: 1px solid #1976D2;
            selection-background-color: #1976D2; /* Highlighted item */
            selection-color: white;
        }
    """

    MESSAGE_BOX = """
        QMessageBox {
            background-color: linear-gradient(to bottom, #EAF2F8, #D6EAF8);
            color: #0D47A1; /* Dark Blue Text */
            border: 1px solid #1976D2; /* Matches Button Border */
        }

        QMessageBox QLabel {
            color: #0D47A1; /* Dark Blue Text */
            font-size: 16px;
        }

        QMessageBox QPushButton {
            background-color: #1976D2; /* Medium Blue - Matches Button */
            color: white;
            font-size: 14px;
            padding: 5px 15px;
            border-radius: 5px;
            border: 1px solid #0D47A1; /* Dark Blue Border */
        }

        QMessageBox QPushButton:hover {
            background-color: #1565C0; /* Slightly Darker Blue */
        }
    """

    LINE_EDIT = """
        QLineEdit {
            background-color: linear-gradient(to bottom, #EAF2F8, #D6EAF8);
            color: #0D47A1; /* Dark Blue Text */
            font-size: 16px;
            padding: 2px;
            border-radius: 5px;
            border: 1px solid #1976D2; /* Matches Button Border */
        }

        QLineEdit:focus {
            border: 2px solid #1565C0; /* Slightly Darker Blue on Focus */
        }
    """

    PROGRESS_BAR = """
        QProgressBar {
            background-color: linear-gradient(to bottom, #EAF2F8, #D6EAF8);
            color: #0D47A1; /* Dark Blue Text */
            border: 2px solid #1976D2; /* Medium Blue Border */
            border-radius: 5px;
            text-align: center;
        }

        QProgressBar::chunk {
            background-color: #1976D2; /* Medium Blue - Matches Buttons */
            width: 20px;
        }
    """



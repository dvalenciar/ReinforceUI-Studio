class Styles:
    """Central location for all styling in the application."""

    # Color palette - centralized for easy theme changes
    COLORS = {
        "primary": "#1976D2",  # Medium blue (primary brand color)
        "primary_dark": "#0D47A1",  # Dark blue for text and borders
        "primary_light": "#BBDEFB",  # Light blue for hover states
        "background": "#F0F8FF",  # Very light blue for backgrounds
        "success": "#388E3C",  # Green for positive actions
        "success_hover": "#2E7D32",  # Darker green for hover
        "danger": "#D32F2F",  # Red for negative/warning actions
        "danger_hover": "#B71C1C",  # Darker red for hover
        "text_primary": "#333333",  # Dark gray for primary text
        "text_secondary": "#555555",  # Medium gray for secondary text
        "white": "#FFFFFF",  # White for text on dark backgrounds
    }

    # Base gradients
    BG_GRADIENT = (
        f"background: linear-gradient(to bottom, {COLORS['background']}, #E3F2FD);"
    )
    WIDGET_GRADIENT = f"background: linear-gradient(to bottom, #EAF2F8, #D6EAF8);"

    # Common properties
    FONT_FAMILY = "font-family: Arial, sans-serif;"
    BORDER_RADIUS_SM = "border-radius: 5px;"
    BORDER_RADIUS_MD = "border-radius: 10px;"
    BORDER_RADIUS_LG = "border-radius: 17px;"

    # Main background
    MAIN_BACKGROUND = BG_GRADIENT

    # Text styles with consistent hierarchy
    BIG_TITLE_LABEL = f"""
        {FONT_FAMILY}
        color: {COLORS['text_primary']};
        font-size: 28px;
        font-weight: bold;
        padding: 10px;
    """

    WELCOME_LABEL = f"""
        {FONT_FAMILY}
        color: {COLORS['text_primary']};
        font-size: 20px;
        font-weight: bold;
        padding: 10px;
    """

    SUBTITLE_LABEL = f"""
        {FONT_FAMILY}
        color: {COLORS['text_primary']};
        font-size: 18px;
        font-weight: bold;
        padding: 10px;
    """

    TEXT_LABEL = f"""
        {FONT_FAMILY}
        color: {COLORS['text_secondary']};
        font-size: 16px;
        padding: 1px;
    """

    SEPARATOR_LINE = f"""
        color: {COLORS['primary']};
        border: 1px solid {COLORS['primary']};
    """

    # Button styles with consistent properties
    BUTTON = f"""
        QPushButton {{
            {FONT_FAMILY}
            background-color: rgba(227, 242, 253, 0.6);
            color: {COLORS['primary_dark']};
            font-size: 15px;
            padding: 10px 20px;
            {BORDER_RADIUS_LG}
            border: 1.1px solid {COLORS['primary']};
        }}
        QPushButton:hover {{
            background-color: {COLORS['primary']};
            color: {COLORS['white']};
        }}
        QPushButton:disabled {{
            background-color: #D6D6D6;
            color: #888888;
            border: 1.1px solid #AAAAAA;
        }}
    """

    SELECTED_BUTTON = f"""
        QPushButton {{
            {FONT_FAMILY}
            background-color: {COLORS['success']};
            color: {COLORS['white']};
            font-size: 15px;
            padding: 10px 20px;
            {BORDER_RADIUS_LG}
            border: 2px solid {COLORS['success']};
        }}
        QPushButton:hover {{
            background-color: {COLORS['success_hover']};
        }}
    """

    START_BUTTON = f"""
        QPushButton {{
            {FONT_FAMILY}
            background-color: {COLORS['success']};
            color: {COLORS['white']};
            font-size: 14px;
            padding: 5px 15px;
            {BORDER_RADIUS_SM}
            border: 1px solid {COLORS['success']};
        }}
        QPushButton:hover {{
            background-color: {COLORS['success_hover']};
            border: 1px solid {COLORS['primary']};
        }}
        QPushButton:disabled {{
            background-color: #A0C0A0;
            color: #EFEFEF;
            border: 1px solid #A0C0A0;
        }}
    """

    STOP_BUTTON = f"""
        QPushButton {{
            {FONT_FAMILY}
            background-color: {COLORS['danger']};
            color: {COLORS['white']};
            font-size: 14px;
            padding: 5px 15px;
            {BORDER_RADIUS_SM}
            border: 1px solid {COLORS['danger']};
        }}
        QPushButton:hover {{
            background-color: {COLORS['danger_hover']};
            border: 1px solid #FF6F61;
        }}
        QPushButton:disabled {{
            background-color: #D6A0A0;
            color: #EFEFEF;
            border: 1px solid #D6A0A0;
        }}
    """

    COMBO_BOX = f"""
        QComboBox {{
            {FONT_FAMILY}
            {WIDGET_GRADIENT}
            font-size: 16px;
            padding: 5px;
            {BORDER_RADIUS_MD}
            border: 1px solid {COLORS['primary']};
        }}
        QComboBox:hover {{
            background-color: {COLORS['primary_light']};
            color: {COLORS['primary_dark']};
        }}

        QComboBox QAbstractItemView {{
            {FONT_FAMILY}
            {WIDGET_GRADIENT}
            selection-color: {COLORS['white']};
            selection-background-color: {COLORS['primary']};
            border: 1px solid {COLORS['primary']};
            {BORDER_RADIUS_SM}
        }}
    """

    LINE_EDIT = f"""
        QLineEdit {{
            {FONT_FAMILY}
            {WIDGET_GRADIENT}
            color: {COLORS['primary_dark']};
            font-size: 16px;
            padding: 5px;
            {BORDER_RADIUS_SM}
            border: 1px solid {COLORS['primary']};
        }}
        QLineEdit:focus {{
            border: 2px solid #1565C0;
        }}
        QLineEdit:disabled {{
            background-color: #EEEEEE;
            color: #888888;
            border: 1px solid #CCCCCC;
        }}
    """

    PROGRESS_BAR = f"""
        QProgressBar {{
            {FONT_FAMILY}
            {WIDGET_GRADIENT}
            color: {COLORS['text_primary']};
            border: 2px solid {COLORS['primary']};
            {BORDER_RADIUS_MD}
            text-align: center;
            min-height: 15px;
        }}
        QProgressBar::chunk {{
            background-color: {COLORS['primary']};
            width: 20px;
        }}
    """

    MESSAGE_BOX = f"""
        QMessageBox {{
            {FONT_FAMILY}
            {WIDGET_GRADIENT}
            color: {COLORS['primary_dark']};
            border: 1px solid {COLORS['primary']};
        }}
        QMessageBox QLabel {{
            {FONT_FAMILY}
            color: {COLORS['primary_dark']};
            font-size: 16px;
        }}
        QMessageBox QPushButton {{
            {FONT_FAMILY}
            background-color: {COLORS['primary']};
            color: {COLORS['white']};
            font-size: 14px;
            padding: 5px 15px;
            {BORDER_RADIUS_SM}
            border: 1px solid {COLORS['primary_dark']};
            min-width: 80px;
        }}
        QMessageBox QPushButton:hover {{
            background-color: #1565C0;
        }}
    """

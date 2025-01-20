# Use the official Python image from the Docker Hub
FROM python:3.10-slim

# Set environment variables for non-buffered Python output, GUI compatibility, and OpenGL rendering
ENV PYTHONUNBUFFERED=1 \
    QT_QPA_PLATFORM=xcb \
    DISPLAY=:0 \
    LIBGL_ALWAYS_INDIRECT=1 \
    QT_DEBUG_PLUGINS=1

# Install required libraries for Qt, X11, OpenGL, and fonts
RUN apt-get update && apt-get install -y \
    libdbus-1-3 \
    libxcb-xkb1 \
    libx11-xcb1 \
    libxcb-util1 \
    libxcb-render0 \
    libxcb-render-util0 \
    libxcb-shape0 \
    libxcb-xfixes0 \
    libxcb-icccm4 \
    libxcb-image0 \
    libxcb-keysyms1 \
    libxcb-randr0 \
    libxcb-xinerama0 \
    libxcb-glx0 \
    libxkbcommon-x11-0 \
    libqt5core5a \
    libqt5gui5 \
    libqt5widgets5 \
    libqt5dbus5 \
    libqt5xml5 \
    x11-apps \
    libssl-dev \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxtst6 \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy and install dependencies separately to leverage Docker caching
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

## Command to run your app
#CMD ["python3", "main.py"]
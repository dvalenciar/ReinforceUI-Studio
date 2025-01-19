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
    libx11-dev libxcomposite1 libxrandr2 libxi6 libgl1-mesa-glx libegl1-mesa \
    libxcb1 libxcb-util1 libxcb-icccm4 libxcb-image0 libxcb-keysyms1 libxcb-render0 \
    libxcb-shape0 libxcb-shm0 libxcb-xfixes0 libxcb-xinerama0 libxcb-randr0 libxcb-glx0 \
    libxcb-render-util0 libxkbcommon0 libglib2.0-0 libfontconfig1 libxext6 libxrender1 \
    libxfixes3 libxcursor1 libxinerama1 libxss1 libxtst6 fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*



# Set the working directory
WORKDIR /app

# Copy and install dependencies separately to leverage Docker caching
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Command to run your app
CMD ["python3", "main.py"]

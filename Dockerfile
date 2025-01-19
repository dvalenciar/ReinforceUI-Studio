# Use the official Python image from the Docker Hub
FROM python:3.10-slim

# Set environment variables for non-buffered Python output and GUI compatibility
ENV PYTHONUNBUFFERED=1 \
    QT_QPA_PLATFORM=wayland

# Set the working directory
WORKDIR /app

# Copy and install dependencies separately to leverage Docker caching
COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

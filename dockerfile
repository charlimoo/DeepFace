# Dockerfile for Declarative Deployment Services (Final, Compatible Version)

# Use an official Python runtime as a parent image.
FROM python:3.10-slim

# Set environment variables.
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Install system-level dependencies.
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# --- COMPATIBLE USER CREATION ---
# Create the user and group, then manually create the home directory.
RUN addgroup --system app && adduser --system --ingroup app app

# Set the HOME environment variable.
ENV HOME=/home/app

# Set the working directory.
WORKDIR /app

# Copy the requirements file.
COPY requirements.txt .

# Install Python dependencies.
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code.
COPY . .

# --- FINAL PERMISSION FIX ---
# Create the home directory and set ownership for all necessary directories at once.
# This runs as root before we switch to the 'app' user.
RUN mkdir -p /home/app && chown -R app:app /app /home/app

# Declare the directories that should be mounted to persistent disks.
VOLUME ["/app/face_database", "/home/app/.deepface/weights"]

# Switch to the non-root user for security.
USER app

# Expose the port that Streamlit runs on.
EXPOSE 8501

# The command to run the application.
CMD ["streamlit", "run", "app.py", "--server.enableCORS=false", "--server.enableXsrfProtection=false"]
# Dockerfile

# Use an official Python runtime as a parent image.
# Using 'slim' keeps the image size down.
FROM python:3.10-slim

# Set environment variables to prevent buffering of prints and
# to handle debian package installation non-interactively.
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Install system-level dependencies required by OpenCV and other libraries.
# 'ffmpeg' is for video processing, 'libgl1' is for graphics.
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user for security purposes.
RUN addgroup --system app && adduser --system --group app

# Set the working directory in the container.
WORKDIR /app

# Create the application directories and set permissions.
# We pre-create the .deepface directory to mount model weights later.
RUN mkdir -p /app/face_database /home/app/.deepface \
    && chown -R app:app /app /home/app/.deepface

# Copy the requirements file first to leverage Docker's layer caching.
# This step will only be re-run if requirements.txt changes.
COPY --chown=app:app requirements.txt .

# Install Python dependencies.
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container.
COPY --chown=app:app . .

# Switch to the non-root user.
USER app

# Expose the port that Streamlit runs on.
EXPOSE 8501

# The command to run the application.
CMD ["streamlit", "run", "app.py"]
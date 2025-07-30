# Dockerfile for Declarative Deployment Services

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

# Create a non-root user for security purposes.
RUN addgroup --system app && adduser --system --group app

# FIX: Explicitly set the HOME environment variable for the 'app' user.
ENV HOME=/home/app

# Set the working directory in the container.
WORKDIR /app

# Create the application directories that will be used for persistent storage.
# We no longer need to chown here, as the VOLUME instruction handles it.
RUN mkdir -p /app/face_database /home/app/.deepface/weights

# --- KEY CHANGE FOR YOUR DEPLOYMENT SERVICE ---
# Declare the directories that should be mounted to persistent disks.
# Your deployment platform will use this information.
VOLUME ["/app/face_database", "/home/app/.deepface/weights"]

# Copy the requirements file.
COPY requirements.txt .

# Install Python dependencies.
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code.
COPY . .

# Set permissions for the app code after copying.
RUN chown -R app:app /app

# Switch to the non-root user.
USER app

# Expose the port that Streamlit runs on.
EXPOSE 8501

# The command to run the application.
CMD ["streamlit", "run", "app.py"]
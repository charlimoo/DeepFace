# Dockerfile for Declarative Deployment Services (FINAL)

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

# Create a non-root user.
# The --create-home flag will automatically create /home/app
RUN addgroup --system app && adduser --system --group --create-home app

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
# This is the most important step.
# It ensures that the app user owns the work directory AND the volume mount points.
# We run this as root BEFORE switching to the app user.
RUN chown -R app:app /app /home/app

# Declare the directories that should be mounted to persistent disks.
# This must come BEFORE switching the user if we need to set permissions.
VOLUME ["/app/face_database", "/home/app/.deepface/weights"]

# Switch to the non-root user for security.
USER app

# Expose the port that Streamlit runs on.
EXPOSE 8501

# The command to run the application.
CMD ["streamlit", "run", "app.py", "--server.enableCORS=false", "--server.enableXsrfProtection=false"]
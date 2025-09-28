# Use a standard Python image
FROM python:3.10

# Set the working directory
WORKDIR /app

# Install system dependencies required by OpenCV (these are critical)
# Note: Using python:3.10-slim may sometimes cause issues with these packages.
# python:3.10 (non-slim) often has better compatibility out of the box.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgl1-mesa-glx \
        libsm6 \
        libxext6 \
        libxrender1 \
        gfortran \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file and install dependencies early
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code and assets
COPY . .

# Expose the port
EXPOSE 8000

# Set the command to run the application
CMD ["python", "traffic_full.py"]
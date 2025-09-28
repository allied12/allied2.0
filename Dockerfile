# Use a standard Python image
FROM python:3.10

# Set the working directory
WORKDIR /app

# Install system dependencies required by OpenCV
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgl1-mesa-glx \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file and install dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy your entire project
COPY . /app

# Create data directory for logging
RUN mkdir -p /app/data

# Expose port 8000
EXPOSE 8000

# Run the FastAPI app
CMD ["uvicorn", "traffic_full:app", "--host", "0.0.0.0", "--port", "8000"]

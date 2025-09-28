# Base Python image
FROM python:3.10

# Set working directory
WORKDIR /app

# Install OpenCV dependencies
RUN apt-get update && \
    apt-get install -y libgl1-mesa-glx libsm6 libxext6 libxrender1 && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy all project files
COPY . /app

# Create folder for data/logs
RUN mkdir -p /app/data

# Expose FastAPI port
EXPOSE 8000

# Run FastAPI app
CMD ["uvicorn", "traffic_full:app", "--host", "0.0.0.0", "--port", "8000"]

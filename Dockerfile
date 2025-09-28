# Use standard Python 3.10 image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install only essential system dependencies for OpenCV
RUN apt-get update && \
    apt-get install -y libgl1-mesa-glx libsm6 libxext6 && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . .

# Create folder for logs/db
RUN mkdir -p /app/data

# Expose port
EXPOSE 8000

# Start FastAPI app
CMD ["uvicorn", "traffic_full:app", "--host", "0.0.0.0", "--port", "8000"]

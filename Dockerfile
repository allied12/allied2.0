# Use a standard Python image
FROM python:3.10

# Set working directory
WORKDIR /app

# Install only the necessary system dependencies for OpenCV
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgl1-mesa-glx \
        libsm6 \
        libxext6 \
        libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python packages
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy the rest of the project files
COPY . /app

# Create a folder for logs and database inside container
RUN mkdir -p /app/data

# Expose the port FastAPI will run on
EXPOSE 8000

# Run FastAPI app with Uvicorn
CMD ["uvicorn", "traffic_full:app", "--host", "0.0.0.0", "--port", "8000"]

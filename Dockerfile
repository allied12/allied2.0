# Use a standard Python image
FROM python:3.10

# Set the working directory
WORKDIR /app

# Install system dependencies required by OpenCV
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgl1-mesa-glx \
        libsm6 \
        libxext6 \
        libxrender1 \
        gfortran && \
    rm -rf /var/lib/apt/lists/*

# Copy the requirements file and install dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy the rest of the application code and assets
COPY . /app

# Expose the port
EXPOSE 8000

# Set the command to run the application
CMD ["uvicorn", "traffic_full:app", "--host", "0.0.0.0", "--port", "8000"]

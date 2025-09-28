# Use a Python image with OpenCV precompiled (headless)
FROM python:3.10

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . .

# Create folder for logs/db
RUN mkdir -p /app/data

# Expose FastAPI port
EXPOSE 8000

# Start FastAPI app
CMD ["uvicorn", "traffic_full:app", "--host", "0.0.0.0", "--port", "8000"]

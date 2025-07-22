# Use Python 3.11 slim image as base
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONPATH=/app
ENV HOST=0.0.0.0
ENV PORT=8491
ENV PYTHONUNBUFFERED=1

# Create necessary directories first
RUN mkdir -p logs qdrant_data

# Copy requirements file first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire backend code
COPY . .

# Expose the port
EXPOSE 8491

# Run the application
CMD ["python", "app.py"] 
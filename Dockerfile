# TitanFrame Studio - Production Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY . /app

# Install python dependencies and titanframe package
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir pandas polars pytest && \
    pip install --no-cache-dir -e .

# Expose web dashboard port
EXPOSE 8080

# Environment configuration
ENV PORT=8080
ENV PYTHONUNBUFFERED=1

# Launch TitanFrame Studio Web Server & Engine Telemetry
CMD ["python", "run_ecom_dashboard.py"]

FROM python:3.12-slim

# Prevent Python from writing .pyc files and enable unbuffered output logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install system build dependencies and curl for healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install application dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application codebase
COPY . .

# Ensure data directory exists for SQLite database and vector indices
RUN mkdir -p /app/data

EXPOSE 8000

# Default command: launch HTTP API service via Uvicorn
CMD ["uvicorn", "interface.http_api:app", "--host", "0.0.0.0", "--port", "8000"]

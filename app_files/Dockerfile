# Use an official Python base image
FROM python:3.11-slim

# Work directory in the container
WORKDIR /app

# System deps (if needed). Often you can skip or just add what torch needs.
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     build-essential \
#  && rm -rf /var/lib/apt/lists/*

# Copy dependency list
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy your code
COPY . .

# Tell Cloud Run that we listen on $PORT
ENV PORT=8080

# Run via uvicorn; note the use of $PORT
CMD ["sh", "-c", "uvicorn ssd_demo_app:app --host 0.0.0.0 --port ${PORT}"]

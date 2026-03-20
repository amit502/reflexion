FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy only the requirements file from its subdirectory
COPY hotpotqa_runs/requirements.txt ./requirements.txt

# Install Python dependencies
# RUN pip install --no-cache-dir -r requirements.txt

# Default command (overridden by the job YAML at runtime)
CMD ["bash"]
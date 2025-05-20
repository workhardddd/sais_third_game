# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir torch numpy pandas tqdm biopython scikit-learn

# Create necessary directories
RUN mkdir -p saisresult

# Verify gvp_src directory exists
RUN if [ ! -d "gvp_src" ]; then \
    echo "Error: gvp_src directory not found!" && exit 1; \
    fi

# Make port 80 available to the world outside this container
EXPOSE 80

# Run main_competion.py when the container launches
CMD ["python", "main_competion.py"] 
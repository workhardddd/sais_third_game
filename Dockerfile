# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /app

# Create required directories
RUN mkdir -p /app /saisresult /saisdata

# Copy the current directory contents into the container at /app
COPY . /app/

# 设置pip国内源
RUN pip config set global.index-url https://mirrors.aliyun.com/pypi/simple

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch and related packages
RUN pip install --no-cache-dir torch torchvision torchaudio

# Install PyTorch Geometric and its dependencies
RUN pip install --no-cache-dir torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-1.7.0+cu101.html
RUN pip install --no-cache-dir torch-geometric

# Install other Python dependencies
RUN pip install --no-cache-dir \
    numpy>=1.19.0 \
    biotite>=0.36.0 \
    biopython>=1.79 \
    scikit-learn>=0.24.0 \
    tqdm>=4.50.0 \
    pandas

# Verify gvp_src directory exists
RUN if [ ! -d "/app/gvp_src" ]; then \
    echo "Error: gvp_src directory not found!" && exit 1; \
    fi

# Make port 80 available to the world outside this container
EXPOSE 80

# Set environment variables
ENV INPUT_DIR=/saisdata
ENV OUTPUT_DIR=/saisresult
ENV OUTPUT_FILE=submit.csv

# Run the script
CMD ["/bin/bash", "/app/run.sh"] 
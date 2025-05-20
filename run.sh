#!/bin/bash

# Create virtual environment
python -m venv venv

# Activate virtual environment
source venv/bin/activate

# Install requirements
pip install torch numpy pandas tqdm biopython scikit-learn

# Create necessary directories
mkdir -p saisresult

# Copy gvp_src directory if it doesn't exist
if [ ! -d "gvp_src" ]; then
    echo "Error: gvp_src directory not found!"
    exit 1
fi

# Run the model
python main_competion.py

# Deactivate virtual environment
deactivate 
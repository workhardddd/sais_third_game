#!/bin/bash

# Create virtual environment
python -m venv venv

# Activate virtual environment
source venv/bin/activate

# Install requirements
pip install torch numpy pandas tqdm biopython scikit-learn

# Verify gvp_src directory exists
if [ ! -d "/app/gvp_src" ]; then
    echo "Error: gvp_src directory not found!"
    exit 1
fi

# Verify input directory exists
if [ ! -d "$INPUT_DIR" ]; then
    echo "Error: Input directory $INPUT_DIR not found!"
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Run the model
python /app/main_competion.py

# Deactivate virtual environment
deactivate 
deactivate 
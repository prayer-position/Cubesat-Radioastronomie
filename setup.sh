#!/bin/bash

echo "--- Starting GMSK Project Setup ---"

# 1. Create a virtual environment named 'venv'
python -m venv venv

# 2. Activate the virtual environment
# Note: In Git Bash on Windows, use this specific path
source venv/Scripts/activate

# 3. Upgrade pip
pip install --upgrade pip

# 4. Install requirements
if [ -f requirements.txt ]; then
    pip install -r requirements.txt
    echo "--- Libraries installed successfully! ---"
else
    echo "--- Error: requirements.txt not found. ---"
fi

echo "--- Setup Complete. To start, type: source venv/Scripts/activate ---

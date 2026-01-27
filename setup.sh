#!/bin/bash

echo "--- Starting GMSK Project Setup ---"

# Create a virtual environment named $ENV_NAME
ENV_NAME="venv_cubesat"
if [ ! -d "$ENV_NAME" ]; then
	python -m venv $ENV_NAME
	echo "Environment created"
fi

# Activate the virtual environment

source $ENV_NAME/Scripts/activate

# Upgrade pip
pip3 install --upgrade pip

# Install requirements
if [ -f requirements.txt ]; then
    pip install -r requirements.txt
    echo "--- Libraries installed successfully! ---"
else
    echo "--- Error: requirements.txt not found. ---"
fi

chmod 755 setup.sh

echo "--- Setup Complete. To start, type: source $ENV_NAME/Scripts/activate ---"

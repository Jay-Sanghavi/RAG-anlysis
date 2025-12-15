#!/bin/bash

# Quick Start Script for RAG Analysis Project
# This script helps set up and run the project

set -e

echo "=========================================="
echo "RAG Analysis - Quick Start"
echo "=========================================="

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo ""
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "✓ Virtual environment created"
fi

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source venv/bin/activate
echo "✓ Virtual environment activated"

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt
echo "✓ Dependencies installed"

# Create necessary directories
echo ""
echo "Creating project directories..."
mkdir -p data logs checkpoints experiments
echo "✓ Directories created"

# Run setup test
echo ""
echo "Testing setup..."
python test_setup.py

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Review configuration: config/config.yaml"
echo "  2. Run test: python test_setup.py"
echo "  3. Start experiment: python main.py"
echo "  4. Or use notebooks: jupyter notebook"
echo ""
echo "To activate the virtual environment in the future:"
echo "  source venv/bin/activate"
echo ""

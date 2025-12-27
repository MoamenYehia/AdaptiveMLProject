#!/bin/bash
# Shell script to run the Streamlit application
# macOS/Linux users: Run with: bash run.sh

echo ""
echo "===================================="
echo "Adaptive Fraud Detection System"
echo "===================================="
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed"
    echo "Please install Python 3.8+ from https://www.python.org/"
    exit 1
fi

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Virtual environment not found. Creating one..."
    python3 -m venv venv
    source venv/bin/activate
    echo "Installing dependencies..."
    pip install -r requirements.txt
else
    source venv/bin/activate
fi

echo ""
echo "Starting Streamlit application..."
echo "Opening in browser at http://localhost:8501"
echo ""

streamlit run src/main.py

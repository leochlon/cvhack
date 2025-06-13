#!/bin/bash

# CV Service Startup Script

echo "Starting CV Hacker Service..."

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed or not in PATH"
    exit 1
fi

# Check if pip is available
if ! command -v pip3 &> /dev/null; then
    echo "Error: pip3 is not installed or not in PATH"
    exit 1
fi

# Install requirements if they don't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

echo "Activating virtual environment..."
source venv/bin/activate

echo "Installing requirements..."
pip install -r requirements.txt

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "Warning: .env file not found. Please create one based on the example."
    echo "The service may not work properly without proper configuration."
fi

# Check if LaTeX is installed
if ! command -v pdflatex &> /dev/null; then
    echo "Warning: LaTeX (pdflatex) is not installed."
    echo "PDF generation will not work without LaTeX."
    echo "On macOS, install with: brew install mactex"
    echo "On Ubuntu, install with: sudo apt-get install texlive-latex-base texlive-fonts-recommended"
fi

echo "Starting CV Hacker Service..."
uvicorn cv_service:app --host 0.0.0.0 --port 8000 --reload

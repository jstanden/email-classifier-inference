#!/bin/bash

# Email Classifier Inference API Startup Script

echo "Starting Email Classifier Inference API..."
echo "=========================================="

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed or not in PATH"
    exit 1
fi

# Check if requirements are installed
echo "Checking dependencies..."
if ! python3 -c "import fastapi, transformers, torch" 2>/dev/null; then
    echo "Installing dependencies..."
    pip install -r requirements.txt
fi

# Check if model directory exists
if [ ! -d "models/email_classifier_final" ]; then
    echo "❌ Model directory not found at models/email_classifier_final"
    echo "Please ensure the model files are in the correct location"
    exit 1
fi

echo "✅ Dependencies and model files verified"
echo "🚀 Starting server on http://localhost:8000"
echo "📖 API documentation will be available at http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Start the server
python3 main.py 
#!/bin/bash
"""
ENHANCED PIP SYSTEM - ACTIVATION SCRIPT
======================================
Activates virtual environment and runs the Enhanced PIP System.
"""

echo "Enhanced PIP System - Activation Script"
echo "======================================"

# Check if virtual environment exists
if [ -d ".venv" ]; then
    echo "✓ Virtual environment found"
    source .venv/bin/activate
    echo "✓ Virtual environment activated"
elif [ -d "../venv" ]; then
    echo "✓ Virtual environment found in parent directory"
    source ../venv/bin/activate
    echo "✓ Virtual environment activated"
else
    echo "⚠ No virtual environment found. Creating one..."
    python3 -m venv .venv
    source .venv/bin/activate
    echo "✓ Virtual environment created and activated"
fi

# Install/upgrade dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "Virtual environment is now active!"
echo "Available commands:"
echo "  python enhanced_pip_system.py --input 0    # Run with camera"
echo "  python demo_enhanced_pip.py                # Run interactive demo"
echo "  python test_enhanced_pip.py                # Run test suite"
echo "  python setup_enhanced_pip.py               # Run setup validation"
echo ""
echo "To deactivate: deactivate"
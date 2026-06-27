#!/bin/bash
# Setup Script for Modular CV System
# ═══════════════════════════════════════════════════════════════════════════════

echo "🔧 Setting up Modular CV System..."
echo

# Check if we're in a virtual environment
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo "⚠️  Warning: Not in a virtual environment"
    echo "   Consider creating one with: python -m venv venv && source venv/bin/activate"
    echo
fi

# Install required packages
echo "📦 Installing required packages..."
pip install -r requirements.txt

echo
echo "✅ Setup complete!"
echo
echo "🚀 To run the system:"
echo "   python main.py"
echo
echo "🧪 To test the system:"
echo "   python test_system.py"
echo
echo "📖 For more options:"
echo "   python main.py --help"
echo
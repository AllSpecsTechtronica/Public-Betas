#!/bin/bash
# Launcher Script for Modular CV System
# ═══════════════════════════════════════════════════════════════════════════════

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}🚀 Modular CV System Launcher${NC}"
echo "========================================"

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

echo "📁 Working directory: $SCRIPT_DIR"

# Check if virtual environment is activated
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo -e "${YELLOW}⚠️  No virtual environment detected${NC}"
    echo "   Consider activating your environment first:"
    echo "   source /path/to/your/venv/bin/activate"
    echo
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check if Python is available
if ! command -v python &> /dev/null; then
    echo -e "${RED}❌ Python not found${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Python found: $(python --version)${NC}"

# Check for PyQt5
if ! python -c "import PyQt5" &> /dev/null; then
    echo -e "${RED}❌ PyQt5 not found${NC}"
    echo "Install with: pip install PyQt5"
    exit 1
fi

echo -e "${GREEN}✅ PyQt5 available${NC}"

# Parse arguments
ARGS=""
DEBUG_MODE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --debug|-d)
            DEBUG_MODE="--debug"
            ARGS="$ARGS $1"
            shift
            ;;
        --camera|-c)
            ARGS="$ARGS $1 $2"
            shift 2
            ;;
        --model|-m)
            ARGS="$ARGS $1 $2"
            shift 2
            ;;
        --help|-h)
            python main.py --help
            exit 0
            ;;
        *)
            ARGS="$ARGS $1"
            shift
            ;;
    esac
done

# Show startup info
echo "========================================"
echo -e "${GREEN}🎯 Starting Modular CV System...${NC}"

if [[ ! -z "$DEBUG_MODE" ]]; then
    echo -e "${YELLOW}🔧 Debug mode enabled${NC}"
fi

echo "📋 Arguments: $ARGS"
echo "========================================"

# Launch the application
python main.py $ARGS

echo
echo -e "${GREEN}👋 Application closed${NC}"
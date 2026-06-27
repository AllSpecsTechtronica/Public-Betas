#!/usr/bin/env python3
# Simplified Launcher for Modular CV System
# ═══════════════════════════════════════════════════════════════════════════════

"""
Simplified launcher that handles common issues and provides a better user experience.
"""

import sys
import os

def check_and_run():
    """Check system and run the main application."""
    
    print("🚀 Starting Modular CV System...")
    
    # Add current directory to path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, current_dir)
    
    # Check PyQt5 availability
    try:
        from PyQt5.QtWidgets import QApplication
        print("✅ PyQt5 GUI framework available")
    except ImportError:
        print("❌ PyQt5 not found. Install with: pip install PyQt5")
        return 1
    
    # Check OpenCV
    try:
        import cv2
        print("✅ OpenCV available")
    except ImportError:
        print("❌ OpenCV not found. Install with: pip install opencv-python")
        return 1
    
    # Check YOLO
    try:
        from ultralytics import YOLO
        print("✅ YOLO available")
    except ImportError:
        print("❌ Ultralytics YOLO not found. Install with: pip install ultralytics")
        return 1
    
    # Import and run main application
    try:
        from main import main
        print("✅ All components loaded successfully")
        print("🎯 Launching application...")
        return main()
    except Exception as e:
        print(f"❌ Failed to start application: {e}")
        print("\nTry running with debug mode:")
        print("  python main.py --debug")
        return 1

if __name__ == '__main__':
    sys.exit(check_and_run())
#!/usr/bin/env python3
# Test Script for Modular CV System
# ═══════════════════════════════════════════════════════════════════════════════

"""
Test script to validate that the modular CV system is properly set up.
This can be run to check if all components are working without starting the GUI.
"""

import sys
import os

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

def test_imports():
    """Test all module imports."""
    print("🔍 Testing module imports...")
    
    try:
        from core.controller import CVController
        print("✅ Core controller imported successfully")
        
        from core.config import DEBUG_MODE, MODEL_PATH, CONF, IOU
        print("✅ Configuration imported successfully")
        
        from core.session_tracker import SessionTracker
        print("✅ Session tracker imported successfully")
        
        from detection.yolo_detector import BackgroundObjectDetector
        print("✅ YOLO detector imported successfully")
        
        from detection.trackers import KalmanBoxTracker, AdvancedHighlightTracker
        print("✅ Tracking systems imported successfully")
        
        from ai_integration.openai_client import OpenAIClient
        print("✅ OpenAI integration imported successfully")
        
        from utils.debug import debug_print
        from utils.image_processing import encode_image_from_array
        print("✅ Utility modules imported successfully")
        
        # Test PyQt5 imports (might fail in headless environments)
        try:
            from gui.main_window import MainWindow
            from gui.video_widget import VideoWidget
            from gui.entity_panel import EntityPanel
            print("✅ GUI modules imported successfully")
        except Exception as e:
            print(f"⚠️  GUI modules import failed (may be normal in headless environment): {e}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_configuration():
    """Test configuration loading."""
    print("\n🔍 Testing configuration...")
    
    try:
        from core.config import MODEL_PATH, CONF, IOU, IMG_SIZE, DEBUG_MODE
        
        print(f"✅ Model path: {MODEL_PATH}")
        print(f"✅ Confidence threshold: {CONF}")
        print(f"✅ IOU threshold: {IOU}")
        print(f"✅ Image size: {IMG_SIZE}")
        print(f"✅ Debug mode: {DEBUG_MODE}")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return False

def test_session_tracker():
    """Test session tracking functionality."""
    print("\n🔍 Testing session tracker...")
    
    try:
        from core.session_tracker import SessionTracker
        
        tracker = SessionTracker()
        duration = tracker.get_session_duration()
        cost_info = tracker.get_total_session_cost()
        
        print(f"✅ Session duration: {duration:.2f} minutes")
        print(f"✅ Total analyses: {tracker.analyses_count}")
        print(f"✅ Total cost: ${cost_info['total_cost_dollars']:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Session tracker error: {e}")
        return False

def test_dependencies():
    """Test required dependencies."""
    print("\n🔍 Testing dependencies...")
    
    dependencies = [
        ('cv2', 'OpenCV'),
        ('numpy', 'NumPy'),
        ('ultralytics', 'Ultralytics YOLO'),
    ]
    
    optional_dependencies = [
        ('PyQt5', 'PyQt5 GUI Framework'),
        ('openai', 'OpenAI API Client'),
    ]
    
    all_good = True
    
    for module, name in dependencies:
        try:
            __import__(module)
            print(f"✅ {name} available")
        except ImportError:
            print(f"❌ {name} missing - install with: pip install {module}")
            all_good = False
    
    for module, name in optional_dependencies:
        try:
            __import__(module)
            print(f"✅ {name} available")
        except ImportError:
            print(f"⚠️  {name} missing (optional) - install with: pip install {module}")
    
    return all_good

def test_model_file():
    """Test if YOLO model file exists."""
    print("\n🔍 Testing model file...")
    
    try:
        from core.config import MODEL_PATH
        
        if os.path.exists(MODEL_PATH):
            size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)
            print(f"✅ Model file found: {MODEL_PATH} ({size_mb:.1f} MB)")
            return True
        else:
            print(f"⚠️  Model file not found: {MODEL_PATH}")
            print("   You can still run the system by specifying a different model with --model")
            return False
            
    except Exception as e:
        print(f"❌ Model file check error: {e}")
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("Modular CV System - Component Test")
    print("=" * 60)
    
    tests = [
        ("Module Imports", test_imports),
        ("Configuration", test_configuration),
        ("Session Tracker", test_session_tracker),
        ("Dependencies", test_dependencies),
        ("Model File", test_model_file),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"❌ {test_name} test failed with exception: {e}")
    
    print("\n" + "=" * 60)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The modular CV system is ready to run.")
        print("\nTo start the application:")
        print("  python main.py")
        print("\nFor help:")
        print("  python main.py --help")
    else:
        print("⚠️  Some tests failed. Please address the issues above.")
        print("\nYou may still be able to run the system with:")
        print("  python main.py --debug")
    
    print("=" * 60)
    
    return 0 if passed == total else 1

if __name__ == '__main__':
    sys.exit(main())
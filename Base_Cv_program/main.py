#!/usr/bin/env python3
# Standalone Main Entry Point for Modular CV System
# ═══════════════════════════════════════════════════════════════════════════════

"""
Modular PyQt-based Computer Vision System
Converted from the original monolithic cv.py web-based system

Features:
- YOLO object detection with tracking
- Real-time video processing
- PyQt5 native desktop interface
- OpenAI integration for AI analysis
- Session tracking and cost monitoring
- Pose estimation support
- Modular architecture for maintainability

Requirements:
- PyQt5
- OpenCV
- NumPy
- Ultralytics YOLO
- OpenAI (optional)

Usage:
    python main.py [--camera CAMERA_INDEX] [--model MODEL_PATH] [--debug]
"""

import sys
import os
import argparse
from PyQt5.QtWidgets import QApplication, QMessageBox
from PyQt5.QtCore import Qt

# Change to script directory and add to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_dir)  # Ensure we're in the right directory
sys.path.insert(0, current_dir)

# Import modular components
try:
    from core.controller import CVController
    from core.config import DEBUG_MODE, MODEL_PATH
    from gui.main_window import MainWindow
    from utils.debug import debug_print
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running from the Base_Cv_program directory")
    sys.exit(1)


class ModularCVApplication:
    """Main application class for the modular CV system."""
    
    def __init__(self, args):
        self.args = args
        self.app = None
        self.main_window = None
        self.controller = None
        
    def setup_application(self):
        """Set up the PyQt application."""
        # Enable high DPI scaling BEFORE creating QApplication
        if hasattr(Qt, 'AA_EnableHighDpiScaling'):
            QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
        if hasattr(Qt, 'AA_UseHighDpiPixmaps'):
            QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
        
        # Create QApplication
        self.app = QApplication(sys.argv)
        self.app.setApplicationName("Modular CV System")
        self.app.setOrganizationName("CV Lab")
        
        # Set application style
        self.app.setStyle("Fusion")  # Modern look
        
        debug_print("[MAIN_APP] PyQt application initialized")
    
    def setup_mvc_components(self):
        """Set up Model-View-Controller components."""
        # Create main window (View)
        self.main_window = MainWindow()
        
        # Create controller (Controller + Model)
        self.controller = CVController(self.main_window)
        
        # Store model path for controller
        self.controller.model_path = getattr(self.args, 'model', MODEL_PATH)
        
        # Connect main window to controller
        self.main_window.controller = self.controller
        
        # Connect signals
        self.main_window.detection_settings_changed.connect(self.on_detection_settings_changed)
        self.main_window.display_settings_changed.connect(self.on_display_settings_changed)
        
        debug_print("[MAIN_APP] MVC components initialized")
    
    def on_detection_settings_changed(self):
        """Handle detection settings changes from GUI."""
        settings = self.main_window.get_detection_settings()
        self.controller.update_detection_settings(settings)
    
    def on_display_settings_changed(self):
        """Handle display settings changes from GUI."""
        settings = self.main_window.get_display_settings()
        self.controller.update_display_settings(settings)
    
    def check_dependencies(self):
        """Check if all required dependencies are available."""
        missing_deps = []
        
        try:
            import cv2
        except ImportError:
            missing_deps.append("opencv-python")
        
        try:
            import numpy
        except ImportError:
            missing_deps.append("numpy")
        
        try:
            from ultralytics import YOLO
        except ImportError:
            missing_deps.append("ultralytics")
        
        try:
            from PyQt5 import QtWidgets
        except ImportError:
            missing_deps.append("PyQt5")
        
        if missing_deps:
            error_msg = f"Missing required dependencies: {', '.join(missing_deps)}\n\n"
            error_msg += "Please install them using:\n"
            error_msg += f"pip install {' '.join(missing_deps)}"
            
            if self.app:
                QMessageBox.critical(None, "Missing Dependencies", error_msg)
            else:
                print(f"ERROR: {error_msg}")
            return False
        
        return True
    
    def check_model_file(self):
        """Check if the YOLO model file exists."""
        model_path = self.args.model if hasattr(self.args, 'model') and self.args.model else MODEL_PATH
        
        if not os.path.exists(model_path):
            # Try to download a default model
            try:
                from ultralytics import YOLO
                print(f"⚠️  Model file not found: {model_path}")
                print("🔄 Attempting to download a default YOLO model...")
                
                # Create models directory
                model_dir = os.path.dirname(model_path)
                if model_dir and not os.path.exists(model_dir):
                    os.makedirs(model_dir, exist_ok=True)
                
                # Download yolo11n.pt as a fallback
                fallback_model = "yolo11n.pt"
                model = YOLO(fallback_model)
                print(f"✅ Downloaded fallback model: {fallback_model}")
                
                # Update the model path in args for the application to use
                self.args.model = fallback_model
                return True
                
            except Exception as e:
                error_msg = f"YOLO model file not found: {model_path}\n\n"
                error_msg += f"Failed to download fallback model: {e}\n\n"
                error_msg += "Please specify a valid model path using --model"
                
                if self.app:
                    QMessageBox.warning(None, "Model File Not Found", error_msg)
                else:
                    print(f"WARNING: {error_msg}")
                return False
        
        return True
    
    def run(self):
        """Run the application."""
        try:
            # Set up PyQt application
            self.setup_application()
            
            # Check dependencies
            if not self.check_dependencies():
                return 1
            
            # Check model file (warning only, app can still run)
            self.check_model_file()
            
            # Set up MVC components
            self.setup_mvc_components()
            
            # Show main window
            self.main_window.show()
            
            # Start video capture if requested
            camera_index = getattr(self.args, 'camera', 0)
            if self.controller.start_video_capture(camera_index):
                debug_print(f"[MAIN_APP] Started video capture from camera {camera_index}")
            else:
                debug_print(f"[MAIN_APP] Failed to start video capture from camera {camera_index}")
                # Show warning but continue
                if self.main_window:
                    QMessageBox.warning(self.main_window, "Camera Error", 
                                      f"Failed to open camera {camera_index}. You can try a different camera index in the settings.")
            
            debug_print("[MAIN_APP] Application started successfully")
            
            # Run the application
            result = self.app.exec_()
            
            # Cleanup
            if self.controller:
                self.controller.cleanup()
            
            return result
            
        except KeyboardInterrupt:
            debug_print("[MAIN_APP] Application interrupted by user")
            return 0
        except Exception as e:
            error_msg = f"Application error: {str(e)}"
            debug_print(f"[MAIN_APP] {error_msg}")
            
            if self.app and self.main_window:
                QMessageBox.critical(self.main_window, "Application Error", error_msg)
            else:
                print(f"ERROR: {error_msg}")
            
            return 1


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Modular PyQt-based Computer Vision System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python main.py                          # Use default camera (0)
    python main.py --camera 1               # Use camera 1
    python main.py --model custom_model.pt  # Use custom YOLO model
    python main.py --debug                  # Enable debug mode
        """
    )
    
    parser.add_argument(
        '--camera', 
        type=int, 
        default=0,
        help='Camera index to use (default: 0)'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default=MODEL_PATH,
        help=f'Path to YOLO model file (default: {MODEL_PATH})'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode with verbose logging'
    )
    
    parser.add_argument(
        '--version',
        action='version',
        version='Modular CV System v1.0'
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    # Parse arguments
    args = parse_arguments()
    
    # Update debug mode if requested
    if args.debug:
        import core.config as config
        config.DEBUG_MODE = True
    
    # Print startup banner
    if DEBUG_MODE:
        print("=" * 70)
        print("Modular PyQt-based Computer Vision System")
        print("Converted from monolithic cv.py web-based system")
        print("=" * 70)
        print(f"Camera: {args.camera}")
        print(f"Model: {args.model}")
        print(f"Debug: {args.debug}")
        print("=" * 70)
    
    # Create and run application
    app = ModularCVApplication(args)
    return app.run()


if __name__ == '__main__':
    sys.exit(main())
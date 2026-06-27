# Main PyQt Application Window
# ═══════════════════════════════════════════════════════════════════════════════

import os
import sys

# Project root on path when this file is run directly (e.g. python gui/main_window.py)
_pkg_dir = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_pkg_dir)
if _root not in sys.path:
    sys.path.insert(0, _root)

from PyQt5.QtWidgets import (QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, 
                             QPushButton, QLabel, QFrame, QTextEdit, QSplitter,
                             QGroupBox, QCheckBox, QSlider, QSpinBox, QComboBox)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QFont, QPalette, QColor

from gui.video_widget import VideoWidget
from gui.entity_panel import EntityPanel
from core.config import DEBUG_MODE
from utils.debug import debug_print


class MainWindow(QMainWindow):
    """Main application window for the modular CV system."""
    
    # Signals
    detection_settings_changed = pyqtSignal()
    display_settings_changed = pyqtSignal()
    
    def __init__(self, controller=None):
        super().__init__()
        self.controller = controller
        self.video_widget = None
        self.entity_panel = None
        
        # UI update timer
        self.ui_timer = QTimer()
        self.ui_timer.timeout.connect(self.update_ui_info)
        self.ui_timer.start(1000)  # Update every second
        
        self.init_ui()
        self.apply_dark_theme()
        
        debug_print("[MAIN_WINDOW] Initialized")
    
    def init_ui(self):
        """Initialize the user interface."""
        self.setWindowTitle("Modular CV System - PyQt Interface")
        self.setGeometry(100, 100, 1600, 1000)
        
        # Create central widget with splitter
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QVBoxLayout(central_widget)
        
        # Create splitter for main content
        main_splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(main_splitter)
        
        # Left side - Video and controls
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        
        # Video widget
        self.video_widget = VideoWidget()
        left_layout.addWidget(self.video_widget)
        
        # Control panel
        control_panel = self.create_control_panel()
        left_layout.addWidget(control_panel)
        
        # Segmentation status indicator
        self.segmentation_status = QLabel("Segmentation: Ready")
        self.segmentation_status.setStyleSheet("""
            QLabel {
                color: #39ff7a;
                font-family: 'Consolas', 'Courier New', monospace;
                font-weight: bold;
                padding: 5px;
                background-color: rgba(0, 0, 0, 0.7);
                border: 1px solid #39ff7a;
                border-radius: 3px;
            }
        """)
        left_layout.addWidget(self.segmentation_status)
        
        main_splitter.addWidget(left_widget)
        
        # Right side - Entity panel and settings
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        
        # Entity panel
        self.entity_panel = EntityPanel()
        right_layout.addWidget(self.entity_panel)
        
        # Settings panel
        settings_panel = self.create_settings_panel()
        right_layout.addWidget(settings_panel)
        
        # Info panel
        info_panel = self.create_info_panel()
        right_layout.addWidget(info_panel)
        
        main_splitter.addWidget(right_widget)
        
        # Set splitter proportions
        main_splitter.setStretchFactor(0, 3)  # Video area gets more space
        main_splitter.setStretchFactor(1, 1)  # Right panel gets less space
        
        # Connect video widget signals if controller is available
        if self.controller:
            self.video_widget.mouse_clicked.connect(self.controller.handle_click)
            self.video_widget.mouse_pressed.connect(self.controller.handle_mouse_down)
            self.video_widget.mouse_moved.connect(self.controller.handle_mouse_move)
            self.video_widget.mouse_released.connect(self.controller.handle_mouse_up)
            self.video_widget.right_clicked.connect(self.controller.handle_right_click)
            self.video_widget.double_clicked.connect(self.controller.handle_double_click)
    
    def create_control_panel(self):
        """Create the control panel with buttons."""
        panel = QFrame()
        panel.setFrameStyle(QFrame.StyledPanel)
        panel.setMaximumHeight(80)
        
        layout = QHBoxLayout(panel)
        
        # Start/Stop button
        self.start_stop_btn = QPushButton("Start Detection")
        self.start_stop_btn.clicked.connect(self.toggle_detection)
        layout.addWidget(self.start_stop_btn)
        
        # Camera selection
        layout.addWidget(QLabel("Camera:"))
        self.camera_combo = QComboBox()
        self.camera_combo.addItems(["Camera 0", "Camera 1", "Camera 2"])
        layout.addWidget(self.camera_combo)
        
        # FPS display
        self.fps_label = QLabel("FPS: 0.0")
        layout.addWidget(self.fps_label)
        
        layout.addStretch()
        
        return panel
    
    def create_settings_panel(self):
        """Create the settings panel."""
        group = QGroupBox("Detection Settings")
        layout = QVBoxLayout(group)
        
        # Confidence threshold
        conf_layout = QHBoxLayout()
        conf_layout.addWidget(QLabel("Confidence:"))
        self.conf_slider = QSlider(Qt.Horizontal)
        self.conf_slider.setRange(1, 100)
        self.conf_slider.setValue(25)
        self.conf_slider.valueChanged.connect(self.on_detection_settings_changed)
        conf_layout.addWidget(self.conf_slider)
        self.conf_value_label = QLabel("0.25")
        conf_layout.addWidget(self.conf_value_label)
        layout.addLayout(conf_layout)
        
        # IOU threshold
        iou_layout = QHBoxLayout()
        iou_layout.addWidget(QLabel("IOU:"))
        self.iou_slider = QSlider(Qt.Horizontal)
        self.iou_slider.setRange(1, 100)
        self.iou_slider.setValue(10)
        self.iou_slider.valueChanged.connect(self.on_detection_settings_changed)
        iou_layout.addWidget(self.iou_slider)
        self.iou_value_label = QLabel("0.10")
        iou_layout.addWidget(self.iou_value_label)
        layout.addLayout(iou_layout)
        
        # Display options
        self.show_titles_cb = QCheckBox("Show Object Titles")
        self.show_titles_cb.stateChanged.connect(self.on_display_settings_changed)
        layout.addWidget(self.show_titles_cb)
        
        self.show_confidence_cb = QCheckBox("Show Confidence Scores")
        self.show_confidence_cb.stateChanged.connect(self.on_display_settings_changed)
        layout.addWidget(self.show_confidence_cb)
        
        self.show_arrows_cb = QCheckBox("Show Detection Arrows")
        self.show_arrows_cb.setChecked(True)
        self.show_arrows_cb.stateChanged.connect(self.on_display_settings_changed)
        layout.addWidget(self.show_arrows_cb)
        
        self.show_corners_cb = QCheckBox("Show Corner Markers")
        self.show_corners_cb.stateChanged.connect(self.on_display_settings_changed)
        layout.addWidget(self.show_corners_cb)
        
        # Pose estimation
        self.pose_estimation_cb = QCheckBox("Enable Pose Estimation")
        self.pose_estimation_cb.stateChanged.connect(self.on_detection_settings_changed)
        layout.addWidget(self.pose_estimation_cb)
        
        return group
    
    def create_info_panel(self):
        """Create the info panel for session tracking."""
        group = QGroupBox("Session Info")
        layout = QVBoxLayout(group)
        
        self.info_text = QTextEdit()
        self.info_text.setMaximumHeight(150)
        self.info_text.setReadOnly(True)
        layout.addWidget(self.info_text)
        
        return group
    
    def apply_dark_theme(self):
        """Apply a dark theme to the application."""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #2b2b2b;
                color: #ffffff;
            }
            
            QGroupBox {
                font-weight: bold;
                border: 2px solid #555555;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
            
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
            
            QPushButton {
                background-color: #404040;
                border: 1px solid #606060;
                border-radius: 3px;
                padding: 5px;
                min-width: 80px;
            }
            
            QPushButton:hover {
                background-color: #505050;
            }
            
            QPushButton:pressed {
                background-color: #303030;
            }
            
            QFrame {
                background-color: #3b3b3b;
                border: 1px solid #555555;
                border-radius: 3px;
            }
            
            QTextEdit {
                background-color: #1e1e1e;
                border: 1px solid #555555;
                border-radius: 3px;
            }
            
            QSlider::groove:horizontal {
                border: 1px solid #555555;
                height: 8px;
                background: #404040;
                margin: 2px 0;
                border-radius: 3px;
            }
            
            QSlider::handle:horizontal {
                background: #39ff7a;
                border: 1px solid #39ff7a;
                width: 18px;
                margin: -2px 0;
                border-radius: 3px;
            }
            
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
            }
            
            QCheckBox::indicator:unchecked {
                border: 2px solid #555555;
                background-color: #2b2b2b;
                border-radius: 3px;
            }
            
            QCheckBox::indicator:checked {
                border: 2px solid #39ff7a;
                background-color: #39ff7a;
                border-radius: 3px;
            }
        """)
    
    def toggle_detection(self):
        """Toggle detection on/off."""
        if self.controller:
            if self.controller.is_detection_running():
                self.controller.stop_detection()
                self.start_stop_btn.setText("Start Detection")
            else:
                model_path = getattr(self.controller, 'model_path', None)
                if self.controller.start_detection(model_path):
                    self.start_stop_btn.setText("Stop Detection")
                else:
                    self.start_stop_btn.setText("Start Detection")
                    # Show error message
                    from PyQt5.QtWidgets import QMessageBox
                    QMessageBox.warning(self, "Detection Error", 
                                      "Failed to start detection. Check the model file and try again.")
    
    def on_detection_settings_changed(self):
        """Handle detection settings changes."""
        # Update value labels
        self.conf_value_label.setText(f"{self.conf_slider.value() / 100.0:.2f}")
        self.iou_value_label.setText(f"{self.iou_slider.value() / 100.0:.2f}")
        
        # Emit signal
        self.detection_settings_changed.emit()
    
    def on_display_settings_changed(self):
        """Handle display settings changes."""
        self.display_settings_changed.emit()
    
    def update_ui_info(self):
        """Update UI information display."""
        if not self.controller:
            return
        
        try:
            # Update FPS
            fps = self.controller.get_detection_fps()
            self.fps_label.setText(f"FPS: {fps:.1f}")
            
            # Update session info
            session_info = self.controller.get_session_info()
            if session_info:
                info_text = f"""Session Duration: {session_info['duration_minutes']:.1f} minutes
Analyses: {session_info['analyses_count']}
Total Tokens: {session_info['total_tokens']:,}
Total Cost: ${session_info['total_cost']['total_cost_dollars']:.4f}
Detection FPS: {fps:.1f}
"""
                self.info_text.setPlainText(info_text)
                
        except Exception as e:
            if DEBUG_MODE:
                debug_print(f"[MAIN_WINDOW] Error updating UI info: {e}")
    
    def get_detection_settings(self):
        """Get current detection settings."""
        return {
            'confidence': self.conf_slider.value() / 100.0,
            'iou': self.iou_slider.value() / 100.0,
            'enable_pose_estimation': self.pose_estimation_cb.isChecked()
        }
    
    def get_display_settings(self):
        """Get current display settings."""
        return {
            'show_object_titles': self.show_titles_cb.isChecked(),
            'show_confidence_scores': self.show_confidence_cb.isChecked(),
            'show_detection_arrows': self.show_arrows_cb.isChecked(),
            'show_corner_markers': self.show_corners_cb.isChecked()
        }
    
    def update_frame(self, frame):
        """Update the video frame display."""
        if self.video_widget:
            self.video_widget.update_frame(frame)
    
    def add_entity_card(self, entity_id, text, status='complete'):
        """Add an entity card to the panel."""
        if self.entity_panel:
            self.entity_panel.add_entity(entity_id, text, status)
    
    def update_entity_card(self, entity_id, text, status='streaming'):
        """Update an existing entity card."""
        if self.entity_panel:
            self.entity_panel.update_entity(entity_id, text, status)
    
    def update_segmentation_status(self, status_text, color="#39ff7a"):
        """Update the segmentation status indicator."""
        if hasattr(self, 'segmentation_status'):
            self.segmentation_status.setText(f"Segmentation: {status_text}")
            self.segmentation_status.setStyleSheet(f"""
                QLabel {{
                    color: {color};
                    font-family: 'Consolas', 'Courier New', monospace;
                    font-weight: bold;
                    padding: 5px;
                    background-color: rgba(0, 0, 0, 0.7);
                    border: 1px solid {color};
                    border-radius: 3px;
                }}
            """)
    
    def closeEvent(self, event):
        """Handle window close event."""
        if self.controller:
            self.controller.cleanup()
        event.accept()
# Video Display Widget for PyQt Interface
# ═══════════════════════════════════════════════════════════════════════════════

import cv2
import numpy as np
import time
from PyQt5.QtWidgets import QWidget, QLabel, QVBoxLayout
from PyQt5.QtCore import Qt, pyqtSignal, QTimer, QEvent
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor

from utils.debug import debug_print


class VideoWidget(QWidget):
    """Custom widget for displaying video frames with mouse interaction."""
    
    # Signals for mouse events
    mouse_clicked = pyqtSignal(int, int)
    mouse_pressed = pyqtSignal(int, int)
    mouse_moved = pyqtSignal(int, int)
    mouse_released = pyqtSignal(int, int)
    right_clicked = pyqtSignal(int, int)
    double_clicked = pyqtSignal(int, int)
    
    def __init__(self):
        super().__init__()
        self.current_frame = None
        self.display_frame = None
        self.frame_scale_x = 1.0
        self.frame_scale_y = 1.0
        
        # Mouse tracking
        self.mouse_pressed_flag = False
        self.last_mouse_pos = None
        
        # Double-click detection
        self.last_click_time = 0
        self.last_click_pos = None
        self.double_click_threshold = 300  # milliseconds
        self.double_click_distance = 10     # pixels
        
        self.init_ui()
        
        debug_print("[VIDEO_WIDGET] Initialized")
    
    def init_ui(self):
        """Initialize the user interface."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Create label for displaying video
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: black; border: 1px solid #555555;")
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setScaledContents(False)
        
        layout.addWidget(self.video_label)
        
        # Enable mouse tracking
        self.setMouseTracking(True)
        self.video_label.setMouseTracking(True)
    
    def update_frame(self, frame):
        """Update the displayed video frame."""
        if frame is None:
            return
        
        try:
            self.current_frame = frame.copy()
            self.display_frame_in_widget(frame)
        except Exception as e:
            debug_print(f"[VIDEO_WIDGET] Error updating frame: {e}")
    
    def display_frame_in_widget(self, frame):
        """Display a frame in the video widget."""
        try:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_frame.shape
            bytes_per_line = ch * w
            
            # Create QImage
            qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            
            # Get widget size
            widget_size = self.video_label.size()
            widget_width = widget_size.width()
            widget_height = widget_size.height()
            
            if widget_width <= 0 or widget_height <= 0:
                return
            
            # Calculate scaling to fit while maintaining aspect ratio
            scale_x = widget_width / w
            scale_y = widget_height / h
            scale = min(scale_x, scale_y)
            
            # Store scaling factors for mouse coordinate conversion
            self.frame_scale_x = scale
            self.frame_scale_y = scale
            
            # Scale the image
            scaled_width = int(w * scale)
            scaled_height = int(h * scale)
            scaled_image = qt_image.scaled(scaled_width, scaled_height, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            
            # Convert to pixmap and display
            pixmap = QPixmap.fromImage(scaled_image)
            self.video_label.setPixmap(pixmap)
            self.display_frame = frame
            
        except Exception as e:
            debug_print(f"[VIDEO_WIDGET] Error displaying frame: {e}")
    
    def mousePressEvent(self, event):
        """Handle mouse press events."""
        x, y = self.convert_widget_to_frame_coords(event.x(), event.y())
        if x is None or y is None:
            return
            
        if event.button() == Qt.LeftButton:
            self.mouse_pressed_flag = True
            self.last_mouse_pos = (x, y)
            self.mouse_pressed.emit(x, y)
        elif event.button() == Qt.RightButton:
            # Right-click for drag selection - enable tracking for right-click drag
            self.mouse_pressed_flag = True
            self.last_mouse_pos = (x, y)
            self.mouse_pressed.emit(x, y)
    
    def mouseMoveEvent(self, event):
        """Handle mouse move events."""
        x, y = self.convert_widget_to_frame_coords(event.x(), event.y())
        if x is not None and y is not None:
            if self.mouse_pressed_flag:
                self.mouse_moved.emit(x, y)
            self.last_mouse_pos = (x, y)
    
    def mouseReleaseEvent(self, event):
        """Handle mouse release events."""
        if (event.button() == Qt.LeftButton or event.button() == Qt.RightButton) and self.mouse_pressed_flag:
            x, y = self.convert_widget_to_frame_coords(event.x(), event.y())
            if x is not None and y is not None:
                self.mouse_pressed_flag = False
                self.mouse_released.emit(x, y)
                
                # Check for double-click
                current_time = time.time() * 1000  # Convert to milliseconds
                if (self.last_click_pos and 
                    current_time - self.last_click_time < self.double_click_threshold):
                    
                    # Check if clicks are close enough
                    last_x, last_y = self.last_click_pos
                    distance = ((x - last_x)**2 + (y - last_y)**2)**0.5
                    if distance < self.double_click_distance:
                        self.double_clicked.emit(x, y)
                        debug_print(f"[VIDEO_WIDGET] Double-click detected at ({x}, {y})")
                        # Reset for next potential double-click
                        self.last_click_time = 0
                        self.last_click_pos = None
                        return
                
                # Single click - also emit click signal if mouse didn't move much
                if self.last_mouse_pos:
                    last_x, last_y = self.last_mouse_pos
                    distance = ((x - last_x)**2 + (y - last_y)**2)**0.5
                    if distance < 5:  # Small movement threshold
                        self.mouse_clicked.emit(x, y)
                        # Store for potential double-click
                        self.last_click_time = time.time() * 1000  # Convert to milliseconds
                        self.last_click_pos = (x, y)
    
    def convert_widget_to_frame_coords(self, widget_x, widget_y):
        """Convert widget coordinates to frame coordinates."""
        try:
            if not self.display_frame is not None or self.frame_scale_x == 0 or self.frame_scale_y == 0:
                return None, None
            
            # Get the video label geometry
            label_rect = self.video_label.geometry()
            label_x = widget_x - label_rect.x()
            label_y = widget_y - label_rect.y()
            
            # Check if click is within the label
            if label_x < 0 or label_y < 0 or label_x >= label_rect.width() or label_y >= label_rect.height():
                return None, None
            
            # Get the pixmap size
            pixmap = self.video_label.pixmap()
            if not pixmap:
                return None, None
            
            pixmap_width = pixmap.width()
            pixmap_height = pixmap.height()
            
            # Calculate offset to center the pixmap within the label
            x_offset = (label_rect.width() - pixmap_width) // 2
            y_offset = (label_rect.height() - pixmap_height) // 2
            
            # Adjust coordinates relative to the pixmap
            pixmap_x = label_x - x_offset
            pixmap_y = label_y - y_offset
            
            # Check if click is within the pixmap
            if pixmap_x < 0 or pixmap_y < 0 or pixmap_x >= pixmap_width or pixmap_y >= pixmap_height:
                return None, None
            
            # Convert to frame coordinates
            frame_x = int(pixmap_x / self.frame_scale_x)
            frame_y = int(pixmap_y / self.frame_scale_y)
            
            # Ensure coordinates are within frame bounds
            if self.display_frame is not None:
                frame_height, frame_width = self.display_frame.shape[:2]
                frame_x = max(0, min(frame_x, frame_width - 1))
                frame_y = max(0, min(frame_y, frame_height - 1))
            
            return frame_x, frame_y
            
        except Exception as e:
            debug_print(f"[VIDEO_WIDGET] Error converting coordinates: {e}")
            return None, None
    
    def get_current_frame(self):
        """Get the current frame being displayed."""
        return self.current_frame
    
    def clear_frame(self):
        """Clear the current frame display."""
        self.video_label.clear()
        self.video_label.setText("No Video")
        self.current_frame = None
        self.display_frame = None
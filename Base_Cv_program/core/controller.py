# Main Controller for Modular CV System
# ═══════════════════════════════════════════════════════════════════════════════

import cv2
import time
import threading
import numpy as np
from queue import Queue

from detection.yolo_detector import BackgroundObjectDetector
from core.openai_bridge import get_openai_client
from core.session_tracker import SessionTracker
from core.config import *
from utils.debug import debug_print

class CVController:
    """Main controller that coordinates all CV system modules."""
    
    def __init__(self, main_window=None):
        self.main_window = main_window
        self.detector = None
        self.session_tracker = SessionTracker()
        
        # Video capture
        self.video_capture = None
        self.capture_thread = None
        self.capture_running = False
        self.capture_lock = threading.Lock()
        
        # Frame processing
        self.current_frame = None
        self.processed_frame = None
        self.frame_lock = threading.Lock()
        
        # Detection state
        self.detection_running = False
        self.detection_fps = 0.0
        
        # Analysis queue for AI integration
        self.analysis_queue = Queue()
        self.analysis_thread = None
        self.analysis_running = False
        
        # Current settings
        self.detection_settings = {
            'confidence': CONF,
            'iou': IOU,
            'enable_pose_estimation': ENABLE_POSE_ESTIMATION
        }
        
        self.display_settings = {
            'show_object_titles': SHOW_OBJECT_TITLES,
            'show_confidence_scores': SHOW_CONFIDENCE_SCORES,
            'show_detection_arrows': SHOW_DETECTION_ARROWS,
            'show_corner_markers': SHOW_CORNER_MARKERS
        }
        
        debug_print("[CV_CONTROLLER] Initialized")
    
    def initialize_detector(self, model_path=None):
        """Initialize the YOLO detector."""
        try:
            if self.detector:
                self.detector.stop()
            
            # Use provided model path or default
            if model_path is None:
                model_path = MODEL_PATH
            
            self.detector = BackgroundObjectDetector(
                model_path=model_path,
                conf=self.detection_settings['confidence'],
                iou=self.detection_settings['iou']
            )
            
            # Update configuration based on current settings
            self.update_detection_settings()
            
            # Connect entity panel callback if main window exists
            if self.main_window and hasattr(self.main_window, 'entity_panel'):
                self._setup_entity_panel_callback()
            
            debug_print(f"[CV_CONTROLLER] Detector initialized with model: {model_path}")
            return True
            
        except Exception as e:
            debug_print(f"[CV_CONTROLLER] Error initializing detector: {e}")
            return False
    
    def start_video_capture(self, camera_index=0):
        """Start video capture from camera."""
        try:
            self.video_capture = cv2.VideoCapture(camera_index)
            if not self.video_capture.isOpened():
                debug_print(f"[CV_CONTROLLER] Failed to open camera {camera_index}")
                return False
            
            # Set camera properties
            self.video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.video_capture.set(cv2.CAP_PROP_FPS, 30)
            
            self.capture_running = True
            self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
            self.capture_thread.start()
            
            debug_print("[CV_CONTROLLER] Video capture started")
            return True
            
        except Exception as e:
            debug_print(f"[CV_CONTROLLER] Error starting video capture: {e}")
            return False
    
    def stop_video_capture(self):
        """Stop video capture."""
        self.capture_running = False
        
        if self.capture_thread:
            self.capture_thread.join(timeout=2.0)
        
        if self.video_capture:
            self.video_capture.release()
            self.video_capture = None
        
        debug_print("[CV_CONTROLLER] Video capture stopped")
    
    def start_detection(self, model_path=None):
        """Start object detection."""
        if not self.initialize_detector(model_path):
            return False
        
        self.detector.start()
        self.detection_running = True
        
        # Start analysis thread
        if not self.analysis_running:
            self.analysis_running = True
            self.analysis_thread = threading.Thread(target=self._analysis_worker, daemon=True)
            self.analysis_thread.start()
        
        debug_print("[CV_CONTROLLER] Detection started")
        return True
    
    def stop_detection(self):
        """Stop object detection."""
        self.detection_running = False
        
        if self.detector:
            self.detector.stop()
            self.detector = None
        
        # Stop analysis thread
        self.analysis_running = False
        if self.analysis_thread:
            self.analysis_thread.join(timeout=2.0)
        
        debug_print("[CV_CONTROLLER] Detection stopped")
    
    def _capture_loop(self):
        """Main video capture loop."""
        while self.capture_running:
            try:
                if self.video_capture and self.video_capture.isOpened():
                    ret, frame = self.video_capture.read()
                    if ret:
                        with self.frame_lock:
                            self.current_frame = frame.copy()
                        
                        # Update detector with new frame
                        if self.detector and self.detection_running:
                            # Update segmentation engine with frame
                            self.detector.update_frame(frame)
                            
                            # Get processed frame with detections
                            processed = self.detector.draw_detections_on_frame(frame)
                            with self.frame_lock:
                                self.processed_frame = processed
                            
                            # Update FPS
                            self.detection_fps = self.detector.get_detection_fps()
                        else:
                            with self.frame_lock:
                                self.processed_frame = frame.copy()
                        
                        # Update GUI
                        if self.main_window:
                            display_frame = self.processed_frame if self.processed_frame is not None else frame
                            self.main_window.update_frame(display_frame)
                    else:
                        debug_print("[CV_CONTROLLER] Failed to read frame")
                        time.sleep(0.1)
                else:
                    time.sleep(0.1)
                    
            except Exception as e:
                debug_print(f"[CV_CONTROLLER] Error in capture loop: {e}")
                time.sleep(0.1)
    
    def _analysis_worker(self):
        """Worker thread for AI analysis tasks."""
        while self.analysis_running:
            try:
                # Check if there are any analysis tasks
                if not self.analysis_queue.empty():
                    task = self.analysis_queue.get(timeout=1.0)
                    self._process_analysis_task(task)
                else:
                    time.sleep(0.1)
                    
            except Exception as e:
                debug_print(f"[CV_CONTROLLER] Error in analysis worker: {e}")
                time.sleep(0.1)
    
    def _process_analysis_task(self, task):
        """Process an AI analysis task."""
        try:
            image_crop = task['image']
            entity_id = task['entity_id']
            
            client = get_openai_client()
            if client and client.is_available():
                # Start streaming analysis
                if self.main_window:
                    self.main_window.add_entity_card(entity_id, "Analyzing...", "streaming")
                
                def analysis_callback(delta_text, full_text, status):
                    if self.main_window:
                        self.main_window.update_entity_card(entity_id, full_text, status)
                
                # Run async analysis (this is a simplified version)
                # In a full implementation, you'd want to properly handle async/await
                result = client.analyze_image(image_crop)
                if result and self.main_window:
                    self.main_window.update_entity_card(entity_id, result, "complete")
                    
        except Exception as e:
            debug_print(f"[CV_CONTROLLER] Error processing analysis task: {e}")
    
    # Mouse handling methods are already defined above - removing duplicates
    
    def queue_analysis_for_region(self, x, y, width=200, height=200):
        """Queue an analysis task for a region around the clicked point."""
        try:
            with self.frame_lock:
                if self.current_frame is None:
                    return
                
                frame = self.current_frame.copy()
            
            # Extract region around click point
            h, w = frame.shape[:2]
            x1 = max(0, x - width // 2)
            y1 = max(0, y - height // 2)
            x2 = min(w, x1 + width)
            y2 = min(h, y1 + height)
            
            region = frame[y1:y2, x1:x2]
            
            if region.size > 0:
                entity_id = int(time.time() * 1000) % 100000  # Simple ID generation
                task = {
                    'image': region,
                    'entity_id': entity_id,
                    'coordinates': (x, y),
                    'bbox': (x1, y1, x2, y2)
                }
                
                self.analysis_queue.put(task)
                debug_print(f"[CV_CONTROLLER] Queued analysis task for region at ({x}, {y})")
                
        except Exception as e:
            debug_print(f"[CV_CONTROLLER] Error queuing analysis: {e}")
    
    def update_detection_settings(self, settings=None):
        """Update detection settings."""
        if settings:
            self.detection_settings.update(settings)
        
        # Apply settings to detector
        if self.detector:
            self.detector.conf = self.detection_settings['confidence']
            self.detector.iou = self.detection_settings['iou']
            
            # Update pose estimation
            if hasattr(self.detector, 'pose_estimator'):
                if self.detection_settings['enable_pose_estimation'] and not self.detector.pose_estimator:
                    # Enable pose estimation
                    try:
                        from PoseEstimation import PoseEstimator
                        self.detector.pose_estimator = PoseEstimator()
                    except:
                        pass
                elif not self.detection_settings['enable_pose_estimation'] and self.detector.pose_estimator:
                    # Disable pose estimation
                    self.detector.pose_estimator = None
        
        debug_print(f"[CV_CONTROLLER] Updated detection settings: {self.detection_settings}")
    
    def update_display_settings(self, settings=None):
        """Update display settings."""
        if settings:
            self.display_settings.update(settings)
        
        # Update global configuration
        global SHOW_OBJECT_TITLES, SHOW_CONFIDENCE_SCORES, SHOW_DETECTION_ARROWS, SHOW_CORNER_MARKERS
        SHOW_OBJECT_TITLES = self.display_settings['show_object_titles']
        SHOW_CONFIDENCE_SCORES = self.display_settings['show_confidence_scores']
        SHOW_DETECTION_ARROWS = self.display_settings['show_detection_arrows']
        SHOW_CORNER_MARKERS = self.display_settings['show_corner_markers']
        
        debug_print(f"[CV_CONTROLLER] Updated display settings: {self.display_settings}")
    
    def is_detection_running(self):
        """Check if detection is currently running."""
        return self.detection_running
    
    def is_capture_running(self):
        """Check if video capture is currently running."""
        return self.capture_running
    
    def get_detection_fps(self):
        """Get current detection FPS."""
        return self.detection_fps
    
    def get_session_info(self):
        """Get session information."""
        client = get_openai_client()
        if client:
            return client.get_session_info()
        return self.session_tracker.get_total_session_cost() if self.session_tracker else None
    
    def get_current_frame(self):
        """Get the current video frame."""
        with self.frame_lock:
            return self.current_frame.copy() if self.current_frame is not None else None
    
    def get_processed_frame(self):
        """Get the current processed frame with detections."""
        with self.frame_lock:
            return self.processed_frame.copy() if self.processed_frame is not None else None
    
    def handle_click(self, x, y):
        """Handle mouse click event."""
        if self.detector:
            self.detector.handle_double_click(x, y)
    
    def handle_right_click(self, x, y):
        """Handle right-click event - create fixed-size selection."""
        if self.detector:
            self.detector.handle_right_click(x, y)
    
    def handle_double_click(self, x, y):
        """Handle double-click event."""
        if self.detector:
            self.detector.handle_double_click(x, y)
    
    def handle_mouse_down(self, x, y):
        """Handle mouse down event."""
        if self.detector:
            self.detector.handle_mouse_down(x, y)
    
    def handle_mouse_move(self, x, y):
        """Handle mouse move event."""
        if self.detector:
            self.detector.handle_mouse_move(x, y)
    
    def handle_mouse_up(self, x, y):
        """Handle mouse up event."""
        if self.detector:
            self.detector.handle_mouse_up(x, y)
    
    def _setup_entity_panel_callback(self):
        """Set up entity panel callback for AI analysis results."""
        try:
            if (self.detector and hasattr(self.detector, 'segmentation_engine') and 
                self.detector.segmentation_engine and 
                hasattr(self.detector.segmentation_engine, 'set_entity_panel_callback')):
                
                # Create callback function that updates entity panel
                def entity_callback(entity_id, text, status):
                    try:
                        if self.main_window and hasattr(self.main_window, 'entity_panel'):
                            self.main_window.entity_panel.update_entity(entity_id, text, status)
                        else:
                            debug_print(f"[ENTITY_CALLBACK] No entity panel available - Entity {entity_id}: {text[:50]}...")
                    except Exception as e:
                        debug_print(f"[ENTITY_CALLBACK] Error updating entity panel: {e}")
                
                # Set the callback
                self.detector.segmentation_engine.set_entity_panel_callback(entity_callback)
                
                # Set up status callback for main window
                if self.main_window and hasattr(self.main_window, 'update_segmentation_status'):
                    def status_callback(status_text, color):
                        try:
                            self.main_window.update_segmentation_status(status_text, color)
                        except Exception as e:
                            debug_print(f"[STATUS_CALLBACK] Error updating status: {e}")
                    
                    self.detector.segmentation_engine.set_status_callback(status_callback)
                    debug_print("[CV_CONTROLLER] Status callback set up successfully")
                
                debug_print("[CV_CONTROLLER] Entity panel callback set up successfully")
            else:
                debug_print("[CV_CONTROLLER] Could not set up entity panel callback - missing components")
        except Exception as e:
            debug_print(f"[CV_CONTROLLER] Error setting up entity panel callback: {e}")
    
    def cleanup(self):
        """Clean up all resources."""
        debug_print("[CV_CONTROLLER] Starting cleanup...")
        
        self.stop_detection()
        self.stop_video_capture()
        
        # Save session report
        try:
            if self.session_tracker:
                self.session_tracker.save_session_report()
            client = get_openai_client()
            if client:
                client.save_session_report()
        except Exception as e:
            debug_print(f"[CV_CONTROLLER] Error saving session report: {e}")
        
        debug_print("[CV_CONTROLLER] Cleanup complete")
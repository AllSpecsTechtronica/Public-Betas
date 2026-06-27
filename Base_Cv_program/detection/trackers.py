# Tracking Systems Module (Kalman and Highlight Trackers)
# ═══════════════════════════════════════════════════════════════════════════════

import cv2
import time
import threading
import numpy as np
from collections import deque
from queue import Queue

from core.config import DOUBLE_CLICK_BASE_WIDTH, DOUBLE_CLICK_BASE_HEIGHT
from utils.debug import debug_print, DebugLogger
from core.openai_bridge import get_openai_client
from utils.image_processing import encode_image_from_array

class KalmanBoxTracker:
    """A simple Kalman filter wrapper for smoothing bounding box predictions."""
    
    def __init__(self, bbox):
        # bbox is [x, y, w, h]
        self.kf = cv2.KalmanFilter(4, 4)
        self.kf.transitionMatrix = np.eye(4, dtype=np.float32)
        self.kf.measurementMatrix = np.eye(4, dtype=np.float32)
        # Process noise: how much we trust our model of the object's movement
        # Smaller values mean we think the object moves smoothly
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-3
        # Measurement noise: how much we trust the tracker's measurement
        # Smaller values mean we trust the raw tracker output more
        self.kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * 1e-1
        self.kf.errorCovPost = np.eye(4, dtype=np.float32)

        # Initialize state with the first bounding box measurement
        cx = bbox[0] + bbox[2] / 2
        cy = bbox[1] + bbox[3] / 2
        self.kf.statePost = np.array([[cx], [cy], [bbox[2]], [bbox[3]]], dtype=np.float32)

    def update(self, bbox):
        """Update the filter with a new measurement (from the tracker)."""
        try:
            # Validate bbox input
            if bbox is None or len(bbox) < 4:
                debug_print(f"[KALMAN] Invalid bbox: {bbox}")
                return None
                
            # Check for valid numeric values
            if not all(isinstance(x, (int, float)) and not np.isnan(x) and np.isfinite(x) for x in bbox):
                debug_print(f"[KALMAN] Invalid bbox values: {bbox}")
                return None
                
            # Check for positive dimensions
            if bbox[2] <= 0 or bbox[3] <= 0:
                debug_print(f"[KALMAN] Invalid bbox dimensions: {bbox}")
                return None
            
            # Measurement is [cx, cy, w, h]
            cx = float(bbox[0]) + float(bbox[2]) / 2
            cy = float(bbox[1]) + float(bbox[3]) / 2
            w = float(bbox[2])
            h = float(bbox[3])
            
            measurement = np.array([[cx], [cy], [w], [h]], dtype=np.float32)
            
            # Predict the next state
            predicted = self.kf.predict()
            
            # Correct the state with the new measurement
            corrected = self.kf.correct(measurement)
            
            # Validate corrected values
            if np.any(np.isnan(corrected)) or np.any(~np.isfinite(corrected)):
                debug_print(f"[KALMAN] Kalman filter produced invalid values")
                return None
            
            # Return the smoothed bbox as [x, y, w, h]
            corrected_w = max(1.0, float(corrected[2, 0]))  # Ensure positive width
            corrected_h = max(1.0, float(corrected[3, 0]))  # Ensure positive height
            corrected_x = float(corrected[0, 0]) - corrected_w / 2
            corrected_y = float(corrected[1, 0]) - corrected_h / 2
            
            return [corrected_x, corrected_y, corrected_w, corrected_h]
            
        except Exception as e:
            debug_print(f"[KALMAN] Error in Kalman filter update: {e}")
            return None


class AdvancedHighlightTracker:
    """Advanced segmentation engine with contour-based mask generation."""
    
    def __init__(self):
        self.debug_logger = DebugLogger()
        
        # UI colors
        self.ui_color = (57, 255, 186)
        self.text_color = (255, 255, 255)
        
        # Tracking state
        self.highlighted_regions = []
        self.highlight_timers = {}
        self.highlight_duration = 0.3
        self.tracked_objects = []
        self.tracked_regions = []
        
        # Advanced segmentation settings
        self.enable_edge_smoothing = True
        self.blur_kernel_size = (5, 5)
        self.canny_threshold1 = 50
        self.canny_threshold2 = 110
        self.morph_kernel = np.ones((3, 3), np.uint8)
        self.morph_iterations = 1
        self.prev_edges = None
        self.edge_smooth_factor = 0.95
        
        # Background edge computation
        self.background_edges = None
        self.edge_update_interval = 3  # Compute edges every 3 frames
        self._edge_update_counter = 0
        
        # Size constraints for stability
        self.min_object_size = (5, 5)
        self.min_contour_area = 20
        self.max_roi_dimension = 800
        self.max_roi_area = 400000
        
        # Visual settings
        self.contour_thickness = 1
        self.edge_alpha = 0.3
        self.fill_alpha = 0.1
        self.background_removal_in_bbox = False
        self.foreground_mask_alpha = 0.2
        
        # Double-click detection
        self.last_click_time = 0
        self.last_click_pos = None
        self.double_click_threshold = 0.5
        self.double_click_tolerance = 10
        self.double_click_rect_width = DOUBLE_CLICK_BASE_WIDTH
        self.double_click_rect_height = DOUBLE_CLICK_BASE_HEIGHT
        
        # Current frame and processing state
        self.current_frame = None
        self.temp_rectangle = None
        self.temp_rect_timer = None
        self.temp_rect_duration = 0.1
        
        # Advanced selection handling
        self.selection_start = None
        self.selection_end = None
        self.is_selecting = False
        self.slashing = False
        self.slash_start = None
        self.slash_end = None
        self.base_padding_factor = 0.1
        
        # Threading and memory management
        self.display_lock = threading.Lock()
        self.mask_pool = MaskMemoryPool(max_buffers=20)
        self.predictive_processor = PredictiveProcessor()
        
        # AI Analysis integration
        self.enable_ai_analysis = True  # Can be configured
        self.analysis_queue = Queue()
        self.entity_id_counter = 0
        self.analysis_thread = threading.Thread(target=self._analysis_worker, daemon=True)
        self.analysis_thread.start()
        
        # Entity panel callback (to be set by controller)
        self.entity_panel_callback = None
        
        debug_print("[ADVANCED_HIGHLIGHT_TRACKER] Initialized with segmentation engine and AI analysis")

    def update_frame(self, frame):
        """Update the current frame for processing and compute background edges."""
        with self.display_lock:
            self.current_frame = frame
            
            # Update background edges periodically
            if frame is not None:
                self._edge_update_counter += 1
                if self._edge_update_counter >= self.edge_update_interval:
                    self._edge_update_counter = 0
                    try:
                        self.background_edges = self.compute_background_edges(frame)
                        if self.predictive_processor:
                            self.predictive_processor.update_frame_data(frame, self.background_edges)
                    except Exception as e:
                        debug_print(f"[ADVANCED_TRACKER] Error computing background edges: {e}")
    
    def compute_background_edges(self, frame):
        """Compute background edges using Canny detection with morphological operations."""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, self.blur_kernel_size, 0)
            edges = cv2.Canny(blurred, self.canny_threshold1, self.canny_threshold2)
            
            # Morphological operations for better edge connectivity
            edges = cv2.dilate(edges, self.morph_kernel, iterations=1)
            edges = cv2.erode(edges, self.morph_kernel, iterations=1)
            edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, self.morph_kernel, iterations=self.morph_iterations)
            
            # Apply edge smoothing if enabled
            if self.enable_edge_smoothing and self.prev_edges is not None:
                edges = cv2.addWeighted(self.prev_edges, self.edge_smooth_factor, edges, 1 - self.edge_smooth_factor, 0)
            
            self.prev_edges = edges.copy()
            return edges
            
        except Exception as e:
            debug_print(f"[EDGE_DETECTION] Error computing edges: {e}")
            return None

    def handle_mouse_down(self, x, y):
        """Advanced mouse down handling with slashing support."""
        debug_print(f"[MOUSE_DOWN] Starting selection at ({x}, {y})")
        
        # Start selection and slashing modes
        self.is_selecting = True
        self.slashing = True
        self.selection_start = (x, y)
        self.selection_end = (x, y)
        self.slash_start = (x, y)
        self.slash_end = (x, y)

    def handle_mouse_move(self, x, y):
        """Enhanced mouse move handling with velocity tracking and predictive caching."""
        # Update selection if active
        if self.is_selecting:
            self.selection_end = (x, y)
        
        # Update slashing if active
        if self.slashing:
            self.slash_end = (x, y)
        
        # Update predictive processor with mouse movement
        if self.current_frame is not None and self.background_edges is not None:
            try:
                self.predictive_processor.on_mouse_move(x, y, self.current_frame, self.background_edges)
            except Exception as e:
                debug_print(f"[MOUSE_MOVE] Predictive processor error: {e}")

    def handle_mouse_up(self, x, y):
        """Enhanced mouse up handling with drag detection and processing."""
        was_selecting = self.is_selecting
        was_slashing = self.slashing
        
        # End selection and slashing modes
        self.is_selecting = False
        self.slashing = False
        
        if was_slashing and self.slash_start and self.slash_end:
            # Process drag selection
            if (abs(self.slash_end[0] - self.slash_start[0]) > 5 or 
                abs(self.slash_end[1] - self.slash_start[1]) > 5):
                
                debug_print("[MOUSE_UP] Processing drag selection")
                p1, p2 = self.get_box_from_slash(self.slash_start, self.slash_end)
                
                # Set temp rectangle for visual feedback
                self.temp_rectangle = (p1, p2)
                self.temp_rect_timer = time.time()
                
                # Process the selection
                self.process_selection(p1, p2)
            else:
                debug_print("[MOUSE_UP] Drag too small - ignored")
        
        elif was_selecting and self.selection_start and self.selection_end:
            # Process standard selection
            start_x, start_y = self.selection_start
            end_x, end_y = self.selection_end
            
            # Calculate selection rectangle
            x1, y1 = min(start_x, end_x), min(start_y, end_y)
            x2, y2 = max(start_x, end_x), max(start_y, end_y)
            
            # Only process if selection is significant
            if abs(x2 - x1) > 10 and abs(y2 - y1) > 10:
                debug_print(f"[MOUSE_UP] Processing standard selection: ({x1},{y1}) to ({x2},{y2})")
                self.process_selection((x1, y1), (x2, y2))
        
        # Reset selection state
        self.selection_start = None
        self.selection_end = None
        self.slash_start = None
        self.slash_end = None

    def handle_double_click(self, x, y):
        """Enhanced double-click handling with adaptive rectangle sizing."""
        debug_print(f"[DOUBLE_CLICK] Detected at ({x}, {y}) - processing adaptive selection")
        
        # Adaptive rectangle sizing based on current frame content
        adaptive_width = self.double_click_rect_width
        adaptive_height = self.double_click_rect_height
        
        # Analyze local area to determine optimal selection size
        if self.current_frame is not None:
            try:
                # Sample area around click point for content analysis
                sample_size = 50
                x1_sample = max(0, x - sample_size)
                y1_sample = max(0, y - sample_size)
                x2_sample = min(self.current_frame.shape[1], x + sample_size)
                y2_sample = min(self.current_frame.shape[0], y + sample_size)
                
                sample_region = self.current_frame[y1_sample:y2_sample, x1_sample:x2_sample]
                
                # Analyze content complexity to adjust selection size
                if sample_region.size > 0:
                    gray_sample = cv2.cvtColor(sample_region, cv2.COLOR_BGR2GRAY)
                    edges_sample = cv2.Canny(gray_sample, self.canny_threshold1, self.canny_threshold2)
                    edge_density = np.sum(edges_sample > 0) / edges_sample.size
                    
                    # Adjust rectangle size based on edge density
                    if edge_density > 0.1:  # High detail area
                        scale_factor = 0.7  # Smaller selection for detailed areas
                    elif edge_density < 0.02:  # Low detail area  
                        scale_factor = 1.3  # Larger selection for sparse areas
                    else:
                        scale_factor = 1.0  # Normal size
                    
                    adaptive_width = int(self.double_click_rect_width * scale_factor)
                    adaptive_height = int(self.double_click_rect_height * scale_factor)
                    
                    debug_print(f"[DOUBLE_CLICK] Adaptive sizing: edge_density={edge_density:.3f}, scale={scale_factor:.2f}")
                    
            except Exception as e:
                debug_print(f"[DOUBLE_CLICK] Adaptive sizing error: {e}")
        
        # Create selection rectangle with adaptive sizing
        half_width = adaptive_width // 2
        half_height = adaptive_height // 2
        
        # Ensure bounds
        if self.current_frame is not None:
            x1 = max(0, x - half_width)
            y1 = max(0, y - half_height)
            x2 = min(self.current_frame.shape[1], x + half_width)
            y2 = min(self.current_frame.shape[0], y + half_height)
        else:
            x1 = max(0, x - half_width)
            y1 = max(0, y - half_height)
            x2 = x + half_width
            y2 = y + half_height
        
        # Set temp rectangle for immediate visual feedback
        self.temp_rectangle = ((x1, y1), (x2, y2))
        self.temp_rect_timer = time.time()
        
        debug_print(f"[DOUBLE_CLICK] Processing adaptive selection: ({x1},{y1}) to ({x2},{y2})")
        self.process_selection((x1, y1), (x2, y2))
    
    def detect_double_click(self, x, y):
        """Advanced double-click detection with configurable tolerance."""
        current_time = time.time()
        
        # Check if this could be a double-click
        if (self.last_click_pos is not None and self.last_click_time > 0):
            time_diff = current_time - self.last_click_time
            
            # Calculate distance from last click
            dx = x - self.last_click_pos[0]
            dy = y - self.last_click_pos[1]
            distance = (dx * dx + dy * dy) ** 0.5
            
            # Check if within double-click thresholds
            if (time_diff <= self.double_click_threshold and 
                distance <= self.double_click_tolerance):
                
                # Double-click detected!
                self.handle_double_click(x, y)
                
                # Reset to prevent triple-click
                self.last_click_pos = None
                self.last_click_time = 0
                
                return True
        
        # Update last click info
        self.last_click_pos = (x, y)
        self.last_click_time = current_time
        
        return False

    def process_selection(self, start_point, end_point):
        """Advanced segmentation processing with contour-based mask generation."""
        debug_print(f"[SEGMENTATION] Starting advanced processing: {start_point} to {end_point}")
        
        # Thread-safe frame access
        with self.display_lock:
            current_frame_copy = self.current_frame.copy() if self.current_frame is not None else None
            background_edges_copy = self.background_edges.copy() if self.background_edges is not None else None
        
        if current_frame_copy is None:
            debug_print("[SEGMENTATION] ERROR: No current frame available")
            return
        
        # Validate and adjust coordinates
        try:
            x1, y1 = max(0, min(start_point[0], end_point[0])), max(0, min(start_point[1], end_point[1]))
            x2, y2 = min(current_frame_copy.shape[1], max(start_point[0], end_point[0])), min(current_frame_copy.shape[0], max(start_point[1], end_point[1]))
            
            # Calculate ROI dimensions
            roi_width, roi_height = x2 - x1, y2 - y1
            roi_area = roi_width * roi_height
            
            # Validate minimum size
            if roi_width < self.min_object_size[0] or roi_height < self.min_object_size[1]:
                debug_print(f"[SEGMENTATION] Region too small: {roi_width}x{roi_height}")
                return
            
            # Apply maximum size limits for stability
            if roi_width > self.max_roi_dimension or roi_height > self.max_roi_dimension or roi_area > self.max_roi_area:
                debug_print(f"[SEGMENTATION] Region too large, scaling down: {roi_width}x{roi_height}")
                
                # Scale down while maintaining aspect ratio
                scale_factor = min(self.max_roi_dimension / roi_width, self.max_roi_dimension / roi_height)
                if roi_area > self.max_roi_area:
                    area_scale = (self.max_roi_area / roi_area) ** 0.5
                    scale_factor = min(scale_factor, area_scale)
                
                # Calculate new dimensions
                new_width = int(roi_width * scale_factor)
                new_height = int(roi_height * scale_factor)
                
                # Center the reduced ROI
                center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                x1 = max(0, center_x - new_width // 2)
                y1 = max(0, center_y - new_height // 2)
                x2 = min(current_frame_copy.shape[1], x1 + new_width)
                y2 = min(current_frame_copy.shape[0], y1 + new_height)
                
                debug_print(f"[SEGMENTATION] Scaled to: {x2-x1}x{y2-y1}")
        
        except Exception as e:
            debug_print(f"[SEGMENTATION] Coordinate validation error: {e}")
            return
        
        # Get or compute edges
        if background_edges_copy is None:
            debug_print("[SEGMENTATION] Computing edges on-demand")
            try:
                frame_gray = cv2.cvtColor(current_frame_copy, cv2.COLOR_BGR2GRAY)
                background_edges_copy = cv2.Canny(frame_gray, self.canny_threshold1, self.canny_threshold2)
            except Exception as e:
                debug_print(f"[SEGMENTATION] Emergency edge computation failed: {e}")
                return
        
        # Extract edge ROI
        try:
            edge_roi = background_edges_copy[y1:y2, x1:x2]
            if edge_roi.size == 0:
                debug_print("[SEGMENTATION] Empty edge ROI")
                return
        except Exception as e:
            debug_print(f"[SEGMENTATION] Edge ROI extraction error: {e}")
            return
        
        # Find contours
        try:
            contours, _ = cv2.findContours(edge_roi.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = [cnt for cnt in contours if cv2.contourArea(cnt) > self.min_contour_area]
            debug_print(f"[SEGMENTATION] Found {len(contours)} valid contours")
            
            if contours:
                # Create masks from contours
                bbox = (x1, y1, x2, y2)
                edge_mask, filled_mask = self.create_masks_from_contours(contours, bbox)
                
                if edge_mask is not None and filled_mask is not None:
                    # Store segmentation result
                    with self.display_lock:
                        segmentation_info = {
                            'bbox': bbox,
                            'contours': contours,
                            'edge_mask': edge_mask,
                            'filled_mask': filled_mask,
                            'timestamp': time.time(),
                            'type': 'segmentation'
                        }
                        self.highlighted_regions.append(segmentation_info)
                        debug_print("[SEGMENTATION] Segmentation completed successfully")
                    
                    # Trigger AI analysis if enabled
                    client = get_openai_client()
                    if self.enable_ai_analysis and client:
                        self._queue_ai_analysis(current_frame_copy, bbox)
                    else:
                        debug_print("[AI_ANALYSIS] Skipping AI analysis - disabled or client not available")
                        
        except Exception as e:
            debug_print(f"[SEGMENTATION] Contour processing error: {e}")
        
        # Clean up old highlights
        self._cleanup_old_highlights()
    
    def create_masks_from_contours(self, contours, bbox):
        """Create edge and filled masks from contours with thread-safe buffer management."""
        try:
            x1, y1, x2, y2 = bbox
            height, width = y2 - y1, x2 - x1
            
            # Validate dimensions
            if height <= 0 or width <= 0:
                debug_print(f"[MASK_CREATION] Invalid dimensions: {width}x{height}")
                return None, None
            
            # Get frame shape safely
            with self.display_lock:
                if self.current_frame is None:
                    debug_print("[MASK_CREATION] No current frame available")
                    return None, None
                frame_height, frame_width = self.current_frame.shape[:2]
            
            # Allocate buffers
            try:
                edge_mask = self.mask_pool.get_buffer(height, width)
                filled_mask = self.mask_pool.get_buffer(height, width)
                full_edge_mask = self.mask_pool.get_buffer(frame_height, frame_width)
                full_filled_mask = self.mask_pool.get_buffer(frame_height, frame_width)
            except Exception as e:
                debug_print(f"[MASK_CREATION] Buffer allocation failed: {e}")
                return None, None
            
            try:
                # Draw contours on local masks
                for cnt in contours:
                    if cnt is not None and len(cnt) > 0:
                        cv2.drawContours(edge_mask, [cnt], -1, 255, self.contour_thickness)
                        cv2.drawContours(filled_mask, [cnt], -1, 255, -1)
                
                # Copy to full-frame masks with bounds checking
                y1_clipped = max(0, min(y1, frame_height))
                y2_clipped = max(0, min(y2, frame_height))
                x1_clipped = max(0, min(x1, frame_width))
                x2_clipped = max(0, min(x2, frame_width))
                
                actual_height = y2_clipped - y1_clipped
                actual_width = x2_clipped - x1_clipped
                
                if actual_height > 0 and actual_width > 0:
                    # Resize masks if needed
                    if edge_mask.shape != (actual_height, actual_width):
                        edge_mask = cv2.resize(edge_mask, (actual_width, actual_height), interpolation=cv2.INTER_NEAREST)
                        filled_mask = cv2.resize(filled_mask, (actual_width, actual_height), interpolation=cv2.INTER_NEAREST)
                    
                    full_edge_mask[y1_clipped:y2_clipped, x1_clipped:x2_clipped] = edge_mask
                    full_filled_mask[y1_clipped:y2_clipped, x1_clipped:x2_clipped] = filled_mask
                
                # Return local buffers to pool
                self.mask_pool.return_buffer(edge_mask)
                self.mask_pool.return_buffer(filled_mask)
                
                return full_edge_mask, full_filled_mask
                
            except Exception as e:
                # Clean up on error
                self.mask_pool.return_buffer(edge_mask)
                self.mask_pool.return_buffer(filled_mask)
                self.mask_pool.return_buffer(full_edge_mask)
                self.mask_pool.return_buffer(full_filled_mask)
                debug_print(f"[MASK_CREATION] Error drawing contours: {e}")
                return None, None
                
        except Exception as e:
            debug_print(f"[MASK_CREATION] Failed to create masks: {e}")
            return None, None
    
    def _cleanup_old_highlights(self):
        """Remove old highlight regions to prevent memory buildup."""
        try:
            current_time = time.time()
            with self.display_lock:
                old_count = len(self.highlighted_regions)
                self.highlighted_regions = [
                    r for r in self.highlighted_regions
                    if current_time - r['timestamp'] < self.highlight_duration * 10
                ]
                
                # Return masks to pool for cleaned up regions
                for region in [r for r in self.highlighted_regions if current_time - r['timestamp'] >= self.highlight_duration * 10]:
                    if 'edge_mask' in region and region['edge_mask'] is not None:
                        self.mask_pool.return_buffer(region['edge_mask'])
                    if 'filled_mask' in region and region['filled_mask'] is not None:
                        self.mask_pool.return_buffer(region['filled_mask'])
                
                if len(self.highlighted_regions) < old_count:
                    debug_print(f"[CLEANUP] Removed {old_count - len(self.highlighted_regions)} old highlights")
                    
        except Exception as e:
            debug_print(f"[CLEANUP] Error cleaning highlights: {e}")

    def draw_overlays(self, frame):
        """Draw all overlays on the frame with advanced segmentation visualization."""
        if frame is None:
            return frame
            
        output = frame.copy()
        current_time = time.time()
        
        with self.display_lock:
            # Draw current selection if active (real-time rectangle drawing)
            if self.is_selecting and self.selection_start and self.selection_end:
                x1, y1 = self.selection_start
                x2, y2 = self.selection_end
                
                # Draw animated selection rectangle
                cv2.rectangle(output, (x1, y1), (x2, y2), self.ui_color, 2)
                
                # Draw semi-transparent overlay with animation
                overlay = output.copy()
                cv2.rectangle(overlay, (x1, y1), (x2, y2), self.ui_color, -1)
                cv2.addWeighted(overlay, 0.2, output, 0.8, 0, output)
                
                # Draw corner markers for active selection
                self._draw_corner_markers(output, (x1, y1, x2, y2), self.ui_color, current_time)
            
            # Draw slashing indicator (drag selection)
            if self.slashing and self.slash_start and self.slash_end:
                cv2.line(output, self.slash_start, self.slash_end, self.ui_color, 3)
                
                # Draw predicted bounding box
                if abs(self.slash_end[0] - self.slash_start[0]) > 5 or abs(self.slash_end[1] - self.slash_start[1]) > 5:
                    p1, p2 = self.get_box_from_slash(self.slash_start, self.slash_end)
                    cv2.rectangle(output, p1, p2, self.ui_color, 2, cv2.LINE_AA)
            
            # Draw segmentation results with advanced visualization
            for region in self.highlighted_regions:
                age = current_time - region['timestamp']
                if age > self.highlight_duration * 10:
                    continue
                
                region_type = region.get('type', 'selection')
                
                if region_type == 'segmentation' and 'edge_mask' in region and 'filled_mask' in region:
                    # Advanced segmentation visualization
                    self._draw_segmentation_overlay(output, region, age, current_time)
                else:
                    # Simple selection visualization
                    self._draw_simple_selection(output, region, age)
            
            # Draw temporary rectangle (from double-click or other sources)
            if (self.temp_rectangle is not None and self.temp_rect_timer is not None and 
                current_time - self.temp_rect_timer < self.temp_rect_duration):
                p1, p2 = self.temp_rectangle
                cv2.rectangle(output, p1, p2, (255, 255, 0), 2)  # Yellow for temp
        
        return output
    
    def _draw_segmentation_overlay(self, output, region, age, current_time):
        """Draw advanced segmentation overlay with masks and contours."""
        try:
            fade_factor = max(0, 1 - (age / (self.highlight_duration * 10)))
            alpha = self.edge_alpha * fade_factor
            fill_alpha = self.fill_alpha * fade_factor
            
            edge_mask = region.get('edge_mask')
            filled_mask = region.get('filled_mask')
            
            if edge_mask is not None and filled_mask is not None:
                # Background removal effect (if enabled)
                if self.background_removal_in_bbox:
                    x1, y1, x2, y2 = region['bbox']
                    
                    # Create foreground mask overlay
                    foreground_overlay = output.copy()
                    
                    # Apply filled mask as alpha blend
                    mask_3channel = cv2.cvtColor(filled_mask, cv2.COLOR_GRAY2BGR)
                    mask_normalized = mask_3channel.astype(np.float32) / 255.0
                    
                    # Highlight foreground, dim background
                    background_dimmed = (output.astype(np.float32) * (1 - mask_normalized * self.foreground_mask_alpha)).astype(np.uint8)
                    foreground_highlighted = (output.astype(np.float32) * (1 + mask_normalized * 0.1)).astype(np.uint8)
                    
                    # Combine
                    output[:] = np.where(mask_3channel > 0, foreground_highlighted, background_dimmed)
                
                # Draw edge contours
                if alpha > 0:
                    edge_overlay = np.zeros_like(output)
                    edge_overlay[edge_mask > 0] = self.ui_color
                    cv2.addWeighted(edge_overlay, alpha, output, 1, 0, output)
                
                # Draw filled regions with transparency
                if fill_alpha > 0:
                    filled_overlay = np.zeros_like(output)
                    filled_overlay[filled_mask > 0] = self.ui_color
                    cv2.addWeighted(filled_overlay, fill_alpha, output, 1, 0, output)
                
                # Draw bounding box
                x1, y1, x2, y2 = region['bbox']
                box_color = tuple(int(c * fade_factor) for c in self.ui_color)
                cv2.rectangle(output, (x1, y1), (x2, y2), box_color, 2)
                
                # Animated corner markers
                self._draw_corner_markers(output, region['bbox'], box_color, current_time)
        
        except Exception as e:
            debug_print(f"[OVERLAY] Error drawing segmentation: {e}")
    
    def _draw_simple_selection(self, output, region, age):
        """Draw simple selection overlay."""
        fade_factor = max(0, 1 - (age / (self.highlight_duration * 5)))
        alpha = self.edge_alpha * fade_factor
        
        x1, y1, x2, y2 = region['bbox']
        
        # Draw border
        color = tuple(int(c * fade_factor) for c in self.ui_color)
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        
        # Draw semi-transparent fill
        if alpha > 0:
            overlay = output.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), self.ui_color, -1)
            cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)
    
    def _draw_corner_markers(self, output, bbox, color, current_time):
        """Draw animated corner markers for selection boxes."""
        try:
            x1, y1, x2, y2 = bbox
            corner_size = 12
            thickness = 2
            
            # Animate corner markers
            animation_phase = (current_time * 3) % (2 * np.pi)  # 3 Hz animation
            pulse = 0.5 + 0.5 * np.sin(animation_phase)
            animated_color = tuple(int(c * (0.7 + 0.3 * pulse)) for c in color)
            
            # Top-left corner
            cv2.line(output, (x1, y1 + corner_size), (x1 + corner_size, y1 + corner_size), animated_color, thickness, cv2.LINE_AA)
            cv2.line(output, (x1 + corner_size, y1 + corner_size), (x1 + corner_size, y1), animated_color, thickness, cv2.LINE_AA)
            
            # Top-right corner
            cv2.line(output, (x2 - corner_size, y1), (x2 - corner_size, y1 + corner_size), animated_color, thickness, cv2.LINE_AA)
            cv2.line(output, (x2 - corner_size, y1 + corner_size), (x2, y1 + corner_size), animated_color, thickness, cv2.LINE_AA)
            
            # Bottom-left corner
            cv2.line(output, (x1, y2 - corner_size), (x1 + corner_size, y2 - corner_size), animated_color, thickness, cv2.LINE_AA)
            cv2.line(output, (x1 + corner_size, y2 - corner_size), (x1 + corner_size, y2), animated_color, thickness, cv2.LINE_AA)
            
            # Bottom-right corner
            cv2.line(output, (x2 - corner_size, y2), (x2 - corner_size, y2 - corner_size), animated_color, thickness, cv2.LINE_AA)
            cv2.line(output, (x2 - corner_size, y2 - corner_size), (x2, y2 - corner_size), animated_color, thickness, cv2.LINE_AA)
            
        except Exception as e:
            debug_print(f"[OVERLAY] Error drawing corner markers: {e}")
    
    def get_box_from_slash(self, start, end):
        """Convert slash/drag coordinates to bounding box with padding."""
        try:
            dx, dy = abs(end[0] - start[0]), abs(end[1] - start[1])
            base_padding = int(max(min(dx if dx > dy else dy, 40), 10) * self.base_padding_factor)
            padding_x = max(base_padding * 0.2, 5) if dx > dy else max(base_padding, 5)
            padding_y = max(base_padding, 5) if dx > dy else max(base_padding * 0.2, 5)
            
            x1, y1 = min(start[0], end[0]) - padding_x, min(start[1], end[1]) - padding_y
            x2, y2 = max(start[0], end[0]) + padding_x, max(start[1], end[1]) + padding_y
            
            return (int(x1), int(y1)), (int(x2), int(y2))
        except Exception as e:
            debug_print(f"[GEOMETRY] Error computing box from slash: {e}")
            return start, end
    
    def _queue_ai_analysis(self, frame, bbox):
        """Queue AI analysis for the selected region."""
        try:
            # Generate unique entity ID
            self.entity_id_counter += 1
            entity_id = self.entity_id_counter
            
            # Extract ROI from frame
            x1, y1, x2, y2 = bbox
            roi_image = frame[y1:y2, x1:x2]
            
            # Create context window (larger area around selection)
            context_image = self._create_context_window(frame, bbox)
            
            # Queue analysis task
            analysis_task = {
                'entity_id': entity_id,
                'roi_image': roi_image,
                'context_image': context_image,
                'bbox': bbox,
                'timestamp': time.time()
            }
            
            self.analysis_queue.put(analysis_task)
            debug_print(f"[AI_ANALYSIS] Queued analysis for entity {entity_id} at bbox {bbox}")
            
            # Notify entity panel about new analysis starting
            if self.entity_panel_callback:
                self.entity_panel_callback(entity_id, "Analysis starting...", 'streaming')
        
        except Exception as e:
            debug_print(f"[AI_ANALYSIS] Error queuing analysis: {e}")
    
    def _create_context_window(self, frame, bbox):
        """Create context window around selection for better AI analysis."""
        try:
            x1, y1, x2, y2 = bbox
            h, w = frame.shape[:2]
            
            # Calculate context window with padding (50% larger than selection)
            padding_factor = 0.5
            roi_width, roi_height = x2 - x1, y2 - y1
            pad_x = int(roi_width * padding_factor / 2)
            pad_y = int(roi_height * padding_factor / 2)
            
            # Ensure context window stays within frame bounds
            ctx_x1 = max(0, x1 - pad_x)
            ctx_y1 = max(0, y1 - pad_y)
            ctx_x2 = min(w, x2 + pad_x)
            ctx_y2 = min(h, y2 + pad_y)
            
            context_image = frame[ctx_y1:ctx_y2, ctx_x1:ctx_x2]
            debug_print(f"[AI_ANALYSIS] Created context window: {ctx_x1},{ctx_y1} to {ctx_x2},{ctx_y2}")
            
            return context_image
            
        except Exception as e:
            debug_print(f"[AI_ANALYSIS] Error creating context window: {e}")
            return frame[y1:y2, x1:x2]  # Fall back to just the ROI
    
    def _analysis_worker(self):
        """Background worker thread for processing AI analysis requests."""
        debug_print("[AI_ANALYSIS] Analysis worker thread started")
        
        while True:
            try:
                # Get next analysis task
                task = self.analysis_queue.get()
                if task is None:  # Shutdown signal
                    break
                
                entity_id = task['entity_id']
                roi_image = task['roi_image']
                context_image = task['context_image']
                bbox = task['bbox']
                
                debug_print(f"[AI_ANALYSIS] Processing analysis for entity {entity_id}")
                
                # Combine ROI and context for analysis
                combined_image = self._combine_roi_and_context(roi_image, context_image)
                
                # Encode image for OpenAI
                base64_image = encode_image_from_array(combined_image)
                
                if base64_image:
                    # Request AI analysis
                    client = get_openai_client()
                    analysis_text = (
                        client.analyze_image(
                            combined_image,
                            prompt="Analyze what you see in this image. Describe the objects, their appearance, and any notable details.",
                        )
                        if client
                        else None
                    )
                    
                    if analysis_text:
                        debug_print(f"[AI_ANALYSIS] Completed analysis for entity {entity_id}")
                        
                        # Update entity panel with results
                        if self.entity_panel_callback:
                            self.entity_panel_callback(entity_id, analysis_text, 'done')
                    else:
                        debug_print(f"[AI_ANALYSIS] No analysis text returned for entity {entity_id}")
                        if self.entity_panel_callback:
                            self.entity_panel_callback(entity_id, "Analysis failed - no response from AI", 'error')
                else:
                    debug_print(f"[AI_ANALYSIS] Failed to encode image for entity {entity_id}")
                    if self.entity_panel_callback:
                        self.entity_panel_callback(entity_id, "Analysis failed - image encoding error", 'error')
                
                # Mark task as done
                self.analysis_queue.task_done()
                
            except Exception as e:
                debug_print(f"[AI_ANALYSIS] Worker error: {e}")
                if self.entity_panel_callback and 'entity_id' in locals():
                    self.entity_panel_callback(entity_id, f"Analysis failed: {e}", 'error')
    
    def _combine_roi_and_context(self, roi_image, context_image):
        """Combine ROI and context images for analysis."""
        try:
            # For now, just use the context image as it includes the ROI
            # In the future, we could create a composite image highlighting the ROI
            return context_image
        except Exception as e:
            debug_print(f"[AI_ANALYSIS] Error combining images: {e}")
            return roi_image
    
    def set_entity_panel_callback(self, callback):
        """Set callback function to update entity panel with analysis results."""
        self.entity_panel_callback = callback
        debug_print("[AI_ANALYSIS] Entity panel callback set")

    def cleanup(self):
        """Clean up resources when the tracker is stopped."""
        # Stop analysis worker
        self.analysis_queue.put(None)  # Shutdown signal
        
        with self.display_lock:
            self.highlighted_regions.clear()
            self.tracked_objects.clear()
            self.tracked_regions.clear()
        
        debug_print("[HIGHLIGHT_TRACKER] Cleaned up resources")


class MaskMemoryPool:
    """Simple memory pool for mask buffers to reduce allocations."""
    
    def __init__(self, max_buffers=20):
        self.max_buffers = max_buffers
        self.available_buffers = []
        self.lock = threading.Lock()
        
    def get_buffer(self, height, width, dtype=np.uint8):
        """Get a buffer from the pool or create a new one."""
        with self.lock:
            # Try to find a suitable buffer
            for i, (buffer, buf_height, buf_width, buf_dtype) in enumerate(self.available_buffers):
                if buf_height == height and buf_width == width and buf_dtype == dtype:
                    return self.available_buffers.pop(i)[0]
            
            # Create new buffer if none available
            return np.zeros((height, width), dtype=dtype)
    
    def return_buffer(self, buffer):
        """Return a buffer to the pool."""
        if buffer is None:
            return
            
        with self.lock:
            if len(self.available_buffers) < self.max_buffers:
                height, width = buffer.shape[:2]
                dtype = buffer.dtype
                buffer.fill(0)  # Clear the buffer
                self.available_buffers.append((buffer, height, width, dtype))


class PredictiveProcessor:
    """Enhanced predictive processor with spatial caching."""
    
    def __init__(self):
        self.prediction_history = {}
        self.max_history = 10
        
        # Enhanced features for segmentation
        self.cached_edges = {}
        self.mouse_velocity = (0, 0)
        self.last_mouse_pos = None
        self.last_mouse_time = 0
        
    def predict_next_position(self, track_id, current_bbox):
        """Predict the next position of a tracked object."""
        if track_id not in self.prediction_history:
            self.prediction_history[track_id] = []
        
        history = self.prediction_history[track_id]
        history.append(current_bbox)
        
        # Keep only recent history
        if len(history) > self.max_history:
            history.pop(0)
        
        # Simple linear prediction based on recent movement
        if len(history) >= 2:
            last_bbox = history[-1]
            prev_bbox = history[-2]
            
            # Calculate velocity
            dx = last_bbox[0] - prev_bbox[0]
            dy = last_bbox[1] - prev_bbox[1]
            
            # Predict next position
            predicted_x = last_bbox[0] + dx
            predicted_y = last_bbox[1] + dy
            
            return [predicted_x, predicted_y, last_bbox[2], last_bbox[3]]
        
        return current_bbox
    
    def update_frame_data(self, frame, edges):
        """Update frame and edge data for predictive processing."""
        # Store frame data for future predictions
        if frame is not None and edges is not None:
            # Cache edge data around likely interaction areas
            self.cache_edge_regions(edges)
    
    def cache_edge_regions(self, edges):
        """Cache edge data for frequently accessed regions."""
        try:
            # Simple grid-based caching
            h, w = edges.shape
            grid_size = 200
            
            for y in range(0, h, grid_size):
                for x in range(0, w, grid_size):
                    x2 = min(x + grid_size, w)
                    y2 = min(y + grid_size, h)
                    
                    cache_key = f"{x}_{y}_{x2}_{y2}"
                    self.cached_edges[cache_key] = {
                        'area': (x, y, x2, y2),
                        'edges': edges[y:y2, x:x2].copy(),
                        'timestamp': time.time()
                    }
            
            # Clean old cached data
            current_time = time.time()
            self.cached_edges = {
                k: v for k, v in self.cached_edges.items()
                if current_time - v['timestamp'] < 5.0  # Keep cache for 5 seconds
            }
            
        except Exception as e:
            debug_print(f"[PREDICTIVE] Error caching edges: {e}")
    
    def get_cached_edges(self, x, y):
        """Get cached edge data for a position."""
        try:
            for cache_key, cache_data in self.cached_edges.items():
                x1, y1, x2, y2 = cache_data['area']
                if x1 <= x <= x2 and y1 <= y <= y2:
                    return cache_data
        except Exception:
            pass
        return None
    
    def on_mouse_move(self, x, y, frame, edges):
        """Handle mouse movement for predictive processing."""
        try:
            current_time = time.time()
            
            if self.last_mouse_pos is not None and self.last_mouse_time > 0:
                dt = current_time - self.last_mouse_time
                if dt > 0:
                    dx = x - self.last_mouse_pos[0]
                    dy = y - self.last_mouse_pos[1]
                    self.mouse_velocity = (dx / dt, dy / dt)
            
            self.last_mouse_pos = (x, y)
            self.last_mouse_time = current_time
            
            # Predictive edge caching based on mouse movement
            if edges is not None:
                # Cache region around current mouse position
                cache_size = 100
                x1 = max(0, x - cache_size)
                y1 = max(0, y - cache_size)
                x2 = min(edges.shape[1], x + cache_size)
                y2 = min(edges.shape[0], y + cache_size)
                
                cache_key = f"mouse_{x1}_{y1}_{x2}_{y2}"
                self.cached_edges[cache_key] = {
                    'area': (x1, y1, x2, y2),
                    'edges': edges[y1:y2, x1:x2].copy(),
                    'timestamp': current_time
                }
                
        except Exception as e:
            debug_print(f"[PREDICTIVE] Mouse move error: {e}")
    
    def cleanup_track(self, track_id):
        """Clean up prediction history for a track."""
        if track_id in self.prediction_history:
            del self.prediction_history[track_id]
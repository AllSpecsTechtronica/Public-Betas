# Segmentation Engine for Modular CV System
# ═══════════════════════════════════════════════════════════════════════════════

import cv2
import numpy as np
import time
import threading
from queue import Queue
from utils.debug import debug_print

class SegmentationEngine:
    """Advanced segmentation engine with interactive drawing and AI analysis."""
    
    def __init__(self):
        self.current_frame = None
        self.background_edges = None
        self.display_lock = threading.Lock()
        
        # Drawing state
        self.slashing = False
        self.slash_start = None
        self.slash_end = None
        self.temp_rectangle = None
        self.temp_rect_timer = None
        
        # Selection parameters
        self.min_object_size = (50, 50)
        self.double_click_rect_width = 420
        self.double_click_rect_height = 420
        
        # Analysis queue
        self.analysis_queue = Queue()
        self.analysis_thread = None
        self.analysis_running = False
        
        # Entity panel callback
        self.entity_panel_callback = None
        
        # Status callback for main window
        self.status_callback = None
        
        # Start analysis thread
        self.start_analysis_thread()
        
        debug_print("[SEGMENTATION_ENGINE] Initialized")
    
    def start_analysis_thread(self):
        """Start the background analysis thread."""
        self.analysis_running = True
        self.analysis_thread = threading.Thread(target=self._analysis_worker, daemon=True)
        self.analysis_thread.start()
        debug_print("[SEGMENTATION_ENGINE] Analysis thread started")
    
    def stop_analysis_thread(self):
        """Stop the background analysis thread."""
        self.analysis_running = False
        if self.analysis_thread:
            self.analysis_thread.join(timeout=2.0)
        debug_print("[SEGMENTATION_ENGINE] Analysis thread stopped")
    
    def set_entity_panel_callback(self, callback):
        """Set callback for updating entity panel."""
        self.entity_panel_callback = callback
        debug_print("[SEGMENTATION_ENGINE] Entity panel callback set")
    
    def set_status_callback(self, callback):
        """Set callback for updating status."""
        self.status_callback = callback
        debug_print("[SEGMENTATION_ENGINE] Status callback set")
    
    def update_frame(self, frame):
        """Update the current frame and generate background edges."""
        with self.display_lock:
            self.current_frame = frame.copy()
            self.background_edges = self._generate_background_edges(frame)
    
    def _generate_background_edges(self, frame):
        """Generate background edge detection for the frame."""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Detect edges using Canny
            edges = cv2.Canny(blurred, 50, 150)
            
            # Dilate edges slightly to connect broken lines
            kernel = np.ones((2, 2), np.uint8)
            dilated = cv2.dilate(edges, kernel, iterations=1)
            
            return dilated
            
        except Exception as e:
            debug_print(f"[SEGMENTATION_ENGINE] Error generating background edges: {e}")
            return None
    
    def handle_mouse_down(self, x, y):
        """Handle mouse down event - start selection."""
        debug_print(f"[SEGMENTATION_ENGINE] Mouse down at ({x}, {y})")
        self.slashing = True
        self.slash_start = (x, y)
        self.slash_end = (x, y)
        
        # Update status
        if self.status_callback:
            self.status_callback("Drawing...", "#ff6b35")
    
    def handle_mouse_move(self, x, y):
        """Handle mouse move event - update selection."""
        if self.slashing:
            self.slash_end = (x, y)
    
    def handle_mouse_up(self, x, y):
        """Handle mouse up event - process selection."""
        if self.slashing:
            self.slashing = False
            if (self.slash_start and self.slash_end and 
                (abs(self.slash_end[0] - self.slash_start[0]) > 5 or 
                 abs(self.slash_end[1] - self.slash_start[1]) > 5)):
                
                p1, p2 = self._get_box_from_slash(self.slash_start, self.slash_end)
                self.temp_rectangle = (p1, p2)
                self.temp_rect_timer = time.time()
                
                debug_print(f"[SEGMENTATION_ENGINE] Processing selection from {self.slash_start} to {self.slash_end}")
                debug_print(f"[SEGMENTATION_ENGINE] Converted to box: {p1} to {p2}")
                
                self._process_selection(p1, p2)
            else:
                debug_print("[SEGMENTATION_ENGINE] Mouse drag too small - selection ignored")
            
            self.slash_start, self.slash_end = None, None
            
            # Update status
            if self.status_callback:
                self.status_callback("Processing...", "#39ff7a")
    
    def handle_double_click(self, x, y):
        """Handle double-click event - create fixed-size selection."""
        if not self.current_frame or self.background_edges is None:
            debug_print("[SEGMENTATION_ENGINE] Double-click ignored - no frame or edges available")
            return
        
        frame_h, frame_w = self.current_frame.shape[:2]
        x1, y1 = max(0, x - self.double_click_rect_width // 2), max(0, y - self.double_click_rect_height // 2)
        x2, y2 = min(frame_w, x + self.double_click_rect_width // 2), min(frame_h, y + self.double_click_rect_height // 2)
        
        if (x2 - x1) < self.min_object_size[0] or (y2 - y1) < self.min_object_size[1]:
            debug_print("[SEGMENTATION_ENGINE] Double-click area too small - selection ignored")
            return
        
        debug_print(f"[SEGMENTATION_ENGINE] Double-click detected - Creating {self.double_click_rect_width}x{self.double_click_rect_height} selection")
        self.temp_rectangle = ((x1, y1), (x2, y2))
        self.temp_rect_timer = time.time()
        
        # Update status
        if self.status_callback:
            self.status_callback("Processing...", "#39ff7a")
        
        self._process_selection((x1, y1), (x2, y2))
    
    def handle_right_click(self, x, y):
        """Handle right-click event - create smaller fixed-size selection."""
        if not self.current_frame or self.background_edges is None:
            debug_print("[SEGMENTATION_ENGINE] Right-click ignored - no frame or edges available")
            return
        
        # Use smaller selection size for right-click
        rect_width = 200
        rect_height = 200
        
        frame_h, frame_w = self.current_frame.shape[:2]
        x1, y1 = max(0, x - rect_width // 2), max(0, y - rect_height // 2)
        x2, y2 = min(frame_w, x1 + rect_width), min(frame_h, y1 + rect_height)
        
        if (x2 - x1) < self.min_object_size[0] or (y2 - y1) < self.min_object_size[1]:
            debug_print("[SEGMENTATION_ENGINE] Right-click area too small - selection ignored")
            return
        
        debug_print(f"[SEGMENTATION_ENGINE] Right-click detected - Creating {rect_width}x{rect_height} selection")
        self.temp_rectangle = ((x1, y1), (x2, y2))
        self.temp_rect_timer = time.time()
        
        # Update status with different color for right-click
        if self.status_callback:
            self.status_callback("Right-click Selection", "#ff6b35")
        
        self._process_selection((x1, y1), (x2, y2))
    
    def _get_box_from_slash(self, start, end):
        """Convert slash coordinates to bounding box."""
        x1, y1 = start
        x2, y2 = end
        
        # Ensure proper order (top-left to bottom-right)
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        
        return (x1, y1), (x2, y2)
    
    def _process_selection(self, p1, p2):
        """Process the selected region for segmentation and analysis."""
        debug_print(f"[SEGMENTATION_ENGINE] Processing selection from {p1} to {p2}")
        
        # Thread-safe frame access
        with self.display_lock:
            if self.current_frame is None:
                debug_print("[SEGMENTATION_ENGINE] No current frame available")
                return
            
            current_frame_copy = self.current_frame.copy()
        
        # Extract the selected region
        x1, y1 = p1
        x2, y2 = p2
        
        # Ensure coordinates are within frame bounds
        frame_h, frame_w = current_frame_copy.shape[:2]
        x1 = max(0, min(x1, frame_w - 1))
        y1 = max(0, min(y1, frame_h - 1))
        x2 = max(0, min(x2, frame_w - 1))
        y2 = max(0, min(y2, frame_h - 1))
        
        # Extract region
        region = current_frame_copy[y1:y2, x1:x2]
        
        if region.size == 0:
            debug_print("[SEGMENTATION_ENGINE] Selected region is empty")
            return
        
        # Create analysis task
        entity_id = int(time.time() * 1000) % 100000
        task = {
            'image': region,
            'entity_id': entity_id,
            'coordinates': ((x1 + x2) // 2, (y1 + y2) // 2),
            'bbox': (x1, y1, x2, y2),
            'timestamp': time.time()
        }
        
        # Queue for analysis
        self.analysis_queue.put(task)
        debug_print(f"[SEGMENTATION_ENGINE] Queued analysis task for region at ({x1}, {y1}) to ({x2}, {y2})")
        
        # Update entity panel if callback exists
        if self.entity_panel_callback:
            try:
                self.entity_panel_callback(entity_id, "Processing selection...", "streaming")
            except Exception as e:
                debug_print(f"[SEGMENTATION_ENGINE] Error updating entity panel: {e}")
    
    def _analysis_worker(self):
        """Background worker for processing analysis tasks."""
        while self.analysis_running:
            try:
                # Get task from queue with timeout
                try:
                    task = self.analysis_queue.get(timeout=0.1)
                except:
                    continue
                
                # Process the task
                self._analyze_region(task)
                
            except Exception as e:
                debug_print(f"[SEGMENTATION_ENGINE] Error in analysis worker: {e}")
                continue
    
    def _analyze_region(self, task):
        """Analyze the selected region and update entity panel."""
        entity_id = task['entity_id']
        region = task['image']
        coordinates = task['coordinates']
        bbox = task['bbox']
        
        try:
            # Simulate AI analysis (replace with actual AI processing)
            analysis_text = self._generate_analysis_text(region, coordinates, bbox)
            
            # Update entity panel
            if self.entity_panel_callback:
                self.entity_panel_callback(entity_id, analysis_text, "complete")
            
            # Reset status
            if self.status_callback:
                self.status_callback("Ready", "#39ff7a")
            
            debug_print(f"[SEGMENTATION_ENGINE] Analysis complete for entity {entity_id}")
            
        except Exception as e:
            error_text = f"Error analyzing region: {str(e)}"
            if self.entity_panel_callback:
                self.entity_panel_callback(entity_id, error_text, "error")
            
            debug_print(f"[SEGMENTATION_ENGINE] Analysis failed for entity {entity_id}: {e}")
    
    def _generate_analysis_text(self, region, coordinates, bbox):
        """Generate analysis text for the selected region."""
        try:
            # Basic image analysis
            height, width = region.shape[:2]
            area = width * height
            
            # Color analysis
            if len(region.shape) == 3:  # Color image
                mean_color = np.mean(region, axis=(0, 1))
                b, g, r = mean_color
                dominant_color = "Blue" if b > g and b > r else "Green" if g > r else "Red"
            else:  # Grayscale
                mean_intensity = np.mean(region)
                dominant_color = "Grayscale"
            
            # Edge density
            if self.background_edges is not None:
                edge_region = self.background_edges[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                edge_density = np.sum(edge_region > 0) / edge_region.size if edge_region.size > 0 else 0
            else:
                edge_density = 0
            
            # Generate analysis text
            analysis = f"""REGION ANALYSIS
Coordinates: {coordinates}
Bounding Box: {bbox}
Dimensions: {width} x {height} pixels
Area: {area:,} pixels²
Dominant Color: {dominant_color}
Edge Density: {edge_density:.2%}

Visual Characteristics:
- Region contains {width * height:,} pixels
- Average intensity: {np.mean(region):.1f}
- Standard deviation: {np.std(region):.1f}
- Aspect ratio: {width/height:.2f}

Analysis completed at {time.strftime('%H:%M:%S')}"""
            
            return analysis
            
        except Exception as e:
            return f"Error generating analysis: {str(e)}"
    
    def get_current_selection(self):
        """Get the current selection rectangle."""
        return self.temp_rectangle
    
    def clear_selection(self):
        """Clear the current selection."""
        self.temp_rectangle = None
        self.temp_rect_timer = None
    
    def draw_overlays(self, frame):
        """Draw segmentation overlays on the frame."""
        try:
            out = frame.copy()
            
            # Draw current selection rectangle if exists
            if self.temp_rectangle:
                p1, p2 = self.temp_rectangle
                x1, y1 = p1
                x2, y2 = p2
                
                # Draw selection rectangle
                cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 255), 2)
                
                # Draw corner markers
                cs, th = 8, 2
                color = (0, 255, 255)
                
                # Top-left corner
                cv2.line(out, (x1, y1 + cs), (x1 + cs, y1 + cs), color, th, cv2.LINE_AA)
                cv2.line(out, (x1 + cs, y1 + cs), (x1 + cs, y1), color, th, cv2.LINE_AA)
                
                # Top-right corner
                cv2.line(out, (x2 - cs, y1), (x2 - cs, y1 + cs), color, th, cv2.LINE_AA)
                cv2.line(out, (x2 - cs, y1 + cs), (x2, y1 + cs), color, th, cv2.LINE_AA)
                
                # Bottom-left corner
                cv2.line(out, (x1, y2 - cs), (x1 + cs, y2 - cs), color, th, cv2.LINE_AA)
                cv2.line(out, (x1 + cs, y2 - cs), (x1 + cs, y2), color, th, cv2.LINE_AA)
                
                # Bottom-right corner
                cv2.line(out, (x2 - cs, y2), (x2 - cs, y2 - cs), color, th, cv2.LINE_AA)
                cv2.line(out, (x2 - cs, y2 - cs), (x2, y2 - cs), color, th, cv2.LINE_AA)
                
                # Add label
                label = "SELECTION"
                font_scale = 0.6
                thickness = 2
                (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                
                # Background rectangle for text
                cv2.rectangle(out, (x1, y1 - text_height - 10), (x1 + text_width + 10, y1), (0, 0, 0), -1)
                cv2.putText(out, label, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), thickness, cv2.LINE_AA)
            
            # Draw current drawing state if slashing
            if self.slashing and self.slash_start and self.slash_end:
                cv2.line(out, self.slash_start, self.slash_end, (255, 0, 255), 2)
                
                # Draw start and end points
                cv2.circle(out, self.slash_start, 4, (255, 0, 255), -1)
                cv2.circle(out, self.slash_end, 4, (255, 0, 255), -1)
            
            return out
            
        except Exception as e:
            debug_print(f"[SEGMENTATION_ENGINE] Error drawing overlays: {e}")
            return frame
    
    def cleanup(self):
        """Clean up resources."""
        self.stop_analysis_thread()
        debug_print("[SEGMENTATION_ENGINE] Cleanup complete")

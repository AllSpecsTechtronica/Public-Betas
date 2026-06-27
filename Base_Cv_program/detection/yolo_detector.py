# YOLO Object Detection Module
# ═══════════════════════════════════════════════════════════════════════════════

import cv2
import time
import threading
from collections import deque
from ultralytics import YOLO

# Import from parent modules
from core.config import *
from utils.debug import debug_print
from detection.segmentation_engine import SegmentationEngine
from detection.edge_background import HairlineEdgeBackground

# Import pose estimation from parent directory
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
try:
    from PoseEstimation import PoseEstimator
except ImportError:
    PoseEstimator = None
    debug_print("[YOLO_DETECTOR] PoseEstimation not available")

class BackgroundObjectDetector:
    """YOLO-based object detection with tracking and pose estimation capabilities."""
    
    def __init__(self, model_path=MODEL_PATH, device=DEVICE, conf=CONF, iou=IOU,
                 priority_classes=PRIORITY_CLASSES, priority_conf=PRIORITY_CONF,
                 use_priority_classes=USE_PRIORITY_CLASSES):

        self.conf = conf
        self.priority_conf = priority_conf
        self.priority_classes = priority_classes
        self.use_priority_classes = use_priority_classes
        self.iou = iou
        self.img_size = IMG_SIZE
        self.max_det = MAX_DET
        self.device = self._get_best_device() if device == "auto" else device
        self._model_path = model_path

        # Model loaded lazily on background thread to avoid blocking startup
        self.model = None
        self.class_labels = {}
        self.available_classes = []
        self.enabled_classes = set()
        self._model_ready = threading.Event()

        # Threading and detection state
        self.filter_lock = threading.Lock()
        self.latest_detections = []
        self.detection_lock = threading.Lock()
        self.selected_track_ids = set()
        self.selection_lock = threading.Lock()
        self.frame_queue = deque(maxlen=1)
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run_detection_loop, daemon=True)

        # Initialize tracking and pose estimation
        self.segmentation_engine = SegmentationEngine()
        self.edge_background = HairlineEdgeBackground()
        self.detection_times = []
        self.detection_fps = 0
        self.pose_estimator = None
        self.frame_counter = 0
        self.inference_interval = 3  # Run detection every 3rd frame to lower CPU

        if ENABLE_POSE_ESTIMATION and PoseEstimator:
            try:
                self.pose_estimator = PoseEstimator()
                debug_print("[POSE_ESTIMATION] Initialized successfully")
            except Exception as e:
                debug_print(f"[POSE_ESTIMATION] Failed to initialize: {e}")

        debug_print(f"[OBJECT_DETECTION] Init complete, model will load on background thread (device: {self.device})")

    def start(self):
        """Start the background detection thread."""
        self.thread.start()
        debug_print("[OBJECT_DETECTION] Detection thread started.")

    def stop(self):
        """Stop the background detection thread and cleanup resources."""
        debug_print("[OBJECT_DETECTION] Stopping detection thread...")
        self.stop_event.set()
        self.thread.join()
        if self.pose_estimator and hasattr(self.pose_estimator, 'cleanup'):
            self.pose_estimator.cleanup()
        if self.segmentation_engine and hasattr(self.segmentation_engine, 'cleanup'):
            self.segmentation_engine.cleanup()
        debug_print("[OBJECT_DETECTION] Detection thread stopped.")

    def _get_best_device(self):
        """Automatically detect the best available device (CUDA, MPS, or CPU)."""
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass
        return "cpu"

    def handle_click(self, x, y):
        """Handle mouse clicks for object selection and double-click detection."""
        # Check if clicked on any detected object
        object_clicked = False
        with self.detection_lock:
            for d in self.latest_detections:
                x1, y1, x2, y2 = d['bbox']
                if x1 <= x <= x2 and y1 <= y <= y2:
                    if 'track_id' in d:
                        track_id = d['track_id']
                        with self.selection_lock:
                            if track_id in self.selected_track_ids:
                                self.selected_track_ids.remove(track_id)
                            else:
                                self.selected_track_ids.add(track_id)
                        
                        # Trigger segmentation within the bounding box
                        debug_print(f"[BBOX_CLICK] Triggering segmentation within bbox: ({x1}, {y1}) to ({x2}, {y2})")
                        if self.segmentation_engine:
                            self.segmentation_engine.process_selection((x1, y1), (x2, y2))
                        
                        object_clicked = True
                        break
        
        # Delegate to segmentation engine for general click handling
        if not object_clicked and self.segmentation_engine:
            # Check for double-click and handle accordingly
            current_time = time.time()
            click_pos = (x, y)
            
            if (hasattr(self.segmentation_engine, 'last_click_time') and 
                hasattr(self.segmentation_engine, 'last_click_pos') and
                self.segmentation_engine.last_click_pos is not None):
                
                time_diff = current_time - self.segmentation_engine.last_click_time
                pos_diff = ((x - self.segmentation_engine.last_click_pos[0])**2 + 
                           (y - self.segmentation_engine.last_click_pos[1])**2)**0.5
                
                if (time_diff <= self.segmentation_engine.double_click_threshold and 
                    pos_diff <= self.segmentation_engine.double_click_tolerance):
                    # Double click detected
                    self.segmentation_engine.handle_double_click(x, y)
                    self.segmentation_engine.last_click_pos = None
                    return
            
            # Update last click info for double-click detection
            self.segmentation_engine.last_click_time = current_time
            self.segmentation_engine.last_click_pos = click_pos

    def handle_mouse_down(self, x, y):
        """Handle mouse down events."""
        if self.segmentation_engine:
            self.segmentation_engine.handle_mouse_down(x, y)
    
    def handle_mouse_move(self, x, y):
        """Handle mouse move events."""
        if self.segmentation_engine:
            self.segmentation_engine.handle_mouse_move(x, y)
    
    def handle_mouse_up(self, x, y):
        """Handle mouse up events."""
        if self.segmentation_engine:
            self.segmentation_engine.handle_mouse_up(x, y)
        
        # Also trigger the click handler to check for single/double clicks
        self.handle_click(x, y)

    def update_frame(self, frame):
        """Update the current frame for detection and segmentation."""
        if self.segmentation_engine:
            self.segmentation_engine.update_frame(frame)
        self.frame_queue.append(frame)

    def _load_model_and_warmup(self):
        """Load the YOLO model and run a warmup inference (called on background thread)."""
        import numpy as np

        debug_print(f"[OBJECT_DETECTION] Loading YOLO model from {self._model_path}")
        self.model = YOLO(self._model_path)
        self.model.to(self.device)
        self.class_labels = self.model.names
        self.available_classes = list(self.model.names.values())
        with self.filter_lock:
            self.enabled_classes = set(self.available_classes)

        # Warmup: run a dummy inference so the first real frame isn't slow
        debug_print("[OBJECT_DETECTION] Running warmup inference...")
        dummy = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        try:
            self.model.track(
                source=dummy,
                device=self.device,
                imgsz=self.img_size,
                conf=self.conf,
                iou=self.iou,
                max_det=1,
                verbose=False,
                stream=False,
                persist=True,
                tracker="bytetrack.yaml",
            )
        except Exception as e:
            debug_print(f"[OBJECT_DETECTION] Warmup inference note: {e}")

        self._model_ready.set()
        debug_print("[OBJECT_DETECTION] Model loaded and warmed up")

    def _run_detection_loop(self):
        """Main detection loop running in background thread."""
        self._load_model_and_warmup()

        while not self.stop_event.is_set():
            if not self.frame_queue:
                time.sleep(0.01)
                continue
            frame = self.frame_queue.popleft()
            self.frame_counter += 1
            if self.frame_counter % self.inference_interval == 0:
                self._process_frame_background(frame)

    def _process_frame_background(self, frame):
        """Process a single frame for object detection."""
        if frame is None or self.model is None:
            return []
            
        start = time.time()
        results = self.model.track(
            source=frame, 
            device=self.device, 
            imgsz=self.img_size, 
            conf=self.conf, 
            iou=self.iou, 
            max_det=self.max_det, 
            verbose=False, 
            stream=False, 
            persist=True, 
            tracker="bytetrack.yaml"
        )
        
        detections = []
        for r in results:
            if r.boxes.id is None:
                continue
            for box, conf_score, cls, track_id in zip(r.boxes.xyxy, r.boxes.conf, r.boxes.cls, r.boxes.id):
                name = r.names[int(cls)]
                threshold = self.priority_conf if self.use_priority_classes and name in self.priority_classes else self.conf
                if conf_score < threshold:
                    continue
                    
                x1, y1, x2, y2 = map(int, box)
                detections.append({
                    'name': name, 
                    'display_name': "plant" if name == "potted plant" else name, 
                    'category': get_category(name), 
                    'type': get_type(name), 
                    'confidence': float(conf_score), 
                    'bbox': [x1, y1, x2, y2], 
                    'track_id': int(track_id)
                })
        
        with self.detection_lock:
            self.latest_detections = detections
        
        # Update FPS statistics
        dt = time.time() - start
        self.detection_times.append(dt)
        if len(self.detection_times) > 30:
            self.detection_times.pop(0)
        self.detection_fps = len(self.detection_times) / sum(self.detection_times) if sum(self.detection_times) > 0 else 0

    def draw_detections_on_frame(self, frame):
        """Draw all detections and overlays on the frame."""
        out = frame.copy()
        
        # Apply hairline edge background if enabled
        if ENABLE_BASE_EDGE_BACKGROUND:
            out = self.edge_background.apply(out)
        
        # Draw segmentation overlays
        if self.segmentation_engine:
            out = self.segmentation_engine.draw_overlays(out)
        
        # Draw detection bounding boxes and labels
        with self.detection_lock:
            dets = list(self.latest_detections)
        
        for d in dets:
            if d['name'] not in self.enabled_classes:
                continue
                
            x1, y1, x2, y2 = d['bbox']
            with self.selection_lock:
                is_selected = d.get('track_id') in self.selected_track_ids
            color = (255, 0, 0) if is_selected else (0, 255, 0)
            
            # Draw pose estimation for selected persons
            if (is_selected and d['name'] == 'person' and self.pose_estimator and 
                max(x2 - x1, y2 - y1) >= POSE_MIN_BBOX_SIZE):
                try:
                    out = self.pose_estimator.process_pose_in_bbox(out, x1, y1, x2, y2)
                except Exception as e:
                    debug_print(f"[POSE_ESTIMATION] Error: {e}")

            # Draw corner markers
            if SHOW_CORNER_MARKERS:
                cs, th = 12, 2
                overlay = out.copy()
                cv2.line(overlay, (x1, y1 + cs), (x1 + cs, y1 + cs), color, th, cv2.LINE_AA)
                cv2.line(overlay, (x1 + cs, y1 + cs), (x1 + cs, y1), color, th, cv2.LINE_AA)
                cv2.line(overlay, (x2 - cs, y1), (x2 - cs, y1 + cs), color, th, cv2.LINE_AA)
                cv2.line(overlay, (x2 - cs, y1 + cs), (x2, y1 + cs), color, th, cv2.LINE_AA)
                cv2.line(overlay, (x1, y2 - cs), (x1 + cs, y2 - cs), color, th, cv2.LINE_AA)
                cv2.line(overlay, (x1 + cs, y2 - cs), (x1 + cs, y2), color, th, cv2.LINE_AA)
                cv2.line(overlay, (x2 - cs, y2), (x2 - cs, y2 - cs), color, th, cv2.LINE_AA)
                cv2.line(overlay, (x2 - cs, y2 - cs), (x2, y2 - cs), color, th, cv2.LINE_AA)
                cv2.addWeighted(overlay, CORNER_TRANSPARENCY, out, 1 - CORNER_TRANSPARENCY, 0, out)

            # Draw detection arrows
            center_x, arrow_y = (x1 + x2) // 2, y1 - 20
            if SHOW_DETECTION_ARROWS:
                asz, ath = 10, 2
                arrow_overlay = out.copy()
                cv2.line(arrow_overlay, (center_x - asz//2, arrow_y), (center_x, arrow_y + asz), color, ath, cv2.LINE_AA)
                cv2.line(arrow_overlay, (center_x, arrow_y + asz), (center_x + asz//2, arrow_y), color, ath, cv2.LINE_AA)
                cv2.addWeighted(arrow_overlay, ARROW_TRANSPARENCY, out, 1 - ARROW_TRANSPARENCY, 0, out)
            
            # Draw labels
            if SHOW_OBJECT_TITLES or SHOW_CONFIDENCE_SCORES:
                parts = []
                if SHOW_OBJECT_TITLES:
                    parts.append(f"{d['display_name']} ({d['category']}, {d['type']})")
                if SHOW_CONFIDENCE_SCORES:
                    parts.append(f"{d['confidence']:.2f}")
                label = " ".join(parts)
                fs, tt = 0.4, 1
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, tt)
                lx, ly = center_x - tw // 2, arrow_y + 10 + th + 5
                cv2.rectangle(out, (lx - 2, ly - th - 2), (lx + tw + 2, ly + 2), color, -1)
                cv2.putText(out, label, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, fs, (0,0,0), tt, cv2.LINE_AA)
        
        return out

    def get_detections(self):
        """Get current detections (thread-safe)."""
        with self.detection_lock:
            return list(self.latest_detections)
    
    def get_detection_fps(self):
        """Get current detection FPS."""
        return self.detection_fps
    
    def handle_double_click(self, x, y):
        """Handle double-click event for segmentation."""
        if self.segmentation_engine:
            self.segmentation_engine.handle_double_click(x, y)

    def handle_right_click(self, x, y):
        """Handle right-click event for segmentation."""
        if self.segmentation_engine:
            self.segmentation_engine.handle_right_click(x, y)

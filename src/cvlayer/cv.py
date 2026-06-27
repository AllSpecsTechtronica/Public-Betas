# server.py
# --------------------
# Requirements:
#   pip install fastapi uvicorn aiortc opencv-python numpy ultralytics

import cv2
import json
import asyncio
import numpy as np
import threading
import base64
import os
import sys
import math
import argparse
from collections import deque
from datetime import datetime
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
from aiortc.rtcrtpparameters import RTCRtpEncodingParameters
from aiortc.mediastreams import VideoFrame
from ultralytics import YOLO
import uvicorn
import time
from queue import Queue, Empty
import unicodedata
import re
from openai import AsyncOpenAI, OpenAI

# Base paths anchored to the cvLayer directory for self-contained assets.
CVLAYER_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(CVLAYER_DIR, "assets")
MODELS_DIR = os.path.join(ASSETS_DIR, "models")
VIDEOS_DIR = os.path.join(ASSETS_DIR, "videos")
DEBUG_DIR = os.path.join(CVLAYER_DIR, "debug")
CONTEXT_WINDOWS_DIR = os.path.join(CVLAYER_DIR, "context_windows")

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# base imports for the cv layer's AI and LLM 
from PoseEstimation import PoseEstimator
from LibBinaries import api_key
from HighLightAdminPrompt import Admin_Prompt
from HighLightAnythingEngineMetaData import MetaData
from heatmap_haze import HeatmapHaze, HeatmapHazeConfig

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

# Runtime mode toggle (ui or terminal). Can be overridden by CV_MODE or --mode.
DEFAULT_RUN_MODE = "ui"  # Change to "terminal" to make terminal mode the default.
RUN_MODE = os.getenv("CV_MODE", DEFAULT_RUN_MODE).strip().lower()
if RUN_MODE not in ("ui", "terminal"):
    RUN_MODE = "ui"

# Global Debug Configuration
DEBUG_MODE = True  # Set to False to disable all print statements
ENHANCED_DEBUG = False  # Additional debug info for segmentation issues
VERBOSE_FRAME_LOGGING = False  # Set to True to enable detailed frame stats every 2 seconds

# Double-click Detection Constants
DOUBLE_CLICK_BASE_WIDTH = 420
DOUBLE_CLICK_BASE_HEIGHT = 420

# Detection Constants
MODEL_PATH = os.path.join(MODELS_DIR, "yolo26s.pt")
IMG_SIZE = 320
CONF = 0.25
IOU = 0.10
MAX_DET = 500
DEVICE = "auto"

# Display Options
SHOW_OBJECT_TITLES = False
SHOW_CONFIDENCE_SCORES = False
SHOW_DETECTION_ARROWS = True
SHOW_CORNER_MARKERS = False
ARROW_TRANSPARENCY = 0.8
CORNER_TRANSPARENCY = 0.8
ENABLE_BASE_EDGE_BACKGROUND = True  # Disable subtle base edge overlay to preserve crispness
USE_LIGHT_OBJECT_TITLES = False
SHOW_QUERY_BAR = True
ENABLE_HEATMAP_MODE = False
HEATMAP_ALPHA = 0.8
HEATMAP_DECAY = 0.6
HEATMAP_UPDATE_INTERVAL = 2
HEATMAP_MAX_OBJECTS = 12
HEATMAP_COLOR = (0, 200, 255)
HEATMAP_COOL_COLOR = (0, 255, 255)
HEATMAP_WARM_COLOR = (0, 0, 255)
HEATMAP_BLUR = 7
HEATMAP_MASK_BLUR = 9
HEATMAP_MOTION_THRESHOLD = 6.0
HEATMAP_MOTION_DECAY = 0.2
HEATMAP_MOTION_COOLDOWN_FRAMES = 2
HEATMAP_MASK_WEIGHT = 0.25
HEATMAP_OVERLAY_BLEND = 0.15
HEATMAP_INTENSITY = 4.0
HEATMAP_STABILIZER_WINDOW = 4
HEATMAP_STABILIZER_THRESHOLD = 10.0

# Entity Card UI
ENTITY_CARD_FONT_SIZE_PX = 11

# Pose Estimation Options
ENABLE_POSE_ESTIMATION = True
POSE_MIN_BBOX_SIZE = 60

# Priority Classes
USE_PRIORITY_CLASSES = True
PRIORITY_CLASSES = {"person", "car"}
PRIORITY_CONF = 0.10

CONFIGURABLE_CONSTANTS = [
    "DOUBLE_CLICK_BASE_WIDTH",
    "DOUBLE_CLICK_BASE_HEIGHT",
    "MODEL_PATH",
    "IMG_SIZE",
    "CONF",
    "IOU",
    "MAX_DET",
    "DEVICE",
    "SHOW_OBJECT_TITLES",
    "SHOW_CONFIDENCE_SCORES",
    "SHOW_DETECTION_ARROWS",
    "SHOW_CORNER_MARKERS",
    "ARROW_TRANSPARENCY",
    "CORNER_TRANSPARENCY",
    "ENABLE_BASE_EDGE_BACKGROUND",
    "USE_LIGHT_OBJECT_TITLES",
    "SHOW_QUERY_BAR",
    "ENABLE_HEATMAP_MODE",
    "HEATMAP_ALPHA",
    "HEATMAP_DECAY",
    "HEATMAP_UPDATE_INTERVAL",
    "HEATMAP_MAX_OBJECTS",
    "HEATMAP_COLOR",
    "HEATMAP_COOL_COLOR",
    "HEATMAP_WARM_COLOR",
    "HEATMAP_BLUR",
    "HEATMAP_MASK_BLUR",
    "HEATMAP_MOTION_THRESHOLD",
    "HEATMAP_MOTION_DECAY",
    "HEATMAP_MOTION_COOLDOWN_FRAMES",
    "HEATMAP_MASK_WEIGHT",
    "HEATMAP_OVERLAY_BLEND",
    "HEATMAP_INTENSITY",
    "ENTITY_CARD_FONT_SIZE_PX",
    "ENABLE_POSE_ESTIMATION",
    "POSE_MIN_BBOX_SIZE",
    "USE_PRIORITY_CLASSES",
    "PRIORITY_CLASSES",
    "PRIORITY_CONF",
]

def _prompt_bool(label, current):
    resp = input(f"{label} [{('y' if current else 'n')}]: ").strip().lower()
    if resp in ("y", "yes", "1", "true", "t"):
        return True
    if resp in ("n", "no", "0", "false", "f"):
        return False
    return current

def _prompt_choice(label, current, choices):
    resp = input(f"{label} [{current}]: ").strip().lower()
    if not resp:
        return current
    if resp in choices:
        return resp
    print(f"Invalid choice, keeping {current}.")
    return current

def _prompt_value(label, current):
    if isinstance(current, bool):
        return _prompt_bool(label, current)
    if isinstance(current, int):
        resp = input(f"{label} [{current}]: ").strip()
        if not resp:
            return current
        try:
            return int(resp)
        except ValueError:
            print(f"Invalid int, keeping {current}.")
            return current
    if isinstance(current, float):
        resp = input(f"{label} [{current}]: ").strip()
        if not resp:
            return current
        try:
            return float(resp)
        except ValueError:
            print(f"Invalid float, keeping {current}.")
            return current
    if isinstance(current, set):
        resp = input(f"{label} (comma-separated) [{', '.join(sorted(current))}]: ").strip()
        if not resp:
            return current
        return {item.strip() for item in resp.split(",") if item.strip()}
    resp = input(f"{label} [{current}]: ").strip()
    return resp if resp else current

def prompt_runtime_settings():
    if not sys.stdin.isatty():
        return
    global RUN_MODE, DEBUG_MODE, ENHANCED_DEBUG, VERBOSE_FRAME_LOGGING
    print("\nRuntime setup (press Enter to keep defaults)")
    RUN_MODE = _prompt_choice("Run mode (ui/terminal)", RUN_MODE, {"ui", "terminal"})
    DEBUG_MODE = _prompt_bool("DEBUG_MODE", DEBUG_MODE)
    ENHANCED_DEBUG = _prompt_bool("ENHANCED_DEBUG", ENHANCED_DEBUG)
    VERBOSE_FRAME_LOGGING = _prompt_bool("VERBOSE_FRAME_LOGGING", VERBOSE_FRAME_LOGGING)
    if _prompt_bool("Configure advanced settings", False):
        for name in CONFIGURABLE_CONSTANTS:
            try:
                globals()[name] = _prompt_value(name, globals()[name])
            except Exception:
                continue

def debug_print(*args, **kwargs):
    """A wrapper for print() that only prints if DEBUG_MODE is True."""
    if DEBUG_MODE:
        print(*args, **kwargs)

def enhanced_debug_print(*args, **kwargs):
    """Enhanced debug print for segmentation issues."""
    if DEBUG_MODE and ENHANCED_DEBUG:
        print("🔍 [ENHANCED_DEBUG]", *args, **kwargs)

# ═══════════════════════════════════════════════════════════════════════════════
# SESSION TRACKER - FOR GPT-4o BILLING AND USAGE
# ═══════════════════════════════════════════════════════════════════════════════

class SessionTracker:
    """Track session duration, API calls, and costs with GPT-4o pricing"""
    
    def __init__(self):
        self.session_start = datetime.now()
        self.analyses_count = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.session_id = int(time.time())
        
        # GPT-4o Pricing (in cents per token for precision)
        self.pricing = {
            'gpt-4o': {
                'input': 0.00025,  # $2.50 per 1M tokens = 0.00025 cents per token
                'output': 0.00025  # Same for regular gpt-4o
            },
            'gpt-4o-2024-08-06': {
                'input': 0.00025,   # $2.50 per 1M tokens = 0.00025 cents per token
                'output': 0.001     # $10.00 per 1M tokens = 0.001 cents per token
            }
        }
        
        self.analyses_history = []
        debug_print(f"[SESSION_TRACKER] Session {self.session_id} started at {self.session_start.strftime('%H:%M:%S')}")
    
    def get_session_duration(self):
        """Get session duration in minutes"""
        return (datetime.now() - self.session_start).total_seconds() / 60
    
    def calculate_cost(self, model, input_tokens, output_tokens):
        """Calculate cost in cents for given token usage"""
        if model not in self.pricing:
            model = 'gpt-4o'  # Default fallback
        
        input_cost = input_tokens * self.pricing[model]['input']
        output_cost = output_tokens * self.pricing[model]['output']
        total_cost = input_cost + output_cost
        
        return {
            'input_cost_cents': input_cost,
            'output_cost_cents': output_cost, 
            'total_cost_cents': total_cost,
            'total_cost_dollars': total_cost / 100
        }
    
    def record_analysis(self, model, input_tokens, output_tokens, analysis_text, rect_id):
        """Record a completed analysis with token usage and cost"""
        self.analyses_count += 1
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        
        cost_info = self.calculate_cost(model, input_tokens, output_tokens)
        
        analysis_record = {
            'analysis_id': self.analyses_count,
            'rect_id': rect_id,
            'timestamp': datetime.now().isoformat(),
            'model': model,
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'total_tokens': input_tokens + output_tokens,
            'cost_info': cost_info,
            'analysis_preview': analysis_text[:100] + '...' if len(analysis_text) > 100 else analysis_text
        }
        
        self.analyses_history.append(analysis_record)
        
        # Print real-time cost info
        duration_min = self.get_session_duration()
        total_session_cost = self.get_total_session_cost()
        
        debug_print(f"[SESSION_TRACKER] Analysis #{self.analyses_count} complete - {input_tokens + output_tokens} tokens | Cost: {cost_info['total_cost_cents']:.4f} cents")
        debug_print(f"[SESSION_TRACKER] Session: {duration_min:.1f}min | {self.analyses_count} analyses | {self.total_input_tokens + self.total_output_tokens:,} tokens | ${total_session_cost['total_cost_dollars']:.4f}")
        
        return analysis_record
    
    def get_total_session_cost(self):
        """Calculate total session cost across all analyses"""
        total_input_cost = 0
        total_output_cost = 0
        
        for analysis in self.analyses_history:
            cost_info = analysis['cost_info']
            total_input_cost += cost_info['input_cost_cents']
            total_output_cost += cost_info['output_cost_cents']
        
        total_cost_cents = total_input_cost + total_output_cost
        
        return {
            'input_cost_cents': total_input_cost,
            'output_cost_cents': total_output_cost,
            'total_cost_cents': total_cost_cents,
            'total_cost_dollars': total_cost_cents / 100
        }
    
    def save_session_report(self):
        """Save comprehensive session report to debug folder"""
        duration_min = self.get_session_duration()
        total_cost = self.get_total_session_cost()
        
        report_data = {
            'session_info': {
                'session_id': self.session_id,
                'start_time': self.session_start.isoformat(),
                'duration_minutes': duration_min,
                'analyses_count': self.analyses_count,
                'total_input_tokens': self.total_input_tokens,
                'total_output_tokens': self.total_output_tokens,
                'total_tokens': self.total_input_tokens + self.total_output_tokens
            },
            'cost_summary': total_cost,
            'analyses_history': self.analyses_history,
            'pricing_used': self.pricing
        }

        os.makedirs(DEBUG_DIR, exist_ok=True)
        
        # Save JSON report
        json_path = os.path.join(DEBUG_DIR, f"session_report_{self.session_id}.json")
        try:
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=2, ensure_ascii=False)
            debug_print(f"[SESSION_TRACKER] Report saved to: {json_path}")
        except Exception as e:
            debug_print(f"[SESSION_TRACKER] Error saving report: {e}")
        
        # Save human-readable report  
        txt_path = os.path.join(DEBUG_DIR, f"session_summary_{self.session_id}.txt")
        summary = f"""SESSION SUMMARY REPORT
========================================
Session ID: {self.session_id}
Start Time: {self.session_start.strftime('%Y-%m-%d %H:%M:%S')}
Duration: {duration_min:.1f} minutes
Total Analyses: {self.analyses_count}

TOKEN USAGE:
- Input Tokens: {self.total_input_tokens:,}
- Output Tokens: {self.total_output_tokens:,}  
- Total Tokens: {self.total_input_tokens + self.total_output_tokens:,}

COST BREAKDOWN:
- Input Cost: {total_cost['input_cost_cents']:.4f} cents
- Output Cost: {total_cost['output_cost_cents']:.4f} cents
- Total Cost: {total_cost['total_cost_cents']:.4f} cents (${total_cost['total_cost_dollars']:.4f})

PERFORMANCE METRICS:
- Average tokens per analysis: {(self.total_input_tokens + self.total_output_tokens) / max(1, self.analyses_count):.1f}
- Average cost per analysis: {total_cost['total_cost_cents'] / max(1, self.analyses_count):.4f} cents
- Cost per minute: {total_cost['total_cost_cents'] / max(1, duration_min):.4f} cents/min

========================================
End of Session Report
"""
        
        try:
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(summary)
            debug_print(f"[SESSION_TRACKER] Summary saved to: {txt_path}")
        except Exception as e:
            debug_print(f"[SESSION_TRACKER] Error saving summary: {e}")

# Initialize OpenAI client
try:
    # Clear any proxy-related environment variables that might interfere
    import os
    env_backup = {}
    proxy_vars = ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy']
    for var in proxy_vars:
        if var in os.environ:
            env_backup[var] = os.environ[var]
            del os.environ[var]
    
    # Read from environment or the sanitized LibBinaries compatibility shim.
    openai_key = os.getenv('OPENAI_API_KEY') or api_key
    if not openai_key or openai_key == 'none' or openai_key == 'your-openai-api-key-here':
        enhanced_debug_print("❌ OpenAI API key is not configured properly!")
        enhanced_debug_print("💡 To fix this:")
        enhanced_debug_print("   1. Go to https://platform.openai.com/api-keys")
        enhanced_debug_print("   2. Create a new API key")
        enhanced_debug_print("   3. Set OPENAI_API_KEY in your environment or an ignored .env file")
        enhanced_debug_print("   4. Restart the application")
        debug_print("Warning: No valid OpenAI API key found in environment")
        client = None
        async_client = None
    else:
        client = OpenAI(api_key=openai_key)
        async_client = AsyncOpenAI(api_key=openai_key)
        enhanced_debug_print(f"✅ OpenAI clients initialized successfully!")
        debug_print(f"OpenAI clients initialized with key starting with: {openai_key[:7]}...")
        # Detect Responses API availability for compatibility
        try:
            SUPPORTS_RESPONSES_API = hasattr(client, 'responses') and hasattr(async_client, 'responses')
        except Exception:
            SUPPORTS_RESPONSES_API = False
        debug_print(f"Responses API available: {SUPPORTS_RESPONSES_API}")
    
    # Restore environment variables
    os.environ.update(env_backup)
    
except Exception as e:
    debug_print(f"Warning: OpenAI client initialization failed: {e}")
    debug_print("Continuing without OpenAI client - analysis features will be disabled")
    client = None
    async_client = None


# ═══════════════════════════════════════════════════════════════════════════════
# ADVANCED SEGMENTATION ENGINE (FROM MACOS VERSION)
# ═══════════════════════════════════════════════════════════════════════════════

# Connection warming system
class ConnectionWarmer:
    def __init__(self):
        self.connection_warmed = False
        self.warmup_thread = None
    
    def warm_connection(self):
        """Warm up the connection with a minimal API call."""
        if self.connection_warmed or not client:
            return
        
        try:
            # Warm using Responses API if available; otherwise fallback to Chat Completions
            if 'SUPPORTS_RESPONSES_API' in globals() and SUPPORTS_RESPONSES_API:
                client.responses.create(
                    model="gpt-4o",
                    input=[{
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": "ready"}
                        ]
                    }]
                )
            else:
                client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": "ready"}],
                    max_tokens=1
                )
            self.connection_warmed = True
            debug_print("[CONNECTION_WARMER] ✓ OpenAI connection warmed successfully")
        except Exception as e:
            debug_print(f"[CONNECTION_WARMER] Connection warming failed: {e}")
    
    def warm_connection_async(self):
        """Start connection warming in background."""
        if not self.warmup_thread or not self.warmup_thread.is_alive():
            self.warmup_thread = threading.Thread(target=self.warm_connection, daemon=True)
            self.warmup_thread.start()

connection_warmer = ConnectionWarmer()

def TokenLoad():
    """Get the maximum token limit for LLM analysis requests."""
    TokenAmount = 3000
    return TokenAmount

def ModelPayLoad():
    """Get the default LLM model to use for analysis."""
    Payload = 'gpt-4o'
    return Payload

# Function to encode the in-memory image to base64 with optimization
def encode_image_from_array(image_array, max_dimension=800, quality=85):
    """Convert a numpy array image to base64 string with size optimization for LLM speed."""
    try:
        if image_array is None or image_array.size == 0:
            return None
            
        # Resize image if too large (for faster LLM processing)
        height, width = image_array.shape[:2]
        if max(height, width) > max_dimension:
            scale = max_dimension / max(height, width)
            new_width = int(width * scale)
            new_height = int(height * scale)
            image_array = cv2.resize(image_array, (new_width, new_height), interpolation=cv2.INTER_AREA)
            debug_print(f"[IMAGE_ENCODE] Resized from {width}x{height} to {new_width}x{new_height} for faster LLM processing")
        
        # Encode with quality setting for speed vs quality balance
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
        is_success, buffer = cv2.imencode(".jpg", image_array, encode_params)
        if not is_success:
            return None
        bytes_data = buffer.tobytes()
        base64_str = base64.b64encode(bytes_data).decode("utf-8")
        return base64_str
    except Exception as e:
        debug_print(f"[IMAGE_ENCODE] Error encoding image: {e}")
        return None



class DebugLogger:
    """Logger that prevents duplicate debug messages from being printed multiple times."""
    
    def __init__(self):
        self.shown_messages = set()

    def log_once(self, message):
        """Log a message only once, preventing duplicates."""
        if message not in self.shown_messages:
            debug_print(message)
            self.shown_messages.add(message)

    def log_with_data(self, template, *args):
        """Log a formatted message template only once, preventing duplicates."""
        if template not in self.shown_messages:
            debug_print(template.format(*args))
            self.shown_messages.add(template)

class CircuitBreaker:
    def __init__(self, fail_threshold=3, cooldown_seconds=45):
        self.fail_threshold = fail_threshold
        self.cooldown_seconds = cooldown_seconds
        self.fail_count = 0
        self.open_until = 0.0  # monotonic time

    def allow(self):
        return time.monotonic() >= self.open_until

    def record_success(self):
        self.fail_count = 0
        self.open_until = 0.0

    def record_failure(self):
        self.fail_count += 1
        if self.fail_count >= self.fail_threshold:
            self.open_until = time.monotonic() + self.cooldown_seconds

class MaskMemoryPool:
    """Pre-allocated pool of mask buffers to reduce memory allocations"""
    
    def __init__(self, max_buffers=10):
        self.max_buffers = max_buffers
        self.available_buffers = {}  # size -> deque of buffers
        self.buffer_map = {}  # buffer memory address -> buffer info
        self.lock = threading.RLock()  # Use RLock for better safety
    
    def _get_buffer_key(self, buffer):
        """Get a unique key for the buffer based on its memory address."""
        try:
            if buffer is None or not hasattr(buffer, '__array_interface__'):
                return None
            return buffer.__array_interface__['data'][0]
        except Exception:
            return None
        
    def get_buffer(self, height, width):
        # Validate input dimensions
        if height <= 0 or width <= 0:
            raise ValueError(f"Invalid buffer dimensions: {width}x{height}")
            
        try:
            with self.lock:
                size_key = (height, width)
                
                # Try to reuse an existing buffer
                if size_key in self.available_buffers and self.available_buffers[size_key]:
                    buffer = self.available_buffers[size_key].popleft()
                    buffer_key = self._get_buffer_key(buffer)
                    if buffer_key and buffer_key in self.buffer_map:
                        self.buffer_map[buffer_key]['in_use'] = True
                    return buffer
                
                # Create new buffer if under limit
                if len(self.buffer_map) < self.max_buffers:
                    buffer = np.zeros((height, width), dtype=np.uint8)
                    buffer_key = self._get_buffer_key(buffer)
                    if buffer_key:
                        self.buffer_map[buffer_key] = {'size': size_key, 'in_use': True}
                    return buffer
                
                # Fallback: return untracked buffer
                return np.zeros((height, width), dtype=np.uint8)
                
        except Exception as e:
            debug_print(f"[MASK_POOL] Error getting buffer: {e}")
            # Fallback: return untracked buffer
            return np.zeros((height, width), dtype=np.uint8)
    
    def return_buffer(self, buffer):
        if buffer is None:
            return
            
        try:
            with self.lock:
                buffer_key = self._get_buffer_key(buffer)
                if not buffer_key or buffer_key not in self.buffer_map:
                    return
                    
                buffer_info = self.buffer_map[buffer_key]
                if buffer_info.get('in_use', False):
                    buffer_info['in_use'] = False
                    size_key = buffer_info['size']
                    
                    if size_key not in self.available_buffers:
                        self.available_buffers[size_key] = deque()
                    
                    # Clear buffer before returning to pool
                    try:
                        buffer.fill(0)
                        self.available_buffers[size_key].append(buffer)
                    except Exception as e:
                        debug_print(f"[MASK_POOL] Error clearing buffer: {e}")
                        # Remove from tracking if can't be reused
                        del self.buffer_map[buffer_key]
                        
        except Exception as e:
            debug_print(f"[MASK_POOL] Error returning buffer: {e}")



class PredictiveProcessor:
    """Pre-computes edge maps for likely selection areas during mouse movement"""
    def __init__(self, cache_size=10):
        self.spatial_cache = {}
        self.cache_size = cache_size
        self.grid_size = 100
        self.last_mouse_pos = None
        self.last_mouse_time = 0
        self.mouse_velocity = (0, 0)
    
    def on_mouse_move(self, x, y, frame, background_edges):
        current_time = time.monotonic()
        if self.last_mouse_pos and self.last_mouse_time:
            dt = current_time - self.last_mouse_time
            if dt > 0:
                dx = x - self.last_mouse_pos[0]
                dy = y - self.last_mouse_pos[1]
                self.mouse_velocity = (dx/dt, dy/dt)
        self.last_mouse_pos, self.last_mouse_time = (x, y), current_time

        # Predict future position and pre-cache edges for that grid cell
        future_x = x + self.mouse_velocity[0] * 0.1  # Predict 100ms ahead
        future_y = y + self.mouse_velocity[1] * 0.1
        grid_key = (int(future_x) // self.grid_size, int(future_y) // self.grid_size)

        if grid_key in self.spatial_cache:
            return

        h, w = frame.shape[:2]
        area_x1 = grid_key[0] * self.grid_size
        area_y1 = grid_key[1] * self.grid_size
        area_x2 = min(w, area_x1 + self.grid_size)
        area_y2 = min(h, area_y1 + self.grid_size)
        
        edge_roi = background_edges[area_y1:area_y2, area_x1:area_x2]
        
        self.spatial_cache[grid_key] = {
            'edges': edge_roi.copy(),
            'area': (area_x1, area_y1, area_x2, area_y2),
            'timestamp': current_time
        }
        
        if len(self.spatial_cache) > self.cache_size:
            oldest_key = min(self.spatial_cache, key=lambda k: self.spatial_cache[k]['timestamp'])
            del self.spatial_cache[oldest_key]
    
    def get_cached_edges(self, x, y):
        grid_key = (x // self.grid_size, y // self.grid_size)
        if grid_key in self.spatial_cache:
            cache_data = self.spatial_cache[grid_key]
            x1, y1, x2, y2 = cache_data['area']
            if x1 <= x <= x2 and y1 <= y <= y2:
                return cache_data
        return None

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

class HighlightTracker:
    def __init__(self):
        self.debug_logger = DebugLogger()
        self.color_options = {'ui_color': (57, 255, 186), 'text_color': (255, 255, 255)}
        _ui_color = self.color_options['ui_color']
        _text_color = self.color_options['text_color']
        self.MetaData = True
        self.background_edges = None
        self.edge_buffer = deque(maxlen=5)
        self.shadow_threshold = 20
        self.min_object_size = (5, 5)
        self.min_contour_area = 20
        self.min_edge_displacement = 20
        self.highlighted_regions = []
        self.highlight_timers = {}
        self.highlight_duration = 0.3
        self.slashing = False
        self.slash_start = None
        self.slash_end = None
        self.temp_rect_timer = None
        self.temp_rect_duration = 0.1
        self.tracked_objects = []
        self.tracked_regions = []
        self.enable_edge_smoothing = True
        self.blur_kernel_size = (5, 5)
        self.canny_threshold1 = 50
        self.canny_threshold2 = 110
        self.use_adaptive_canny = True
        self.canny_sigma = 0.33
        self.morph_kernel = np.ones((3, 3), np.uint8)
        self.morph_iterations = 1
        self.roi_morph_kernel = np.ones((3, 3), np.uint8)
        self.roi_morph_iterations = 1
        self.roi_blur_kernel_size = (3, 3)
        self.roi_canny_sigma = 0.6
        self.roi_canny_threshold1 = 30
        self.roi_canny_threshold2 = 90
        self.roi_min_contour_area = 8
        self.filled_mask_dilate_iterations = 1
        self.prev_edges = None
        self.edge_smooth_factor = 0.95
        self.base_padding_factor = 0.1
        self.paused = False
        self.contour_thickness = 2
        self.edge_alpha = 0.55
        self.fill_alpha = 0.2
        self.segmentationFilterBasedMaskTransparency = 0.02  # Transparency for yellow background filter
        # Background removal mode in selection bbox (disabled when using alpha-blended overlay)
        self.background_removal_in_bbox = False
        self.foreground_mask_alpha = 0.82  # Overlay alpha (0.82 => 18% transparent)
        # Smooth yellow animation settings
        self.yellow_animation_curve = "sine"  # "sine", "ease", or "linear"
        self.yellow_min_intensity = 80        # Light yellow at start/end (0-255)
        self.yellow_max_intensity = 255       # Bright yellow at peak (0-255)
        self.yellow_min_transparency = 0.1    # Light transparency at start/end (0.0-1.0)
        self.yellow_max_transparency = 0.4    # Peak transparency in middle (0.0-1.0)
        self.default_color = _ui_color
        self.context_window_enabled = True
        self.context_save_quality = 95
        self.context_max_dimension = 400
        self.window_gap_percent = 5
        self.window_border_thickness = 1
        self.window_border_alpha = 0.6
        self.window_number_bg_alpha = 0.8
        self.window_border_color = _ui_color
        self.main_window_color = _ui_color
        self.external_numbering = True
        self.number_label_margin = 2
        self.number_box_size = 12
        self.enable_gods_eye_view = True
        self.gods_eye_border_color = _ui_color
        self.use_high_quality_frames = True
        # Speed optimization settings for LLM analysis
        self.gods_eye_simple_mode = True  # Use 1x3 layout instead of 3x3 for speed
        self.gods_eye_max_size = 600  # Max dimension for God's Eye View
        self.roi_max_size = 400  # Max dimension for ROI image
        self.SHOW_CONTEXT_DEBUG = False
        self.context_debug_mode = self.SHOW_CONTEXT_DEBUG
        self.debug_draw_overlays = False  # Toggle debug output for draw_overlays method
        self.analysis_display_duration = 1.0
        self.analysis_font_scale = 0.4
        self.analysis_font = cv2.FONT_HERSHEY_SIMPLEX
        self.analysis_font_thickness = 1
        self.analysis_line_height = 15
        self.analysis_padding = 8
        self.analysis_display_items = []
        self.analysis_tabs = []  # List of tab objects
        self.analysis_window_x_offset = 50
        self.box_animation_duration = 1.0
        self.box_line_thickness = 1
        self.streaming_futures = {}
        self.heatmap_haze = HeatmapHaze(
            config=HeatmapHazeConfig(
                alpha=HEATMAP_ALPHA,
                decay=HEATMAP_DECAY,
                update_interval=HEATMAP_UPDATE_INTERVAL,
                max_objects=HEATMAP_MAX_OBJECTS,
                cool_color=HEATMAP_COOL_COLOR,
                warm_color=HEATMAP_WARM_COLOR,
                blur=HEATMAP_BLUR,
                mask_blur=HEATMAP_MASK_BLUR,
                motion_threshold=HEATMAP_MOTION_THRESHOLD,
                motion_decay=HEATMAP_MOTION_DECAY,
                motion_cooldown_frames=HEATMAP_MOTION_COOLDOWN_FRAMES,
                mask_weight=HEATMAP_MASK_WEIGHT,
                overlay_blend=HEATMAP_OVERLAY_BLEND,
                intensity=HEATMAP_INTENSITY,
                stabilizer_window=HEATMAP_STABILIZER_WINDOW,
                stabilizer_threshold=HEATMAP_STABILIZER_THRESHOLD,
            ),
            enabled=ENABLE_HEATMAP_MODE,
        )
        self._edge_layer = None
        self._edge_layer_shape = None
        self.overlay_time_budget = 0.02
        self.overlay_recover_budget = 0.012
        self.last_heartbeat = 0.0
        self.display_lock = threading.Lock()
        # Edge computation throttle to reduce CPU and latency (increase interval for speed)
        self.edge_update_interval = 3  # compute background edges every 3 frames
        self._edge_update_counter = 0
        self.stop_event = threading.Event()
        self.llm_breaker = CircuitBreaker(fail_threshold=3, cooldown_seconds=45)
        self.enable_background_subtraction = True
        self.bg_learning_rate = 0.03
        self.bg_threshold = 25
        self.bg_morph_kernel = np.ones((3, 3), np.uint8)
        self.bg_morph_iterations = 1
        self.min_foreground_ratio = 0.01
        self.max_foreground_ratio = 0.95
        self.background_model = None
        self.foreground_mask = None
        self.async_loop = asyncio.new_event_loop()
        self.async_thread = threading.Thread(target=self._run_async_loop, daemon=True, name="StreamingAsyncLoop")
        self.async_thread.start()
        time.sleep(0.01)
        os.makedirs(DEBUG_DIR, exist_ok=True)
        os.makedirs(CONTEXT_WINDOWS_DIR, exist_ok=True)
        self.last_click_time = 0
        self.last_click_pos = None
        self.double_click_threshold = 0.5
        self.double_click_tolerance = 10
        self.double_click_rect_width = DOUBLE_CLICK_BASE_WIDTH
        self.double_click_rect_height = DOUBLE_CLICK_BASE_HEIGHT
        self.clear_context_windows_folder()
        self.analysis_queue = Queue(maxsize=2)
        self.analysis_thread = threading.Thread(target=self.analysis_worker, daemon=True)
        self.analysis_thread.start()
        self.session_tracker = SessionTracker()
        
        # Initialize helper classes before pre-initializing resources that use them
        self.predictive_processor = PredictiveProcessor()
        self.mask_pool = MaskMemoryPool(max_buffers=20)

        # Entity card session tracking to ensure a single card per user input
        self.next_entity_card_id = 1
        self.current_entity_card_id = None

        # Now, pre-initialize resources
        self.pre_initialize_resources()

        # Initialize frame and rectangle tracking variables
        self.temp_rectangle = None
        self.current_frame = None

    def start_new_input_session(self):
        """Start a new user input session so all analyses share one entity card."""
        try:
            self.current_entity_card_id = self.next_entity_card_id
            self.next_entity_card_id += 1
        except Exception:
            # Fallback safe increment
            self.current_entity_card_id = (self.current_entity_card_id or 0) + 1

    def _cleanup_tracker_at_index(self, idx):
        """Safely clean up a single tracker at the given index."""
        try:
            # Clean up tracker object
            if idx < len(self.tracked_objects):
                tracked_obj = self.tracked_objects[idx]
                
                # Try to release tracker resources if available
                if 'tracker' in tracked_obj and tracked_obj['tracker'] is not None:
                    try:
                        # Some trackers may have a release method
                        if hasattr(tracked_obj['tracker'], 'release'):
                            tracked_obj['tracker'].release()
                    except:
                        pass  # Ignore any release errors
                    
                    tracked_obj['tracker'] = None
                
                # Clean up tracker
                if 'tracker' in tracked_obj:
                    tracked_obj['tracker'] = None
                
                # Remove from list
                self.tracked_objects.pop(idx)
                debug_print(f"[CLEANUP] Removed tracker object at index {idx}")
            
            # Clean up corresponding region
            if idx < len(self.tracked_regions):
                region = self.tracked_regions[idx]
                
                # Return masks to pool if they exist
                if 'edge_mask' in region and region['edge_mask'] is not None:
                    self.mask_pool.return_buffer(region['edge_mask'])
                    region['edge_mask'] = None
                    
                if 'filled_mask' in region and region['filled_mask'] is not None:
                    self.mask_pool.return_buffer(region['filled_mask'])
                    region['filled_mask'] = None
                
                # Remove from list
                self.tracked_regions.pop(idx)
                debug_print(f"[CLEANUP] Removed tracked region at index {idx}")
                
        except Exception as e:
            debug_print(f"[CLEANUP] Error cleaning up tracker at index {idx}: {e}")

    def cleanup(self):
        """Clean up resources when the engine is stopped."""
        self.stop_event.set()
        try:
            self.analysis_queue.put_nowait(None)
        except Exception:
            pass
        with self.display_lock:
            # Clean up all tracked objects
            while len(self.tracked_objects) > 0:
                self._cleanup_tracker_at_index(0)
            
            # Cancel any running futures
            for future in self.streaming_futures.values():
                if future and not future.done():
                    future.cancel()
            self.streaming_futures.clear()

        if self.analysis_thread.is_alive():
            self.analysis_thread.join(timeout=1.0)
        
        # Stop async loop
        if self.async_loop.is_running():
            self.async_loop.call_soon_threadsafe(self.async_loop.stop)
        self.async_thread.join(timeout=1.0)
        
        # Save session report
        try:
            self.session_tracker.save_session_report()
        except Exception as e:
            debug_print(f"[CLEANUP] Error saving session report: {e}")
        
        debug_print("[HighlightTracker] Cleaned up resources.")

    def update_frame(self, frame):
        """Update the current frame for processing."""
        self.current_frame = frame

    @property
    def heatmap_enabled(self):
        return self.heatmap_haze.enabled

    @heatmap_enabled.setter
    def heatmap_enabled(self, enabled):
        self.heatmap_haze.enabled = bool(enabled)

    def _clear_heatmap_buffers(self):
        self.heatmap_haze.clear_buffers()

    def update_heatmap_from_detections(self, frame, detections):
        self.heatmap_haze.update_from_detections(
            frame,
            detections,
            compute_background_edges=self.compute_background_edges,
            min_contour_area=self.min_contour_area,
            foreground_mask=self.foreground_mask,
            min_foreground_ratio=self.min_foreground_ratio,
            max_foreground_ratio=self.max_foreground_ratio,
            roi_morph_iterations=self.roi_morph_iterations,
            roi_morph_kernel=self.roi_morph_kernel,
        )

    def _run_async_loop(self):
        debug_print("[DEBUG] Async event loop thread started and running.")
        asyncio.set_event_loop(self.async_loop)
        self.async_loop.run_forever()
    
    def pre_initialize_resources(self):
        debug_print("🚀 Pre-initializing resources for optimal performance...")
        connection_warmer.warm_connection_async()
        common_sizes = [(480, 640), (720, 1280), (1080, 1920), (360, 640)]
        for size in common_sizes:
            for _ in range(3):
                buffer = self.mask_pool.get_buffer(size[0], size[1])
                self.mask_pool.return_buffer(buffer)
        debug_print(f"✓ Pre-allocated {len(common_sizes) * 3} mask buffers")
        self.display_lock = threading.Lock()
        debug_print("✓ UI components pre-initialized")
        debug_print("✓ Edge detection system pre-initialized")
        debug_print("✓ Resource pre-initialization complete")
    
    async def stream_analysis(self, combined_image_base64, entity_id, retry_count=0, best_text=""):
        if not async_client:
            print()
            enhanced_debug_print(f"❌ ANALYSIS FAILED for object {entity_id}: OpenAI client not configured!")
            enhanced_debug_print(f"💡 SOLUTION: Set OPENAI_API_KEY in your environment or an ignored .env file")
            enhanced_debug_print(f"🔗 Get your API key from: https://platform.openai.com/api-keys")
            debug_print(f"[ANALYSIS] ERROR for object {entity_id}: OpenAI client not configured.")
            print()
            return

        if not self.llm_breaker.allow():
            target_id = self.current_entity_card_id if self.current_entity_card_id is not None else entity_id
            self.push_entity_card(target_id, "ANALYSIS TEMPORARILY DISABLED (cooldown)", status='done')
            return
        
        # Save raw image for potential retry
        saved_image_path = self.save_analysis_image(combined_image_base64, entity_id, retry_count)
        
        try:
            if not connection_warmer.connection_warmed: connection_warmer.warm_connection()
            model = ModelPayLoad()

            debug_print(f"\n--- Analysis for Object ID: {entity_id} (Attempt {retry_count + 1}) ---")

            full_text = ""
            usage = None
            suppress_streaming = False

            if 'SUPPORTS_RESPONSES_API' in globals() and SUPPORTS_RESPONSES_API:
                # Build Responses API input with text + image blocks
                input_blocks = [{
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": Admin_Prompt(MetaData(None))},
                        {"type": "input_text", "text": "The image below shows the user's selection in two views side-by-side:"},
                        {"type": "input_text", "text": "LEFT: The exact area the user selected (detailed view)"},
                        {"type": "input_text", "text": "RIGHT: The selected area with surrounding context for better understanding"},
                        {"type": "input_image", "image_url": f"data:image/jpeg;base64,{combined_image_base64}"},
                        {"type": "input_text", "text": "Please analyze the selected area using both the detailed view and the context to provide comprehensive insights."}
                    ]
                }]

                # Stream via Responses API
                async with async_client.responses.stream(
                    model=model,
                    input=input_blocks,
                ) as stream: 
                    async for event in stream:
                        try:
                            et = getattr(event, "type", "")
                            if et in ("response.output_text.delta", "response.delta"):
                                delta_text = getattr(event, "delta", "") or ""
                                if delta_text:
                                    print(delta_text, end="", flush=True)
                                    full_text += delta_text
                                    if not suppress_streaming:
                                        if self.is_llm_refusal(full_text):
                                            suppress_streaming = True
                                        else:
                                            target_id = self.current_entity_card_id if self.current_entity_card_id is not None else entity_id
                                            self.push_entity_card(target_id, full_text, status='streaming')
                            elif et == "response.error":
                                err = getattr(event, "error", None)
                                if err:
                                    debug_print(f"[ANALYSIS] Streaming error event: {err}")
                        except Exception:
                            pass

                    # Finalize and fetch usage
                    try:
                        final = await stream.get_final_response()
                        usage = getattr(final, "usage", None)
                    except Exception:
                        usage = None

                print()
                debug_print(f"--- End of Analysis for Object ID: {entity_id} (Attempt {retry_count + 1}) ---\n")
            else:
                # Fallback to Chat Completions streaming
                content = [
                    {"type": "text", "text": Admin_Prompt(MetaData(None))},
                    {"type": "text", "text": "The image below shows the user's selection in two views side-by-side:"},
                    {"type": "text", "text": "LEFT: The exact area the user selected (detailed view)"},
                    {"type": "text", "text": "RIGHT: The selected area with surrounding context for better understanding"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{combined_image_base64}"}},
                    {"type": "text", "text": "Please analyze the selected area using both the detailed view and the context to provide comprehensive insights."}
                ]

                stream = await async_client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": content}],
                    stream=True,
                    stream_options={"include_usage": True}
                )

                async for chunk in stream:
                    if chunk.choices:
                        delta = chunk.choices[0].delta
                        piece = getattr(delta, 'content', None) or ''
                        if piece:
                            print(piece, end='', flush=True)
                            full_text += piece
                            try:
                                if not suppress_streaming:
                                        if self.is_llm_refusal(full_text):
                                            suppress_streaming = True
                                        else:
                                            target_id = self.current_entity_card_id if self.current_entity_card_id is not None else entity_id
                                            self.push_entity_card(target_id, full_text, status='streaming')
                            except Exception:
                                pass
                    if getattr(chunk, 'usage', None):
                        usage = chunk.usage

                print()
                debug_print(f"--- End of Analysis for Object ID: {entity_id} (Attempt {retry_count + 1}) ---\n")

            # If refusal, retry without showing intermediate text
            if self.is_llm_refusal(full_text) and retry_count < 2:
                print("LLM timeout rerunning")
                debug_print(f"[ANALYSIS] LLM refusal detected for object {entity_id}, retrying...")
                enhanced_debug_print(f"🔄 LLM refused request, retrying analysis for object {entity_id}")
                best_text = full_text if len(full_text) > len(best_text) else best_text
                await asyncio.sleep(1)
                return await self.stream_analysis(combined_image_base64, entity_id, retry_count + 1, best_text)

            # Choose the most complete response across attempts
            chosen_text = full_text if len(full_text) >= len(best_text) else best_text

            # Single final UI update
            try:
                # Final text goes to the single entity card for this input session
                target_id = self.current_entity_card_id if self.current_entity_card_id is not None else entity_id
                self.push_entity_card(target_id, chosen_text, status='done')
            except Exception:
                pass
            
            # Schedule image cleanup if analysis was successful or max retries reached
            if not ENHANCED_DEBUG:  # Only cleanup if debugging is off
                self.schedule_image_cleanup(saved_image_path, 30)  # 30 seconds delay

            if usage:
                # Normalize token usage across APIs
                input_tokens = getattr(usage, 'input_tokens', None)
                output_tokens = getattr(usage, 'output_tokens', None)
                if input_tokens is None and isinstance(usage, dict):
                    input_tokens = usage.get('input_tokens')
                if output_tokens is None and isinstance(usage, dict):
                    output_tokens = usage.get('output_tokens')
                # Chat Completions fallback fields
                if input_tokens is None:
                    input_tokens = getattr(usage, 'prompt_tokens', None)
                if output_tokens is None:
                    output_tokens = getattr(usage, 'completion_tokens', None)
                if input_tokens is None and isinstance(usage, dict):
                    input_tokens = usage.get('prompt_tokens')
                if output_tokens is None and isinstance(usage, dict):
                    output_tokens = usage.get('completion_tokens')

                input_tokens = input_tokens or 0
                output_tokens = output_tokens or 0

                self.session_tracker.record_analysis(
                    model=model,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    analysis_text=chosen_text,
                    rect_id=(self.current_entity_card_id if self.current_entity_card_id is not None else entity_id)
                )
            else:
                debug_print("[ANALYSIS] WARNING: Could not get token usage from stream.")
            self.llm_breaker.record_success()

        except asyncio.CancelledError:
            debug_print(f"Streaming for object {entity_id} was cancelled.")
            # Cleanup image on cancellation
            if not ENHANCED_DEBUG:
                self.schedule_image_cleanup(saved_image_path, 0)  # Immediate cleanup
            self.llm_breaker.record_failure()
        except Exception as e:
            debug_print(f"Streaming error: {e}")
            # Cleanup image on error
            if not ENHANCED_DEBUG:
                self.schedule_image_cleanup(saved_image_path, 0)  # Immediate cleanup
            self.llm_breaker.record_failure()
        finally:
            with self.display_lock:
                self.streaming_futures.pop(entity_id, None)

    def is_llm_refusal(self, text):
        """Check if the LLM response contains refusal patterns"""
        if not text or not isinstance(text, str):
            return False
            
        # Convert to lowercase for case-insensitive matching
        text_lower = text.lower().strip()
        
        # Common refusal patterns
        refusal_patterns = [
            "i'm sorry, i can't assist with this request",
            "i'm sorry, i can't help with identifying or analyzing images of people",
            "i'm sorry, i can't help",
            "i'm sorry, i cannot",
            "i'm sorry, but i can't",
            "i'm sorry, but i cannot",
            "i cannot analyze",
            "i cannot assist",
            "i cannot help",
            "i'm not able to",
            "i cannot provide"
        ]
        
        # Check if text starts with "I'm sorry" (most common refusal pattern)
        if text_lower.startswith("i'm sorry"):
            debug_print(f"[LLM_REFUSAL] Detected refusal starting with 'I'm sorry': {text[:100]}...")
            return True
            
        # Check for specific refusal patterns
        for pattern in refusal_patterns:
            if pattern in text_lower:
                debug_print(f"[LLM_REFUSAL] Detected refusal pattern '{pattern}': {text[:100]}...")
                return True
                
        return False

    def save_analysis_image(self, base64_image, object_id, retry_count):
        """Save the base64 image to debug folder for potential retry"""
        try:
            if not base64_image:
                debug_print("[IMAGE_SAVE] No image data to save")
                return None
                
            # Decode base64 image
            import base64
            image_data = base64.b64decode(base64_image)
            
            # Create filename with timestamp and retry info
            timestamp = int(time.time())
            filename = f"analysis_retry_{object_id}_{timestamp}"
            if retry_count > 0:
                filename += f"_retry{retry_count}"
            filename += ".jpg"
            
            # Save to debug folder
            filepath = os.path.join(DEBUG_DIR, filename)
            
            with open(filepath, 'wb') as f:
                f.write(image_data)
                
            debug_print(f"[IMAGE_SAVE] Saved analysis image: {filepath}")
            return filepath
            
        except Exception as e:
            debug_print(f"[IMAGE_SAVE] Error saving analysis image: {e}")
            return None

    def schedule_image_cleanup(self, image_path, delay_seconds):
        """Schedule cleanup of saved analysis image after delay"""
        if not image_path or not os.path.exists(image_path):
            return
            
        def cleanup_image():
            try:
                time.sleep(delay_seconds)
                if os.path.exists(image_path):
                    os.remove(image_path)
                    debug_print(f"[IMAGE_CLEANUP] Removed analysis image: {image_path}")
                else:
                    debug_print(f"[IMAGE_CLEANUP] Image already removed: {image_path}")
            except Exception as e:
                debug_print(f"[IMAGE_CLEANUP] Error removing image {image_path}: {e}")
        
        # Run cleanup in background thread
        cleanup_thread = threading.Thread(target=cleanup_image, daemon=True)
        cleanup_thread.start()

    def sanitize_text_for_opencv(self, text):
        if not text: return ""
        text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')
        return re.sub(r'[ \t]+', ' ', text).strip()

    def format_final_analysis_text(self, text: str, object_id: int) -> str:
        """Prepare final analysis text for UI:
        - Remove '#' characters (markdown headings)
        - Ensure a title line at the top so the UI colors it yellow
        """
        try:
            safe_text = text or ""
            # Remove all '#' characters to avoid markdown-like headings
            safe_text = safe_text.replace('#', '')
            body = safe_text.strip()
            # Always provide a clear title as the first line
            title = f"SELECTION ANALYSIS {object_id}"
            if body:
                return f"{title}\n{body}"
            return title
        except Exception:
            # Fallback: plain text without hashes
            try:
                return (text or "").replace('#', '')
            except Exception:
                return "SELECTION ANALYSIS"
    
    def create_context_window(self, frame, selection_region):
        """Generate God's Eye View context window for better analysis"""
        try:
            # Validate inputs
            if frame is None or frame.size == 0:
                raise ValueError("Invalid frame provided")
                
            if len(selection_region) != 4:
                raise ValueError("Invalid selection region")
                
            x1, y1, x2, y2 = selection_region
            
            # Additional safety check for region size
            region_width, region_height = x2 - x1, y2 - y1
            region_area = region_width * region_height
            
            # If region is too large, use simple mode to prevent crashes
            max_safe_area = 500000  # 707x707 pixels
            if region_area > max_safe_area:
                debug_print(f"[CONTEXT_WINDOW] Large region detected ({region_width}x{region_height}), forcing simple mode")
                self.gods_eye_simple_mode = True
            
            # Validate coordinates
            if x1 >= x2 or y1 >= y2:
                raise ValueError(f"Invalid selection coordinates: ({x1}, {y1}) to ({x2}, {y2})")
                
            frame_h, frame_w = frame.shape[:2]
            
            # Clamp coordinates to frame bounds
            x1 = max(0, min(x1, frame_w - 1))
            y1 = max(0, min(y1, frame_h - 1))
            x2 = max(x1 + 1, min(x2, frame_w))
            y2 = max(y1 + 1, min(y2, frame_h))
            
            center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
            
            # Use simple mode for faster processing
            if self.gods_eye_simple_mode:
                return self._create_simple_context_window(frame, selection_region, center_x, center_y)
            
            # Calculate context window dimensions with safety limits
            window_w = max(100, min(400, (x2 - x1) * 2))  # Smaller for speed
            window_h = max(100, min(400, (y2 - y1) * 2))  # Smaller for speed
            
            # Create 3x3 grid of context windows
            try:
                context_grid = np.zeros((window_h * 3, window_w * 3, 3), dtype=np.uint8)
            except Exception as e:
                debug_print(f"[CONTEXT_WINDOW] Grid allocation failed: {e}")
                # Fallback to simple ROI
                return frame[y1:y2, x1:x2].copy()
            
            for row in range(3):
                for col in range(3):
                    try:
                        # Calculate offset for this grid position
                        offset_x = center_x + (col - 1) * window_w
                        offset_y = center_y + (row - 1) * window_h
                        
                        # Extract region with bounds checking
                        src_x1 = max(0, offset_x - window_w // 2)
                        src_y1 = max(0, offset_y - window_h // 2)
                        src_x2 = min(frame_w, offset_x + window_w // 2)
                        src_y2 = min(frame_h, offset_y + window_h // 2)
                        
                        # Ensure valid region
                        if src_x1 >= src_x2 or src_y1 >= src_y2:
                            continue
                        
                        # Extract source region
                        src_region = frame[src_y1:src_y2, src_x1:src_x2]
                        
                        # Calculate destination position in grid
                        dst_x = col * window_w
                        dst_y = row * window_h
                        
                        # Resize source region to fit grid cell
                        if src_region.size > 0 and src_region.shape[0] > 0 and src_region.shape[1] > 0:
                            try:
                                resized = cv2.resize(src_region, (window_w, window_h))
                                context_grid[dst_y:dst_y+window_h, dst_x:dst_x+window_w] = resized
                            except Exception as e:
                                debug_print(f"[CONTEXT_WINDOW] Resize failed for cell ({row}, {col}): {e}")
                                continue
                        
                        # Highlight the center window (where selection is)
                        if row == 1 and col == 1:
                            try:
                                # Draw border around center window
                                cv2.rectangle(context_grid, (dst_x, dst_y), (dst_x+window_w-1, dst_y+window_h-1), 
                                            self.default_color, 3)
                                
                                # Draw selection box within center window (with safety checks)
                                if src_x2 > src_x1 and src_y2 > src_y1:
                                    sel_x1 = dst_x + (x1 - src_x1) * window_w // (src_x2 - src_x1)
                                    sel_y1 = dst_y + (y1 - src_y1) * window_h // (src_y2 - src_y1)
                                    sel_x2 = dst_x + (x2 - src_x1) * window_w // (src_x2 - src_x1)
                                    sel_y2 = dst_y + (y2 - src_y1) * window_h // (src_y2 - src_y1)
                                    
                                    # Clamp to grid cell bounds
                                    sel_x1 = max(dst_x, min(sel_x1, dst_x + window_w - 1))
                                    sel_y1 = max(dst_y, min(sel_y1, dst_y + window_h - 1))
                                    sel_x2 = max(sel_x1 + 1, min(sel_x2, dst_x + window_w))
                                    sel_y2 = max(sel_y1 + 1, min(sel_y2, dst_y + window_h))
                                    
                                    cv2.rectangle(context_grid, (int(sel_x1), int(sel_y1)), (int(sel_x2), int(sel_y2)), 
                                                (0, 255, 0), 2)
                            except Exception as e:
                                debug_print(f"[CONTEXT_WINDOW] Center highlight failed: {e}")
                        
                    except Exception as e:
                        debug_print(f"[CONTEXT_WINDOW] Error processing cell ({row}, {col}): {e}")
                        continue
            
            return context_grid
            
        except Exception as e:
            debug_print(f"[CONTEXT_WINDOW] Critical error: {e}")
            # Safe fallback to simple ROI
            try:
                if frame is not None and len(selection_region) == 4:
                    x1, y1, x2, y2 = selection_region
                    frame_h, frame_w = frame.shape[:2]
                    x1 = max(0, min(x1, frame_w - 1))
                    y1 = max(0, min(y1, frame_h - 1))
                    x2 = max(x1 + 1, min(x2, frame_w))
                    y2 = max(y1 + 1, min(y2, frame_h))
                    return frame[y1:y2, x1:x2].copy()
                else:
                    # Ultimate fallback - small empty image
                    return np.zeros((100, 100, 3), dtype=np.uint8)
            except Exception as fallback_error:
                debug_print(f"[CONTEXT_WINDOW] Fallback failed: {fallback_error}")
                return np.zeros((100, 100, 3), dtype=np.uint8)

    def _create_simple_context_window(self, frame, selection_region, center_x, center_y):
        """Create a simple 1x3 horizontal context window for faster LLM processing"""
        try:
            x1, y1, x2, y2 = selection_region
            frame_h, frame_w = frame.shape[:2]
            
            # Calculate panel dimensions (smaller for speed)
            panel_w = max(150, min(300, (x2 - x1) * 2))
            panel_h = max(150, min(300, (y2 - y1) * 2))
            
            # Create horizontal 1x3 layout: [LEFT_CONTEXT | CENTER_SELECTION | RIGHT_CONTEXT]
            context_strip = np.zeros((panel_h, panel_w * 3, 3), dtype=np.uint8)
            
            contexts = [
                ("left", center_x - panel_w, center_y),      # Left context
                ("center", center_x, center_y),              # Center (selection)
                ("right", center_x + panel_w, center_y)      # Right context
            ]
            
            for i, (label, offset_x, offset_y) in enumerate(contexts):
                try:
                    # Extract region with bounds checking
                    src_x1 = max(0, offset_x - panel_w // 2)
                    src_y1 = max(0, offset_y - panel_h // 2)
                    src_x2 = min(frame_w, offset_x + panel_w // 2)
                    src_y2 = min(frame_h, offset_y + panel_h // 2)
                    
                    if src_x1 >= src_x2 or src_y1 >= src_y2:
                        continue
                    
                    # Extract and resize source region
                    src_region = frame[src_y1:src_y2, src_x1:src_x2]
                    if src_region.size > 0:
                        resized = cv2.resize(src_region, (panel_w, panel_h))
                        context_strip[0:panel_h, i*panel_w:(i+1)*panel_w] = resized
                    
                    # Highlight the center panel (selection)
                    if label == "center":
                        # Draw green border around center panel
                        start_x, end_x = i*panel_w, (i+1)*panel_w
                        cv2.rectangle(context_strip, (start_x, 0), (end_x-1, panel_h-1), (0, 255, 0), 3)
                        
                        # Draw selection box within center panel
                        if src_x2 > src_x1 and src_y2 > src_y1:
                            sel_x1 = start_x + (x1 - src_x1) * panel_w // (src_x2 - src_x1)
                            sel_y1 = (y1 - src_y1) * panel_h // (src_y2 - src_y1)
                            sel_x2 = start_x + (x2 - src_x1) * panel_w // (src_x2 - src_x1)
                            sel_y2 = (y2 - src_y1) * panel_h // (src_y2 - src_y1)
                            
                            # Clamp to panel bounds
                            sel_x1 = max(start_x, min(sel_x1, end_x - 1))
                            sel_y1 = max(0, min(sel_y1, panel_h - 1))
                            sel_x2 = max(sel_x1 + 1, min(sel_x2, end_x))
                            sel_y2 = max(sel_y1 + 1, min(sel_y2, panel_h))
                            
                            cv2.rectangle(context_strip, (int(sel_x1), int(sel_y1)), (int(sel_x2), int(sel_y2)), 
                                        (255, 255, 0), 2)  # Yellow for selection box
                    
                except Exception as e:
                    debug_print(f"[SIMPLE_CONTEXT] Error processing {label} panel: {e}")
                    continue
            
            debug_print(f"[SIMPLE_CONTEXT] Created {panel_w*3}x{panel_h} context strip for faster processing")
            return context_strip
            
        except Exception as e:
            debug_print(f"[SIMPLE_CONTEXT] Error creating simple context: {e}")
            # Fallback to just the ROI
            x1, y1, x2, y2 = selection_region
            frame_h, frame_w = frame.shape[:2]
            x1 = max(0, min(x1, frame_w - 1))
            y1 = max(0, min(y1, frame_h - 1))
            x2 = max(x1 + 1, min(x2, frame_w))
            y2 = max(y1 + 1, min(y2, frame_h))
            return frame[y1:y2, x1:x2].copy()

    def calculate_text_dimensions(self, text):
        sanitized_text = self.sanitize_text_for_opencv(text)
        lines = sanitized_text.split('\n')
        max_width = 0
        for line in lines:
            if line.strip():
                (line_width, _), _ = cv2.getTextSize(line.strip(), self.analysis_font, self.analysis_font_scale, self.analysis_font_thickness)
                max_width = max(max_width, line_width)
        total_height = len([l for l in lines if l.strip()]) * self.analysis_line_height
        return (max(200, max_width + (2 * self.analysis_padding * 3)), max(50, total_height + (2 * self.analysis_padding * 2)), lines)

    def clear_context_windows_folder(self):
        import glob
        pattern = os.path.join(CONTEXT_WINDOWS_DIR, "*.jpg")
        for f in glob.glob(pattern):
            try: os.remove(f)
            except OSError as e: debug_print(f"Error deleting {f}: {e}")

    def _adaptive_canny(self, gray):
        v = float(np.median(gray))
        sigma = float(self.canny_sigma)
        lower = int(max(0, (1.0 - sigma) * v))
        upper = int(min(255, (1.0 + sigma) * v))
        if lower == upper:
            lower = max(0, lower - 10)
            upper = min(255, upper + 10)
        return cv2.Canny(gray, lower, upper)

    def compute_background_edges(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, self.blur_kernel_size, 0)
        if self.use_adaptive_canny:
            edges = self._adaptive_canny(blurred)
        else:
            edges = cv2.Canny(blurred, self.canny_threshold1, self.canny_threshold2)
        edges = cv2.dilate(edges, self.morph_kernel, iterations=1)
        edges = cv2.erode(edges, self.morph_kernel, iterations=1)
        return cv2.morphologyEx(edges, cv2.MORPH_CLOSE, self.morph_kernel, iterations=self.morph_iterations)

    def compute_roi_edges(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.roi_blur_kernel_size:
            blurred = cv2.GaussianBlur(gray, self.roi_blur_kernel_size, 0)
        else:
            blurred = gray
        if self.use_adaptive_canny:
            v = float(np.median(blurred))
            sigma = float(self.roi_canny_sigma)
            lower = int(max(0, (1.0 - sigma) * v))
            upper = int(min(255, (1.0 + sigma) * v))
            if lower == upper:
                lower = max(0, lower - 10)
                upper = min(255, upper + 10)
            edges = cv2.Canny(blurred, lower, upper)
        else:
            edges = cv2.Canny(blurred, self.roi_canny_threshold1, self.roi_canny_threshold2)
        edges = cv2.dilate(edges, self.roi_morph_kernel, iterations=1)
        edges = cv2.erode(edges, self.roi_morph_kernel, iterations=1)
        return cv2.morphologyEx(edges, cv2.MORPH_CLOSE, self.roi_morph_kernel, iterations=self.roi_morph_iterations)

    def update_background_model(self, frame):
        if not self.enable_background_subtraction:
            return
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.background_model is None:
            self.background_model = gray.astype(np.float32)
            self.foreground_mask = np.zeros_like(gray, dtype=np.uint8)
            return
        cv2.accumulateWeighted(gray, self.background_model, self.bg_learning_rate)
        bg = cv2.convertScaleAbs(self.background_model)
        diff = cv2.absdiff(gray, bg)
        _, fg = cv2.threshold(diff, self.bg_threshold, 255, cv2.THRESH_BINARY)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, self.bg_morph_kernel, iterations=self.bg_morph_iterations)
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, self.bg_morph_kernel, iterations=self.bg_morph_iterations)
        self.foreground_mask = fg

    def get_box_from_slash(self, start, end):
        dx, dy = abs(end[0] - start[0]), abs(end[1] - start[1])
        base_padding = int(max(min(dx if dx > dy else dy, 40), 10) * self.base_padding_factor)
        padding_x = max(base_padding * 0.2, 5) if dx > dy else max(base_padding, 5)
        padding_y = max(base_padding, 5) if dx > dy else max(base_padding * 0.2, 5)
        x1, y1 = min(start[0], end[0]) - padding_x, min(start[1], end[1]) - padding_y
        x2, y2 = max(start[0], end[0]) + padding_x, max(start[1], end[1]) + padding_y
        return (int(x1), int(y1)), (int(x2), int(y2))
    
    def handle_mouse_down(self, x, y):
        debug_print(f"[MOUSE_DOWN] Starting selection at ({x}, {y})")
        # Start a new input session so downstream analyses share one entity card
        try:
            self.start_new_input_session()
        except Exception:
            pass
        self.slashing = True
        self.slash_start = (x, y)
        self.slash_end = (x, y)

    def handle_mouse_move(self, x, y):
        if self.slashing:
            self.slash_end = (x, y)

        if self.current_frame is not None and self.background_edges is not None:
            self.predictive_processor.on_mouse_move(x, y, self.current_frame, self.background_edges)

    def handle_mouse_up(self, x, y):
        if self.slashing:
            self.slashing = False
            if self.slash_start and self.slash_end and (abs(self.slash_end[0] - self.slash_start[0]) > 5 or abs(self.slash_end[1] - self.slash_start[1]) > 5):
                p1, p2 = self.get_box_from_slash(self.slash_start, self.slash_end)
                self.temp_rectangle = (p1, p2)
                self.temp_rect_timer = time.monotonic()
                enhanced_debug_print(f"🖱️ MOUSE DRAG DETECTED - Processing selection box")
                debug_print(f"[MOUSE_UP] Processing selection from ({self.slash_start[0]}, {self.slash_start[1]}) to ({self.slash_end[0]}, {self.slash_end[1]})")
                debug_print(f"[MOUSE_UP] Converted to box: {p1} to {p2}")
                self.process_selection(p1, p2)
            else:
                enhanced_debug_print(f"🖱️ Mouse drag too small - selection ignored")
            self.slash_start, self.slash_end = None, None



    def handle_double_click(self, x, y):
        if not self.current_frame or self.background_edges is None: 
            enhanced_debug_print(f"🖱️ Double-click ignored - no frame or edges available")
            return
        # Start a new input session for this interaction
        try:
            self.start_new_input_session()
        except Exception:
            pass
        frame_h, frame_w = self.current_frame.shape[:2]
        x1, y1 = max(0, x - self.double_click_rect_width // 2), max(0, y - self.double_click_rect_height // 2)
        x2, y2 = min(frame_w, x + self.double_click_rect_width // 2), min(frame_h, y + self.double_click_rect_height // 2)
        if (x2 - x1) < self.min_object_size[0] or (y2 - y1) < self.min_object_size[1]: 
            enhanced_debug_print(f"🖱️ Double-click area too small - selection ignored")
            return
        enhanced_debug_print(f"🖱️ DOUBLE-CLICK DETECTED - Creating {self.double_click_rect_width}x{self.double_click_rect_height} selection")
        self.temp_rectangle, self.temp_rect_timer = ((x1, y1), (x2, y2)), time.monotonic()
        self.process_selection((x1, y1), (x2, y2))

    def create_tracker(self):
        """Creates a tracker, trying several implementations for compatibility."""
        debug_print(f"[CREATE_TRACKER] Attempting to create tracker...")

        # Preferred order of trackers
        tracker_constructors = [
            ("CSRT", "TrackerCSRT_create"),
            ("KCF", "TrackerKCF_create"),
            ("MOSSE", "TrackerMOSSE_create"),
            ("MIL", "TrackerMIL_create"),
        ]

        for name, constructor_name in tracker_constructors:
            tracker = None
            try:
                # Try modern location (e.g., cv2.TrackerCSRT_create)
                if hasattr(cv2, constructor_name):
                    tracker = getattr(cv2, constructor_name)()
                    debug_print(f"[CREATE_TRACKER] Successfully created {name} tracker from modern API.")
                    return tracker
                # Try legacy location (e.g., cv2.legacy.TrackerCSRT_create)
                elif hasattr(cv2, 'legacy') and hasattr(cv2.legacy, constructor_name):
                    tracker = getattr(cv2.legacy, constructor_name)()
                    debug_print(f"[CREATE_TRACKER] Successfully created {name} tracker from legacy API.")
                    return tracker
            except cv2.error as e:
                debug_print(f"[CREATE_TRACKER] OpenCV error creating {name} tracker: {e}")
            except Exception as e:
                debug_print(f"[CREATE_TRACKER] Generic error creating {name} tracker: {e}")

        debug_print("[CREATE_TRACKER] FATAL: No suitable tracker found after trying all options.")
        return None

    def process_selection(self, p1, p2):
        enhanced_debug_print(f"🎯 SEGMENTATION STARTED - Selection from {p1} to {p2}")
        
        # Thread-safe frame and edge access with proper validation
        with self.display_lock:
            current_frame_copy = self.current_frame.copy() if self.current_frame is not None else None
            background_edges_copy = self.background_edges.copy() if self.background_edges is not None else None
            foreground_mask_copy = self.foreground_mask.copy() if self.foreground_mask is not None else None
        
        if current_frame_copy is None:
            enhanced_debug_print(f"❌ ERROR: No current frame available")
            debug_print(f"[PROCESS_SELECTION] ERROR: No current frame available")
            return
            
        # Validate coordinates first
        try:
            x1, y1 = max(0, min(p1[0], p2[0])), max(0, min(p1[1], p2[1]))
            x2, y2 = min(current_frame_copy.shape[1], max(p1[0], p2[0])), min(current_frame_copy.shape[0], max(p1[1], p2[1]))
            
            # Calculate ROI dimensions
            roi_width, roi_height = x2 - x1, y2 - y1
            roi_area = roi_width * roi_height
            
            # Ensure minimum size
            if roi_width < self.min_object_size[0] or roi_height < self.min_object_size[1]:
                debug_print(f"[PROCESS_SELECTION] Region too small: {roi_width}x{roi_height}")
                return
            
            # Add maximum size limit to prevent crashes from huge ROIs
            max_roi_dimension = 800  # Maximum width or height
            max_roi_area = 400000  # Maximum total pixels (roughly 632x632)
            
            if roi_width > max_roi_dimension or roi_height > max_roi_dimension or roi_area > max_roi_area:
                debug_print(f"[PROCESS_SELECTION] Region too large for stable processing: {roi_width}x{roi_height} ({roi_area} pixels)")
                debug_print(f"[PROCESS_SELECTION] Reducing to safe dimensions...")
                
                # Scale down to safe dimensions while maintaining aspect ratio
                scale_factor = min(max_roi_dimension / roi_width, max_roi_dimension / roi_height)
                if roi_area > max_roi_area:
                    area_scale = (max_roi_area / roi_area) ** 0.5
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
                
                debug_print(f"[PROCESS_SELECTION] Scaled to safe size: {x2-x1}x{y2-y1}")
                
        except Exception as e:
            debug_print(f"[PROCESS_SELECTION] Coordinate validation error: {e}")
            return
        
        # Fallback edge detection if background_edges not ready
        if background_edges_copy is None:
            debug_print(f"[PROCESS_SELECTION] Computing edges on-demand for region")
            try:
                frame_gray = cv2.cvtColor(current_frame_copy, cv2.COLOR_BGR2GRAY)
                background_edges_copy = cv2.Canny(frame_gray, self.canny_threshold1, self.canny_threshold2)
            except Exception as e:
                debug_print(f"[PROCESS_SELECTION] Emergency edge computation failed: {e}")
                return
        
        # Safe edge ROI extraction
        try:
            edge_roi = None
            center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
            cached_data = self.predictive_processor.get_cached_edges(center_x, center_y)

            if cached_data:
                cached_x1, cached_y1, cached_x2, cached_y2 = cached_data['area']
                if x1 >= cached_x1 and y1 >= cached_y1 and x2 <= cached_x2 and y2 <= cached_y2:
                    debug_print("[PROCESS_SELECTION] Using cached edge data.")
                    sel_x1_rel = x1 - cached_x1
                    sel_y1_rel = y1 - cached_y1
                    edge_roi = cached_data['edges'][sel_y1_rel:(y2 - cached_y1), sel_x1_rel:(x2 - cached_x1)]

            if edge_roi is None:
                debug_print(f"[PROCESS_SELECTION] Extracting ROI from background edges: ({x1}, {y1}) to ({x2}, {y2})")
                edge_roi = background_edges_copy[y1:y2, x1:x2]

            # Optional local adaptive edge pass for better contrast
            if self.use_adaptive_canny:
                roi_frame = current_frame_copy[y1:y2, x1:x2]
                if roi_frame.size > 0:
                    try:
                        local_edges = self.compute_roi_edges(roi_frame)
                        if local_edges is not None and local_edges.shape == edge_roi.shape:
                            edge_roi = local_edges
                    except Exception as e:
                        debug_print(f"[PROCESS_SELECTION] Local adaptive edges failed: {e}")

            # Apply foreground mask if it has meaningful coverage
            if foreground_mask_copy is not None:
                fg_roi = foreground_mask_copy[y1:y2, x1:x2]
                if fg_roi.size > 0:
                    fg_ratio = float(np.mean(fg_roi > 0))
                    if self.min_foreground_ratio <= fg_ratio <= self.max_foreground_ratio:
                        edge_roi = cv2.bitwise_and(edge_roi, fg_roi)

            # Morphological cleanup on ROI edges
            if self.roi_morph_iterations > 0:
                edge_roi = cv2.morphologyEx(edge_roi, cv2.MORPH_CLOSE, self.roi_morph_kernel, iterations=self.roi_morph_iterations)
                edge_roi = cv2.morphologyEx(edge_roi, cv2.MORPH_OPEN, self.roi_morph_kernel, iterations=self.roi_morph_iterations)
                
            # Validate edge ROI
            if edge_roi.size == 0:
                debug_print(f"[PROCESS_SELECTION] Empty edge ROI")
                return
                
        except Exception as e:
            debug_print(f"[PROCESS_SELECTION] Edge ROI extraction error: {e}")
            return

        # Safe contour detection
        try:
            contours, _ = cv2.findContours(edge_roi.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = [cnt for cnt in contours if cv2.contourArea(cnt) > self.roi_min_contour_area]
            debug_print(f"[PROCESS_SELECTION] Found {len(contours)} valid contours")
            
            if not contours: 
                debug_print(f"[PROCESS_SELECTION] No contours found above minimum area threshold")
                return

        except Exception as e:
            debug_print(f"[PROCESS_SELECTION] Contour detection error: {e}")
            return

        # Create masks from the found contours
        try:
            # The masks should be created relative to the full frame, so we need the bbox
            bbox_for_mask = (x1, y1, x2, y2)
            edge_mask, filled_mask = self.create_masks_from_contours(contours, bbox_for_mask)
            
            # Check if masks were created successfully
            if edge_mask is None or filled_mask is None:
                debug_print(f"[PROCESS_SELECTION] Mask creation failed.")
                return

        except Exception as e:
            debug_print(f"[PROCESS_SELECTION] Mask creation error: {e}")
            return


        debug_print(f"[PROCESS_SELECTION] Starting tracking setup for region ({x1}, {y1}) to ({x2}, {y2})")
        
        # Create and initialize tracker (simplified approach from Mac version)
        try:
            tracker = self.create_tracker()
            if tracker:
                bbox = (x1, y1, x2 - x1, y2 - y1)
                debug_print(f"[PROCESS_SELECTION] Initializing tracker with bbox: {bbox}")
                
                # Validate bbox dimensions
                if x2 - x1 <= 0 or y2 - y1 <= 0:
                    debug_print(f"[PROCESS_SELECTION] Invalid bbox dimensions")
                    return
                
                success = tracker.init(current_frame_copy, bbox)
                debug_print(f"[PROCESS_SELECTION] Tracker init result: {success}")

                # Some OpenCV trackers return None instead of True/False on macOS
                # If init() doesn't throw an exception and returns None, consider it successful
                tracker_initialized = (success is True) or (success is None)
                
                if tracker_initialized:
                    object_id = len(self.tracked_objects)
                    debug_print(f"[PROCESS_SELECTION] Successfully created tracked object {object_id}")

                    # Add to tracked objects list
                    self.tracked_objects.append({
                        'id': object_id,
                        'tracker': tracker,
                        'start_time': time.monotonic(),
                        'color': self.default_color,
                        'bbox': bbox,
                        'analysis': None
                    })

                    # Add to tracked regions list
                    self.tracked_regions.append({
                        'id': object_id,
                        'edge_mask': edge_mask.copy() if edge_mask is not None else None,
                        'filled_mask': filled_mask.copy() if filled_mask is not None else None,
                        'color': self.default_color
                    })
                    
                    enhanced_debug_print(f"🎯 Object {object_id} added to tracking") 
                    enhanced_debug_print(f"📊 Total tracked objects: {len(self.tracked_objects)}")
                    
                    # Schedule analysis if OpenAI client is available
                    if async_client is not None:
                        try:
                            # Extract ROI and context for analysis
                            roi_x1, roi_y1 = max(0, x1), max(0, y1)
                            roi_x2, roi_y2 = min(current_frame_copy.shape[1], x2), min(current_frame_copy.shape[0], y2)
                            
                            if roi_x1 < roi_x2 and roi_y1 < roi_y2:
                                roi_image = current_frame_copy[roi_y1:roi_y2, roi_x1:roi_x2]
                                context_image = current_frame_copy  # Use full frame as context
                                
                                if roi_image.size > 0 and context_image.size > 0:
                                    # Queue analysis task (bind to current input session card id)
                                    analysis_task = {
                                        'object_id': object_id,
                                        'entity_id': (self.current_entity_card_id if self.current_entity_card_id is not None else object_id),
                                        'roi': roi_image,
                                        'combined': context_image,
                                        'region': (roi_x1, roi_y1, roi_x2, roi_y2),
                                        'frame_shape': current_frame_copy.shape
                                    }
                                    try:
                                        self.analysis_queue.put(analysis_task, timeout=0.05)
                                        enhanced_debug_print(f"🔄 Analysis queued for object {object_id}")
                                    except Exception as queue_error:
                                        debug_print(f"[ANALYSIS] Dropped analysis task (queue full): {queue_error}")
                                else:
                                    debug_print(f"[PROCESS_SELECTION] Invalid ROI or context image, skipping analysis")
                        except Exception as e:
                            debug_print(f"[PROCESS_SELECTION] Analysis scheduling error: {e}")
                else:
                    debug_print(f"[PROCESS_SELECTION] Tracker initialization failed")
            else:
                debug_print(f"[PROCESS_SELECTION] Failed to create tracker")
                
        except Exception as e:
            debug_print(f"[PROCESS_SELECTION] Tracker setup error: {e}")
            import traceback
            traceback.print_exc()

    def create_masks_from_contours(self, contours, bbox):
        """Create edge and filled masks from contours with thread-safe buffer management."""
        try:
            x1, y1, x2, y2 = bbox
            height, width = y2-y1, x2-x1
            
            # Validate dimensions
            if height <= 0 or width <= 0:
                raise ValueError(f"Invalid mask dimensions: {width}x{height}")
            
            # Get current frame shape safely
            with self.display_lock:
                if self.current_frame is None:
                    raise ValueError("No current frame available for mask creation")
                frame_height, frame_width = self.current_frame.shape[:2]
            
            # Thread-safe buffer allocation
            try:
                edge_mask = self.mask_pool.get_buffer(height, width)
                filled_mask = self.mask_pool.get_buffer(height, width)
                full_edge_mask = self.mask_pool.get_buffer(frame_height, frame_width)
                full_filled_mask = self.mask_pool.get_buffer(frame_height, frame_width)
            except Exception as e:
                debug_print(f"[MASK_CREATION] Buffer allocation failed: {e}")
                raise
            
            try:
                # Draw contours safely
                for cnt in contours:
                    if cnt is not None and len(cnt) > 0:
                        cv2.drawContours(edge_mask, [cnt], -1, 255, self.contour_thickness)
                        cv2.drawContours(filled_mask, [cnt], -1, 255, -1)

                if self.filled_mask_dilate_iterations > 0:
                    cv2.dilate(
                        filled_mask,
                        self.roi_morph_kernel,
                        iterations=self.filled_mask_dilate_iterations,
                        dst=filled_mask,
                    )
                
                # Copy to full-frame masks with bounds checking
                try:
                    # Ensure we don't exceed frame boundaries
                    y1_clipped = max(0, min(y1, frame_height))
                    y2_clipped = max(0, min(y2, frame_height))
                    x1_clipped = max(0, min(x1, frame_width))
                    x2_clipped = max(0, min(x2, frame_width))
                    
                    # Calculate actual dimensions after clipping
                    actual_height = y2_clipped - y1_clipped
                    actual_width = x2_clipped - x1_clipped
                    
                    if actual_height > 0 and actual_width > 0:
                        # Resize masks if needed to match clipped dimensions
                        if edge_mask.shape != (actual_height, actual_width):
                            edge_mask = cv2.resize(edge_mask, (actual_width, actual_height), interpolation=cv2.INTER_NEAREST)
                            filled_mask = cv2.resize(filled_mask, (actual_width, actual_height), interpolation=cv2.INTER_NEAREST)
                        
                        full_edge_mask[y1_clipped:y2_clipped, x1_clipped:x2_clipped] = edge_mask
                        full_filled_mask[y1_clipped:y2_clipped, x1_clipped:x2_clipped] = filled_mask
                    else:
                        debug_print(f"[MASK_CREATION] Invalid clipped dimensions: {actual_width}x{actual_height}")
                        raise ValueError("Invalid clipped dimensions")
                        
                except Exception as copy_error:
                    debug_print(f"[MASK_CREATION] Mask copy error: {copy_error}")
                    debug_print(f"[MASK_CREATION] Frame: {frame_height}x{frame_width}, Bbox: ({x1},{y1})-({x2},{y2}), Mask: {edge_mask.shape}")
                    raise
                
                # Return buffers to pool
                self.mask_pool.return_buffer(edge_mask)
                self.mask_pool.return_buffer(filled_mask)
                
                return full_edge_mask, full_filled_mask
                
            except Exception as e:
                # Clean up buffers on error
                self.mask_pool.return_buffer(edge_mask)
                self.mask_pool.return_buffer(filled_mask)
                self.mask_pool.return_buffer(full_edge_mask)
                self.mask_pool.return_buffer(full_filled_mask)
                debug_print(f"[MASK_CREATION] Contour drawing failed: {e}")
                raise
                
        except Exception as e:
            debug_print(f"[MASK_CREATION] Failed to create masks: {e}")
            # Return empty masks as fallback (create new ones, don't reuse returned buffers)
            frame_height, frame_width = 480, 640  # Fallback dimensions
            with self.display_lock:
                if self.current_frame is not None:
                    frame_height, frame_width = self.current_frame.shape[:2]
            
            # Create new empty masks instead of reusing potentially corrupted buffers
            try:
                empty_edge = np.zeros((frame_height, frame_width), dtype=np.uint8)
                empty_fill = np.zeros((frame_height, frame_width), dtype=np.uint8)
                return empty_edge, empty_fill
            except Exception as fallback_error:
                debug_print(f"[MASK_CREATION] Even fallback failed: {fallback_error}")
                # Ultimate fallback - return very basic masks
                return np.zeros((100, 100), dtype=np.uint8), np.zeros((100, 100), dtype=np.uint8)

    def analysis_worker(self):
        while not self.stop_event.is_set():
            task = None
            got_task = False
            try:
                # Get task with timeout to prevent indefinite blocking
                try:
                    task = self.analysis_queue.get(timeout=1.0)
                    got_task = True
                except Empty:
                    continue  # Timeout, check again
                    
                if task is None: 
                    break
                    
                object_id = task.get('object_id', 'unknown')
                entity_id = task.get('entity_id', object_id)
                debug_print(f"[ANALYSIS_WORKER] Processing analysis for object {object_id}")
                
                # Thread-safe future management
                try:
                    with self.display_lock:
                        future = self.streaming_futures.get(entity_id)
                        if future and not future.done():
                            future.cancel()
                except Exception as e:
                    debug_print(f"[ANALYSIS_WORKER] Future cancellation error: {e}")
                
                # Safe image selection and validation for both ROI and context
                try:
                    roi_image = None
                    context_image = None
                    
                    # Get the raw ROI (region of interest) - the exact selected area
                    if 'roi' in task and task['roi'] is not None:
                        roi_img = task['roi']
                        if hasattr(roi_img, 'size') and roi_img.size > 0:
                            roi_image = roi_img
                    
                    # Get the context image (God's Eye View)  
                    if 'combined' in task and task['combined'] is not None:
                        combined_img = task['combined']
                        if hasattr(combined_img, 'size') and combined_img.size > 0:
                            context_image = combined_img
                    
                    # Both images are required for analysis
                    if roi_image is None:
                        debug_print(f"[ANALYSIS_WORKER] No valid ROI image for object {object_id}")
                        continue
                        
                    if context_image is None:
                        debug_print(f"[ANALYSIS_WORKER] No valid context image for object {object_id}")
                        continue
                        
                except Exception as e:
                    debug_print(f"[ANALYSIS_WORKER] Image validation error for object {object_id}: {e}")
                    continue

                # Create the combined side-by-side image for maximum efficiency
                try:
                    # Let create_combined_analysis_image handle resizing to safe sizes
                    combined_analysis_image = self.create_combined_analysis_image(roi_image, context_image)
                    if combined_analysis_image is None:
                        debug_print(f"[ANALYSIS_WORKER] Failed to create combined image for object {object_id}")
                        continue
                except Exception as e:
                    debug_print(f"[ANALYSIS_WORKER] Combined image creation error for object {object_id}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue

                # Safe base64 encoding of the single combined image
                try:
                    # Encode the combined image with optimal settings for LLM speed
                    combined_base64 = encode_image_from_array(
                        combined_analysis_image,
                        max_dimension=900,   # smaller for speed
                        quality=75           # lower for faster encode/transmit
                    )
                    
                    if not combined_base64:
                        debug_print(f"[ANALYSIS_WORKER] Failed to encode combined image for object {object_id}")
                        continue
                        
                except Exception as e:
                    debug_print(f"[ANALYSIS_WORKER] Base64 encoding error for object {object_id}: {e}")
                    continue
                
                # Safe async analysis launch with the single combined image
                try:
                    debug_print(f"[ANALYSIS_WORKER] Starting combined-image analysis for object {object_id}")
                    debug_print(f"[ANALYSIS_WORKER] Original - ROI: {roi_image.shape}, Context: {context_image.shape}")
                    debug_print(f"[ANALYSIS_WORKER] Combined image: {combined_analysis_image.shape}")
                    debug_print(f"[ANALYSIS_WORKER] Encoded size: {len(combined_base64)} chars")
                    debug_print(f"[ANALYSIS_WORKER] Speed mode: Simple={self.gods_eye_simple_mode}, Max_dimension=1200")
                    
                    # Ensure the stream uses the single entity card ID for this input session
                    coro = self.stream_analysis(combined_base64, entity_id)
                    future = asyncio.run_coroutine_threadsafe(coro, self.async_loop)
                    
                    with self.display_lock:
                        self.streaming_futures[entity_id] = future
                        
                except Exception as e: 
                    debug_print(f"[ANALYSIS_WORKER] Failed to start analysis for object {object_id}: {e}")
                    
            except Exception as e: 
                debug_print(f"[ANALYSIS_WORKER] Unexpected error: {e}")
                import traceback
                traceback.print_exc()
                
            finally: 
                if got_task:
                    try:
                        self.analysis_queue.task_done()
                    except Exception:
                        pass  # Task might not have been properly gotten

    def apply_masks_optimized(self, frame_shape, edge_mask, fill_mask, color):
        edge_layer, fill_layer = np.zeros(frame_shape, dtype=np.float32), np.zeros(frame_shape, dtype=np.float32)
        edge_layer[edge_mask > 0], fill_layer[fill_mask > 0] = color, color
        return (edge_layer * self.edge_alpha).astype(np.uint8), (fill_layer * self.fill_alpha).astype(np.uint8)

    def _get_edge_layer(self, frame):
        shape = frame.shape
        if self._edge_layer is None or self._edge_layer_shape != shape:
            self._edge_layer = np.zeros(shape, dtype=np.float32)
            self._edge_layer_shape = shape
        else:
            self._edge_layer.fill(0)
        return self._edge_layer

    def _apply_mask_overlay(self, frame, mask, color, alpha):
        if mask is None or alpha <= 0:
            return frame
        if mask.shape[:2] != frame.shape[:2]:
            return frame
        mask_idx = mask > 0
        if not np.any(mask_idx):
            return frame
        inv_alpha = 1.0 - alpha
        color_arr = np.array(color, dtype=np.float32)
        blended = frame[mask_idx].astype(np.float32) * inv_alpha + color_arr * alpha
        frame[mask_idx] = blended.astype(np.uint8)
        return frame

    def _apply_heatmap(self, frame):
        return self.heatmap_haze.apply(frame)

    def draw_overlays(self, frame):
        try:
            start_ts = time.monotonic()
            # Thread-safe frame update
            with self.display_lock:
                if self.current_frame is None: 
                    self.current_frame = frame.copy()
                try:
                    self.update_background_model(frame)
                except Exception as e:
                    debug_print(f"[DRAW_OVERLAYS] Background model update error: {e}")
            
            # Safe background edge computation
            try:
                # Compute background edges intermittently to reduce latency
                self._edge_update_counter = (self._edge_update_counter + 1) % max(1, self.edge_update_interval)
                if self._edge_update_counter == 0 or self.background_edges is None:
                    edges = self.compute_background_edges(frame)
                    self.edge_buffer.append(edges)
                    self.background_edges = (
                        np.mean(self.edge_buffer, axis=0).astype(np.uint8)
                        if len(self.edge_buffer) == self.edge_buffer.maxlen else edges
                    )
            except Exception as e:
                debug_print(f"[DRAW_OVERLAYS] Background edge computation error: {e}")
                # Use previous edges or create empty
                if self.background_edges is None:
                    self.background_edges = np.zeros(frame.shape[:2], dtype=np.uint8)
        
            edge_layer = self._get_edge_layer(frame)
        
            objects_to_remove = []
            
            # Thread-safe access to tracked objects
            with self.display_lock:
                tracked_objects_copy = list(self.tracked_objects)
                
            if len(tracked_objects_copy) > 0:
                if self.debug_draw_overlays:
                    debug_print(f"[DRAW_OVERLAYS] Processing {len(tracked_objects_copy)} tracked objects")

                for idx, tracked_obj in enumerate(tracked_objects_copy):
                    try:
                        # Safe tracker update with comprehensive validation
                        success = False
                        raw_bbox = None
                        
                        try:
                            # Validate frame before tracker update
                            if frame is None or frame.size == 0:
                                debug_print(f"[DRAW_OVERLAYS] Invalid frame for tracker {idx}")
                                objects_to_remove.append(idx)
                                continue
                                
                            # Validate tracker exists
                            if 'tracker' not in tracked_obj or tracked_obj['tracker'] is None:
                                debug_print(f"[DRAW_OVERLAYS] Missing tracker for object {idx}")
                                objects_to_remove.append(idx)
                                continue
                            
                            # Perform tracker update
                            update_result = tracked_obj['tracker'].update(frame)
                            
                            # Handle different return formats from OpenCV trackers
                            if isinstance(update_result, tuple) and len(update_result) >= 2:
                                success, raw_bbox = update_result[0], update_result[1]
                            elif isinstance(update_result, bool):
                                success = update_result
                                raw_bbox = None
                            else:
                                debug_print(f"[DRAW_OVERLAYS] Unexpected tracker update result: {update_result}")
                                success = False
                                raw_bbox = None
                                
                        except Exception as e:
                            debug_print(f"[DRAW_OVERLAYS] Tracker update failed for object {idx}: {e}")
                            success = False
                            raw_bbox = None
                        
                        time_elapsed = time.monotonic() - tracked_obj['start_time']
                        
                        if success and raw_bbox is not None:
                            # Comprehensive bbox validation
                            if (len(raw_bbox) >= 4 and 
                                all(isinstance(x, (int, float)) and np.isfinite(x) for x in raw_bbox) and
                                raw_bbox[2] > 0 and raw_bbox[3] > 0):  # positive width and height
                                
                                try:
                                    # Use raw bbox directly (simplified approach)
                                    bbox = raw_bbox
                                    
                                    # Check if bbox is valid
                                    if bbox is None:
                                        debug_print(f"[DRAW_OVERLAYS] Invalid bbox for object {idx}")
                                        objects_to_remove.append(idx)
                                        continue
                                    
                                    x, y, w, h = [int(v) for v in bbox]
                                    
                                    # Validate final bbox coordinates against frame bounds
                                    if (x >= 0 and y >= 0 and w > 0 and h > 0 and 
                                        x + w <= frame.shape[1] and y + h <= frame.shape[0] and
                                        y < self.background_edges.shape[0] and x < self.background_edges.shape[1]):
                                    
                                        # Safe ROI extraction and processing
                                        try:
                                            full_edge_mask = None
                                            full_filled_mask = None
                                            roi_frame = frame[y:y+h, x:x+w]
                                            if roi_frame.size > 0:
                                                roi_edges = self.compute_roi_edges(roi_frame)
                                                contours, _ = cv2.findContours(roi_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                                                valid_contours = [c for c in contours if cv2.contourArea(c) > self.roi_min_contour_area]
                                                
                                                if valid_contours:
                                                    # Adjust contours to full frame coordinates
                                                    adjusted_contours = []
                                                    for cnt in valid_contours:
                                                        adjusted_cnt = cnt.copy()
                                                        adjusted_cnt[:, 0, 0] += x  # Adjust x coordinates
                                                        adjusted_cnt[:, 0, 1] += y  # Adjust y coordinates
                                                        adjusted_contours.append(adjusted_cnt)
                                                    
                                                    # Create masks directly on full frame to avoid shape mismatch
                                                    full_edge_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                                                    full_filled_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                                                    
                                                    for cnt in adjusted_contours:
                                                        cv2.drawContours(full_edge_mask, [cnt], -1, 255, self.contour_thickness)
                                                        cv2.drawContours(full_filled_mask, [cnt], -1, 255, -1)

                                                    if self.filled_mask_dilate_iterations > 0:
                                                        cv2.dilate(
                                                            full_filled_mask,
                                                            self.roi_morph_kernel,
                                                            iterations=self.filled_mask_dilate_iterations,
                                                            dst=full_filled_mask,
                                                        )

                                            if full_edge_mask is not None and full_filled_mask is not None:
                                                with self.display_lock:
                                                    if idx < len(self.tracked_regions):
                                                        self.tracked_regions[idx]['edge_mask'] = full_edge_mask
                                                        self.tracked_regions[idx]['filled_mask'] = full_filled_mask
                                                        
                                        except Exception as e:
                                            debug_print(f"[DRAW_OVERLAYS] ROI processing error for object {idx}: {e}")
                                    else:
                                        debug_print(f"[DRAW_OVERLAYS] Invalid bbox coordinates for object {idx}")
                                        objects_to_remove.append(idx)
                                            
                                except Exception as e:
                                    debug_print(f"[DRAW_OVERLAYS] Kalman filter error for object {idx}: {e}")
                                    objects_to_remove.append(idx)
                            else:
                                debug_print(f"[DRAW_OVERLAYS] Invalid bbox for object {idx}: {raw_bbox}")
                                objects_to_remove.append(idx)
                        else:
                            debug_print(f"[DRAW_OVERLAYS] Tracker failed for object {idx} - Success: {success}, Time: {time_elapsed:.2f}s")
                            objects_to_remove.append(idx)
                    
                    except Exception as e:
                        debug_print(f"[DRAW_OVERLAYS] Error processing object {idx}: {e}")
                        objects_to_remove.append(idx)
                
                # Thread-safe object removal with proper cleanup
                if objects_to_remove:
                    with self.display_lock:
                        for idx in sorted(set(objects_to_remove), reverse=True):
                            self._cleanup_tracker_at_index(idx)

            with self.display_lock:
                tracked_regions_copy = list(self.tracked_regions)

            for region in tracked_regions_copy:
                try:
                    edge_mask = region.get('edge_mask')
                    filled_mask = region.get('filled_mask')
                    mask_color = region.get('color', self.default_color)
                    if self.background_removal_in_bbox and filled_mask is not None and np.any(filled_mask > 0):
                        output_frame = frame.copy()
                        output_frame[filled_mask > 0] = mask_color
                        frame = output_frame
                    else:
                        frame = self._apply_mask_overlay(frame, filled_mask, mask_color, self.foreground_mask_alpha)

                    if edge_mask is not None:
                        color_edge = np.zeros_like(frame, dtype=np.float32)
                        color_edge[edge_mask > 0] = mask_color
                        edge_layer = cv2.add(edge_layer, color_edge * self.edge_alpha)
                except Exception as e:
                    debug_print(f"[DRAW_OVERLAYS] Mask overlay error: {e}")
        
            edge_layer = np.clip(edge_layer, 0, 255).astype(np.uint8)
            
            # Apply edge layer to the frame
            combined_frame = cv2.addWeighted(frame, 1.0, edge_layer, 1.0, 0)

            combined_frame = self._apply_heatmap(combined_frame)
            
            # Draw the animated analysis displays
            combined_frame = self.draw_analysis_displays(combined_frame)
            
            # Draw UI elements safely
            try:
                if self.slashing and self.slash_start and self.slash_end:
                    cv2.line(combined_frame, self.slash_start, self.slash_end, (0, 165, 255), 1)
                if self.temp_rectangle and self.temp_rect_timer and time.monotonic() - self.temp_rect_timer <= self.temp_rect_duration:
                    cv2.rectangle(combined_frame, self.temp_rectangle[0], self.temp_rectangle[1], (255, 255, 255), 1)
            except Exception as e:
                debug_print(f"[DRAW_OVERLAYS] UI drawing error: {e}")
            
            elapsed = time.monotonic() - start_ts
            if elapsed > self.overlay_time_budget:
                self.edge_update_interval = min(6, self.edge_update_interval + 1)
                self.gods_eye_simple_mode = True
            elif elapsed < self.overlay_recover_budget:
                self.edge_update_interval = max(3, self.edge_update_interval - 1)

            if RUN_MODE == "ui":
                if start_ts - self.last_heartbeat >= 1.0:
                    self.last_heartbeat = start_ts
                    broadcast_ui({"type": "heartbeat", "t": start_ts})

            return combined_frame
                
        except Exception as e:
            debug_print(f"[DRAW_OVERLAYS] Critical error: {e}")
            import traceback
            traceback.print_exc()
            return frame  # Return original frame on error

    def draw_analysis_displays(self, frame):
        """Draw the active analysis box with tab system"""
        if not hasattr(self, 'analysis_tabs') or not self.analysis_tabs:
            return frame
            
        # Find the active tab
        active_tab = None
        for tab in self.analysis_tabs:
            if tab['is_active']:
                active_tab = tab
                break
                
        if not active_tab:
            return frame
            
        current_time = time.monotonic()
        elapsed_time = current_time - active_tab['start_time']

        # Determine which text to use and format it
        if active_tab['status'] == 'streaming':
            display_text = active_tab.get('display_text', active_tab['text'])
        elif active_tab['status'] == 'processing':
            display_text = active_tab['text']
        else:
            display_text = active_tab.get('final_text', active_tab['text'])

        # Calculate text dimensions for formatted text
        text_width, text_height, text_lines = self.calculate_text_dimensions(display_text)

        # Get position
        x, y = active_tab['position']

        # Animation progress calculation
        animation_elapsed = min(elapsed_time, self.box_animation_duration)
        animation_progress = animation_elapsed / self.box_animation_duration

        # For streaming items, use dynamic dimensions
        if active_tab.get('streaming', False):
            box_width = active_tab.get('current_width', text_width)
            box_height = active_tab.get('current_height', text_height)
        else:
            box_width = text_width
            box_height = text_height

        # Progressive box drawing based on animation phase
        if animation_progress < 0.25:
            # Phase 1: Draw top line only
            progress_in_phase = animation_progress / 0.25
            line_length = int(box_width * progress_in_phase)
            cv2.line(frame, (x, y), (x + line_length, y), active_tab['color'], self.box_line_thickness)

        elif animation_progress < 0.5:
            # Phase 2: Top line complete, draw side lines down
            cv2.line(frame, (x, y), (x + box_width, y), active_tab['color'], self.box_line_thickness)

            progress_in_phase = (animation_progress - 0.25) / 0.25
            side_length = int(box_height * progress_in_phase)

            # Left side
            cv2.line(frame, (x, y), (x, y + side_length), active_tab['color'], self.box_line_thickness)
            # Right side
            cv2.line(frame, (x + box_width, y), (x + box_width, y + side_length), active_tab['color'],
                     self.box_line_thickness)

        elif animation_progress < 0.75:
            # Phase 3: Top and sides complete, draw bottom line
            cv2.line(frame, (x, y), (x + box_width, y), active_tab['color'], self.box_line_thickness)
            cv2.line(frame, (x, y), (x, y + box_height), active_tab['color'], self.box_line_thickness)
            cv2.line(frame, (x + box_width, y), (x + box_width, y + box_height), active_tab['color'],
                     self.box_line_thickness)

            progress_in_phase = (animation_progress - 0.5) / 0.25
            bottom_length = int(box_width * progress_in_phase)
            cv2.line(frame, (x, y + box_height), (x + bottom_length, y + box_height), active_tab['color'],
                     self.box_line_thickness)

        else:
            # Phase 4: Complete box
            cv2.rectangle(frame, (x, y), (x + box_width, y + box_height), active_tab['color'], self.box_line_thickness)

        # Semi-transparent background (only after box is complete or nearly complete)
        if animation_progress > 0.6:
            bg_alpha = (animation_progress - 0.6) / 0.4 * 0.3  # Max 30% opacity
            overlay = frame.copy()
            cv2.rectangle(overlay, (x, y), (x + box_width, y + box_height), (0, 0, 0), -1)
            cv2.addWeighted(overlay, bg_alpha, frame, 1 - bg_alpha, 0, frame)

        # Draw text (only after box animation is mostly complete)
        if animation_progress > 0.7:
            text_alpha = (animation_progress - 0.7) / 0.3  # Fade in text

            # Draw each line of text
            line_y = y + self.analysis_padding + self.analysis_line_height - 5  # Starting Y position

            for line in text_lines:
                if line.strip():  # Only draw non-empty lines
                    text_x = x + self.analysis_padding

                    # Apply alpha to text color
                    if active_tab.get('streaming', False):
                        # Add subtle pulsing to indicate streaming
                        pulse = abs(math.sin(time.monotonic() * 3)) * 0.3 + 0.7
                        text_color_base = self.color_options['text_color']
                        text_color = tuple(int(c * text_alpha * pulse) for c in text_color_base)
                    else:
                        text_color_base = self.color_options['text_color']
                        text_color = tuple(int(c * text_alpha) for c in text_color_base)

                    cv2.putText(frame, line.strip(), (text_x, line_y),
                                self.analysis_font,
                                self.analysis_font_scale,
                                text_color,
                                self.analysis_font_thickness,
                                cv2.LINE_AA)

                    line_y += self.analysis_line_height  # Move to next line position

        return frame

    def push_entity_card(self, object_id: int, text: str, status: str = 'streaming'):
        """Push an entity card update to browser UI via data channel."""
        if RUN_MODE != "ui":
            return
        payload = {
            'type': 'entity_card',
            'id': object_id,
            'status': status,
            'text': self.sanitize_text_for_opencv(text) if text else ''
        }
        try:
            broadcast_ui(payload)
        except Exception:
            pass

    def create_combined_analysis_image(self, roi_image, context_image):
        """Create a single combined image with ROI and context side-by-side for maximum LLM efficiency"""
        try:
            # Validate inputs
            if roi_image is None or context_image is None:
                debug_print("[COMBINED_IMAGE] Error: Missing ROI or context image")
                return None
                
            if roi_image.size == 0 or context_image.size == 0:
                debug_print("[COMBINED_IMAGE] Error: Empty image provided")
                return None
            
            # Resize images for optimal LLM processing
            roi_resized = roi_image.copy()
            context_resized = context_image.copy()
            
            # Resize ROI to target size (detailed view)
            roi_height, roi_width = roi_resized.shape[:2]
            if max(roi_height, roi_width) > self.roi_max_size:
                scale = self.roi_max_size / max(roi_height, roi_width)
                new_roi_width = int(roi_width * scale)
                new_roi_height = int(roi_height * scale)
                roi_resized = cv2.resize(roi_resized, (new_roi_width, new_roi_height), interpolation=cv2.INTER_AREA)
                debug_print(f"[COMBINED_IMAGE] ROI resized from {roi_width}x{roi_height} to {new_roi_width}x{new_roi_height}")
            
            # Resize context to target size  
            context_height, context_width = context_resized.shape[:2]
            if max(context_height, context_width) > self.gods_eye_max_size:
                scale = self.gods_eye_max_size / max(context_height, context_width)
                new_context_width = int(context_width * scale)
                new_context_height = int(context_height * scale)
                context_resized = cv2.resize(context_resized, (new_context_width, new_context_height), interpolation=cv2.INTER_AREA)
                debug_print(f"[COMBINED_IMAGE] Context resized from {context_width}x{context_height} to {new_context_width}x{new_context_height}")
            
            # Calculate dimensions for side-by-side layout
            roi_h, roi_w = roi_resized.shape[:2]
            context_h, context_w = context_resized.shape[:2]
            
            # Use the taller image's height and combine widths
            combined_height = max(roi_h, context_h)
            combined_width = roi_w + context_w + 10  # 10px gap between images
            
            # Create combined image
            combined_image = np.zeros((combined_height, combined_width, 3), dtype=np.uint8)
            
            # Place ROI on the left
            roi_y_offset = (combined_height - roi_h) // 2
            combined_image[roi_y_offset:roi_y_offset + roi_h, 0:roi_w] = roi_resized
            
            # Place context on the right with gap
            context_x_offset = roi_w + 10
            context_y_offset = (combined_height - context_h) // 2
            combined_image[context_y_offset:context_y_offset + context_h, context_x_offset:context_x_offset + context_w] = context_resized
            
            # Add visual separators and labels
            # Draw vertical line separator
            separator_x = roi_w + 5
            cv2.line(combined_image, (separator_x, 0), (separator_x, combined_height), (255, 255, 255), 2)
            
            # Add labels
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            font_thickness = 2
            
            # "DETAIL" label on ROI side
            cv2.putText(combined_image, "DETAIL", (10, 25), font, font_scale, (0, 255, 0), font_thickness)
            
            # "CONTEXT" label on context side  
            cv2.putText(combined_image, "CONTEXT", (context_x_offset + 10, 25), font, font_scale, (0, 255, 255), font_thickness)
            
            debug_print(f"[COMBINED_IMAGE] Created combined image: {combined_width}x{combined_height} (ROI: {roi_w}x{roi_h}, Context: {context_w}x{context_h})")
            return combined_image
            
        except Exception as e:
            debug_print(f"[COMBINED_IMAGE] Error creating combined image: {e}")
            # Fallback to just ROI if available
            if roi_image is not None and roi_image.size > 0:
                return roi_image
            elif context_image is not None and context_image.size > 0:
                return context_image
            else:
                return np.zeros((400, 600, 3), dtype=np.uint8)  # Empty fallback

app = FastAPI()
pcs = set()
video_tracks = set()
ui_channels = set()
ui_channels_lock = threading.RLock()

def broadcast_ui(payload: dict):
    """Broadcast a UI event to all connected data channels."""
    try:
        msg = json.dumps(payload)
    except Exception:
        return
    with ui_channels_lock:
        stale = []
        for ch in list(ui_channels):
            try:
                ch.send(msg)
            except Exception:
                stale.append(ch)
        for ch in stale:
            ui_channels.discard(ch)

def _coerce_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "y", "on")
    return False

def apply_settings_update(settings, video_track=None):
    if not isinstance(settings, dict):
        return
    global DEBUG_MODE, ENHANCED_DEBUG, VERBOSE_FRAME_LOGGING
    global SHOW_OBJECT_TITLES, SHOW_CONFIDENCE_SCORES, SHOW_DETECTION_ARROWS
    global SHOW_CORNER_MARKERS, ENABLE_BASE_EDGE_BACKGROUND
    global ENABLE_POSE_ESTIMATION, USE_PRIORITY_CLASSES, USE_LIGHT_OBJECT_TITLES
    global ENABLE_HEATMAP_MODE

    bool_settings = {
        "DEBUG_MODE": "DEBUG_MODE",
        "ENHANCED_DEBUG": "ENHANCED_DEBUG",
        "VERBOSE_FRAME_LOGGING": "VERBOSE_FRAME_LOGGING",
        "SHOW_OBJECT_TITLES": "SHOW_OBJECT_TITLES",
        "SHOW_CONFIDENCE_SCORES": "SHOW_CONFIDENCE_SCORES",
        "SHOW_DETECTION_ARROWS": "SHOW_DETECTION_ARROWS",
        "SHOW_CORNER_MARKERS": "SHOW_CORNER_MARKERS",
        "ENABLE_BASE_EDGE_BACKGROUND": "ENABLE_BASE_EDGE_BACKGROUND",
        "USE_LIGHT_OBJECT_TITLES": "USE_LIGHT_OBJECT_TITLES",
        "ENABLE_POSE_ESTIMATION": "ENABLE_POSE_ESTIMATION",
        "USE_PRIORITY_CLASSES": "USE_PRIORITY_CLASSES",
        "ENABLE_HEATMAP_MODE": "ENABLE_HEATMAP_MODE",
    }

    for key, attr in bool_settings.items():
        if key in settings:
            globals()[attr] = _coerce_bool(settings[key])

    if video_track is not None:
        try:
            video_track.verbose_frame_logging = VERBOSE_FRAME_LOGGING
        except Exception:
            pass
        detector = getattr(video_track, "detector", None)
        if detector is not None:
            try:
                detector.use_priority_classes = USE_PRIORITY_CLASSES
            except Exception:
                pass
            try:
                detector.set_pose_estimation_enabled(ENABLE_POSE_ESTIMATION)
            except Exception:
                pass
            try:
                detector.segmentation_engine.heatmap_enabled = ENABLE_HEATMAP_MODE
            except Exception:
                pass
 
# ─── Class → Category Mapping ────────────────────────────────────────────────
CLASS_CATEGORIES = {"person": "Humans", "cat": "Pets", "dog": "Pets", "bird": "Wild Animals", "horse": "Wild Animals", "sheep": "Wild Animals", "cow": "Wild Animals", "elephant": "Wild Animals", "bear": "Wild Animals", "zebra": "Wild Animals", "giraffe": "Wild Animals", "potted plant": "Plants", "bicycle": "Vehicles", "car": "Vehicles", "motorcycle": "Vehicles", "airplane": "Vehicles", "bus": "Vehicles", "train": "Vehicles", "truck": "Vehicles", "boat": "Vehicles", "tv": "Technology", "laptop": "Technology", "mouse": "Technology", "remote": "Technology", "keyboard": "Technology", "cell phone": "Technology", "microwave": "Appliances", "oven": "Appliances", "toaster": "Appliances", "refrigerator": "Appliances", "sink": "Plumbing Fixtures", "toilet": "Plumbing Fixtures", "bench": "Furniture", "chair": "Furniture", "couch": "Furniture", "bed": "Furniture", "dining table": "Furniture", "book": "Furniture", "clock": "Furniture", "vase": "Furniture", "bottle": "Kitchenware", "wine glass": "Kitchenware", "cup": "Kitchenware", "fork": "Kitchenware", "knife": "Kitchenware", "spoon": "Kitchenware", "bowl": "Kitchenware", "banana": "Food", "apple": "Food", "sandwich": "Food", "orange": "Food", "broccoli": "Food", "carrot": "Food", "hot dog": "Food", "pizza": "Food", "donut": "Food", "cake": "Food", "frisbee": "Sports", "skis": "Sports", "snowboard": "Sports", "sports ball": "Sports", "kite": "Sports", "baseball bat": "Sports", "baseball glove": "Sports", "skateboard": "Sports", "surfboard": "Sports", "tennis racket": "Sports", "backpack": "Accessories", "umbrella": "Accessories", "handbag": "Accessories", "tie": "Accessories", "suitcase": "Accessories", "scissors": "Accessories", "teddy bear": "Accessories", "hair drier": "Accessories", "toothbrush": "Accessories", "traffic light": "Urban Infrastructure (Electronic)", "parking meter": "Urban Infrastructure (Electronic)", "fire hydrant": "Urban Infrastructure (Static)", "stop sign": "Urban Infrastructure (Static)"}

TECH_CATEGORIES = {"Vehicles", "Technology", "Appliances", "Urban Infrastructure (Electronic)"}
def get_category(name: str) -> str: return CLASS_CATEGORIES.get(name, "Unknown")
def get_type(name: str) -> str: return "Tech" if get_category(name) in TECH_CATEGORIES or name == "hair drier" else "Organic"

def _print_runtime_instructions():
    debug_print("\n=== Interactive Features (moved from HTML to terminal) ===")
    debug_print("- Click on objects to select/deselect them (they'll turn red when selected)")
    debug_print("- Click and drag to draw rectangles for segmentation (OG style)")
    debug_print("- First click starts rectangle; move mouse to see preview; release to complete")
    debug_print("- Rectangle drawing creates instant segmentation with context windows and overlays")
    debug_print("- Human Pose Estimation: Diamond markers/threat detection on detected humans")
    debug_print("- OG Segmentation: Click empty areas to trigger segmentation with context windows")
    debug_print("- Modern Features: Debug image saving, God's Eye view, visual click feedback, analysis")
    debug_print("- Press '3' to switch between live camera and video file")
    debug_print("- Automatic Analysis: Segmentation and analysis run automatically when you draw rectangles\n")


class BackgroundObjectDetector:
    def __init__(self, model_path=MODEL_PATH, device=DEVICE, conf=CONF, iou=IOU, priority_classes=PRIORITY_CLASSES, priority_conf=PRIORITY_CONF, use_priority_classes=USE_PRIORITY_CLASSES):
        self.conf, self.priority_conf, self.priority_classes, self.use_priority_classes = conf, priority_conf, priority_classes, use_priority_classes
        self.iou, self.img_size, self.max_det = iou, IMG_SIZE, MAX_DET
        self.device = self._get_best_device() if device == "auto" else device
        debug_print(f"[OBJECT_DETECTION] Loading YOLO model from {model_path}")
        self.model = YOLO(model_path)
        self.model.to(self.device)
        self.class_labels, self.available_classes = self.model.names, list(self.model.names.values())
        self.enabled_classes = set(self.available_classes)
        self.filter_lock, self.latest_detections, self.detection_lock = threading.Lock(), [], threading.Lock()
        self.selected_track_ids, self.selection_lock = set(), threading.Lock()
        self.frame_queue, self.stop_event = deque(maxlen=1), threading.Event()
        self.thread = threading.Thread(target=self._run_detection_loop, daemon=True)
        self.segmentation_engine = HighlightTracker()
        self.detection_times, self.detection_fps = [], 0
        self.pose_estimator = None
        self.pose_estimation_enabled = ENABLE_POSE_ESTIMATION
        self.pose_init_failed = False
        self.frame_counter = 0
        self.inference_interval = 3  # Run detection every 3rd frame to lower CPU
        if ENABLE_POSE_ESTIMATION:
            try:
                self.pose_estimator = PoseEstimator()
                debug_print("[POSE_ESTIMATION] Initialized successfully")
            except Exception as e:
                debug_print(f"[POSE_ESTIMATION] Failed to initialize: {e}")
                self.pose_init_failed = True
        self.heatmap_enabled = ENABLE_HEATMAP_MODE
        debug_print(f"[OBJECT_DETECTION] Initialized on device: {self.device}")
        
        # Note: Double-click detection is handled by the segmentation_engine (HighlightTracker)

    def start(self):
        self.thread.start()
        debug_print("[OBJECT_DETECTION] Detection thread started.")

    def stop(self):
        debug_print("[OBJECT_DETECTION] Stopping detection thread...")
        self.stop_event.set()
        self.thread.join()
        if self.pose_estimator: self.pose_estimator.cleanup()
        if self.segmentation_engine: self.segmentation_engine.cleanup()
        debug_print("[OBJECT_DETECTION] Detection thread stopped.")

    def set_pose_estimation_enabled(self, enabled):
        enabled = bool(enabled)
        self.pose_estimation_enabled = enabled
        if enabled and self.pose_estimator is None and not self.pose_init_failed:
            try:
                self.pose_estimator = PoseEstimator()
                self.pose_init_failed = False
                debug_print("[POSE_ESTIMATION] Enabled via settings")
            except Exception as e:
                debug_print(f"[POSE_ESTIMATION] Enable failed: {e}")
                self.pose_estimator = None
                self.pose_init_failed = True
        elif not enabled and self.pose_estimator is not None:
            try:
                self.pose_estimator.cleanup()
            except Exception:
                pass
            self.pose_estimator = None
            debug_print("[POSE_ESTIMATION] Disabled via settings")

    def _get_best_device(self):
        try:
            import torch
            if torch.cuda.is_available(): return "cuda"
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(): return "mps"
        except ImportError: pass
        return "cpu"

    def handle_click(self, x, y):
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
        
        # Delegate to segmentation engine for general click handling (including double-click detection)
        if not object_clicked and self.segmentation_engine:
            # Check for double-click and handle accordingly
            current_time = time.monotonic()
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
                    self.segmentation_engine.last_click_pos = None  # Reset to prevent triple-click
                    return
            
            # Update last click info for double-click detection
            self.segmentation_engine.last_click_time = current_time
            self.segmentation_engine.last_click_pos = click_pos

    def handle_mouse_down(self, x, y):
        if self.segmentation_engine:
            self.segmentation_engine.handle_mouse_down(x, y)
    
    def handle_mouse_move(self, x, y):
        if self.segmentation_engine:
            self.segmentation_engine.handle_mouse_move(x, y)
    
    def handle_mouse_up(self, x, y):
        if self.segmentation_engine:
            self.segmentation_engine.handle_mouse_up(x, y)
        
        # Also trigger the click handler to check for single/double clicks
        self.handle_click(x, y)

    def update_frame(self, frame):
        if self.segmentation_engine:
            self.segmentation_engine.update_frame(frame)
        self.frame_queue.append(frame)

    # ------------------------------------------------------------------
    # Hairline horizontal edge background (from testdnn.py)
    # Always-on subtle edge overlay applied as a base layer each frame
    # ------------------------------------------------------------------
    def _apply_hairline_edge_background(self, frame: np.ndarray) -> np.ndarray:
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            sobel_x = cv2.Sobel(gray, cv2.CV_64F, dx=1, dy=0, ksize=3)
            sobel_x = np.abs(sobel_x)
            sobel_x = np.uint8(np.clip(sobel_x, 0, 255))

            # Base thin horizontal edges
            _, thresh = cv2.threshold(sobel_x, 80, 255, cv2.THRESH_BINARY)
            thin_edges = thresh.copy()
            scan_mask = np.zeros_like(thin_edges)
            scan_mask[::4, :] = 255
            scan_mask = cv2.dilate(scan_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 2)))
            masked_edges = cv2.bitwise_and(thin_edges, scan_mask)

            # Ultra-thin variant (kept for parity with testdnn; not blended by default)
            _ , thresh_blue = cv2.threshold(sobel_x, 120, 255, cv2.THRESH_BINARY)
            thin_edges_blue = thresh_blue.copy()
            scan_mask_blue = np.zeros_like(thin_edges_blue)
            scan_mask_blue[::6, :] = 255
            scan_mask_blue = cv2.dilate(scan_mask_blue, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1)))
            _masked_edges_blue = cv2.bitwise_and(thin_edges_blue, scan_mask_blue)

            # Build red overlay (base) like testdnn
            red_overlay = np.zeros_like(frame)
            red_overlay[masked_edges > 0] = (0, 0, 255)

            # Subtle blend onto the frame
            base = cv2.addWeighted(frame, 0.85, red_overlay, 0.15, 0)
            return base
        except Exception:
            return frame

    def _run_detection_loop(self):
        while not self.stop_event.is_set():
            if not self.frame_queue:
                time.sleep(0.01)
                continue
            frame = self.frame_queue.popleft()
            self.frame_counter += 1
            if self.frame_counter % self.inference_interval == 0:
                self._process_frame_background(frame)

    def _process_frame_background(self, frame):
        if frame is None: return []
        start = time.monotonic()
        results = self.model.track(source=frame, device=self.device, imgsz=self.img_size, conf=self.conf, iou=self.iou, max_det=self.max_det, verbose=False, stream=False, persist=True, tracker="bytetrack.yaml")
        detections = []
        for r in results:
            if r.boxes.id is None: continue
            for box, conf_score, cls, track_id in zip(r.boxes.xyxy, r.boxes.conf, r.boxes.cls, r.boxes.id):
                name = r.names[int(cls)]
                if self.use_priority_classes and name not in self.priority_classes:
                    continue
                threshold = self.priority_conf if self.use_priority_classes and name in self.priority_classes else self.conf
                if conf_score < threshold: continue
                x1, y1, x2, y2 = map(int, box)
                detections.append({'name': name, 'display_name': "plant" if name == "potted plant" else name, 'category': get_category(name), 'type': get_type(name), 'confidence': float(conf_score), 'bbox': [x1, y1, x2, y2], 'track_id': int(track_id)})
        with self.detection_lock: self.latest_detections = detections
        dt = time.monotonic() - start
        self.detection_times.append(dt)
        if len(self.detection_times) > 30: self.detection_times.pop(0)
        self.detection_fps = len(self.detection_times) / sum(self.detection_times) if sum(self.detection_times)>0 else 0
        if dt > 0.08:
            self.inference_interval = min(6, self.inference_interval + 1)
        elif dt < 0.03:
            self.inference_interval = max(2, self.inference_interval - 1)

    def draw_detections_on_frame(self, frame):
        out = frame.copy()
        # Always-on hairline edge background (from testdnn.py)
        if ENABLE_BASE_EDGE_BACKGROUND:
            out = self._apply_hairline_edge_background(out)
        
        if self.pose_estimation_enabled and self.pose_estimator is None and not self.pose_init_failed:
            try:
                self.pose_estimator = PoseEstimator()
                self.pose_init_failed = False
                debug_print("[POSE_ESTIMATION] Enabled via lazy init")
            except Exception as e:
                debug_print(f"[POSE_ESTIMATION] Lazy init failed: {e}")
                self.pose_init_failed = True
        
        with self.detection_lock: dets = list(self.latest_detections)

        if self.segmentation_engine:
            try:
                self.segmentation_engine.update_heatmap_from_detections(out, dets)
            except Exception:
                pass
            out = self.segmentation_engine.draw_overlays(out)
        
        for d in dets:
            if d['name'] not in self.enabled_classes: continue
            x1, y1, x2, y2 = d['bbox']
            with self.selection_lock: is_selected = d.get('track_id') in self.selected_track_ids
            color = (255, 0, 0) if is_selected else (0, 255, 0)
            
            if self.pose_estimation_enabled and d['name'] == 'person' and self.pose_estimator and max(x2 - x1, y2 - y1) >= POSE_MIN_BBOX_SIZE:
                try: out = self.pose_estimator.process_pose_in_bbox(out, x1, y1, x2, y2)
                except Exception as e: debug_print(f"[POSE_ESTIMATION] Error: {e}")

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

            center_x, arrow_y = (x1 + x2) // 2, y1 - 20
            if SHOW_DETECTION_ARROWS:
                asz, ath = 10, 2
                arrow_overlay = out.copy()
                cv2.line(arrow_overlay, (center_x - asz//2, arrow_y), (center_x, arrow_y + asz), color, ath, cv2.LINE_AA)
                cv2.line(arrow_overlay, (center_x, arrow_y + asz), (center_x + asz//2, arrow_y), color, ath, cv2.LINE_AA)
                cv2.addWeighted(arrow_overlay, ARROW_TRANSPARENCY, out, 1 - ARROW_TRANSPARENCY, 0, out)
            
            if SHOW_OBJECT_TITLES or SHOW_CONFIDENCE_SCORES:
                parts = []
                if SHOW_OBJECT_TITLES: parts.append(f"{d['display_name']} ({d['category']}, {d['type']})")
                if SHOW_CONFIDENCE_SCORES: parts.append(f"{d['confidence']:.2f}")
                label = " ".join(parts)
                fs, tt = 0.4, 1
                (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, tt)
                if USE_LIGHT_OBJECT_TITLES:
                    line_y = max(0, arrow_y - 10)
                    marker = ">"
                    marker_size, _ = cv2.getTextSize(marker, cv2.FONT_HERSHEY_SIMPLEX, fs, tt)
                    line_len = marker_size[0] + 6 + tw
                    line_x1 = max(0, center_x - line_len // 2)
                    line_x2 = line_x1 + line_len
                    marker_x = line_x1
                    marker_y = line_y + th // 2
                    cv2.putText(out, marker, (marker_x, marker_y), cv2.FONT_HERSHEY_SIMPLEX, fs, color, tt, cv2.LINE_AA)
                    start_x = marker_x + marker_size[0] + 4
                    text_x = center_x - tw // 2
                    text_y = max(0, line_y - 6)
                    text_x = center_x - tw // 2
                    text_y = max(0, line_y - 6)
                    dash_len = 10
                    dash_gap = 6
                    x = start_x
                    while x < line_x2:
                        seg_start = x
                        seg_end = min(x + dash_len, line_x2)
                        cv2.line(out, (seg_start, line_y), (seg_end, line_y), color, 1, cv2.LINE_AA)
                        x += dash_len + dash_gap
                    cv2.putText(out, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, fs, color, tt, cv2.LINE_AA)
                else:
                    lx, ly = center_x - tw // 2, arrow_y + 10 + th + 5
                    cv2.rectangle(out, (lx - 2, ly - th - 2), (lx + tw + 2, ly + 2), color, -1)
                    cv2.putText(out, label, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, fs, (0,0,0), tt, cv2.LINE_AA)
        return out

class OpenCVVideoTrack(VideoStreamTrack):
    """
    A VideoStreamTrack that pulls frames from OpenCV (webcam/video) with AI object detection.
    Now includes comprehensive screen-aware scaling for optimal quality and full-screen coverage.
    """
    def __init__(self, device_index=0, client_screen_info=None):
        super().__init__()
        self.frame_count = 0
        self.last_frame_time = time.monotonic()
        
        # Store comprehensive client screen information
        self.client_screen_info = client_screen_info or {}
        self._initialize_screen_dimensions()
        
        # Video source management
        self.use_camera = True  # True for camera, False for video file
        self.video_file_path = os.path.join(VIDEOS_DIR, "frenchpeoplewalking.mp4")
        self.video_cap = None  # For video file playback
        self.current_video_frame = 0
        self.video_total_frames = 0
        
        # Initialize object detector
        self.detector = BackgroundObjectDetector()
        self.detector.start()
        
        # Initialize camera with optimal settings for screen dimensions
        self._initialize_camera(device_index)
        
        # Initialize video file capability
        self._initialize_video_file()
        
        # Performance tracking
        self.fps_tracker = []
        self.last_fps_time = time.monotonic()
        
        # Frame debug settings - controlled by global VERBOSE_FRAME_LOGGING constant
        self.verbose_frame_logging = VERBOSE_FRAME_LOGGING
        
        debug_print(f"[VIDEO_TRACK] Screen-aware dimensions: {self.target_width}x{self.target_height}")
        debug_print(f"[VIDEO_TRACK] Quality level: {self.quality_level}, Aspect ratio: {self.aspect_ratio:.3f}")
        debug_print(f"[VIDEO_TRACK] Device: {self.device_type}, DPI: {self.dpi}")
    
    def _initialize_screen_dimensions(self):
        """Initialize screen dimensions from comprehensive client information"""
        # Default dimensions (fallback)
        default_width = 1920
        default_height = 1080
        
        # Extract comprehensive screen information
        self.screen_width = self.client_screen_info.get('screen_width', default_width)
        self.screen_height = self.client_screen_info.get('screen_height', default_height)
        self.available_width = self.client_screen_info.get('available_width', self.screen_width)
        self.available_height = self.client_screen_info.get('available_height', self.screen_height)
        self.ViewPort_width = self.client_screen_info.get('ViewPort_width', self.available_width)
        self.ViewPort_height = self.client_screen_info.get('ViewPort_height', self.available_height)
        self.dpi = self.client_screen_info.get('dpi', 96)
        # Prefer explicit device_pixel_ratio; fallback to dpi field if provided by client
        self.device_pixel_ratio = self.client_screen_info.get('device_pixel_ratio', self.client_screen_info.get('dpi', 1.0))
        
        # Device type information
        self.is_mobile = self.client_screen_info.get('is_mobile', False)
        self.is_tablet = self.client_screen_info.get('is_tablet', False)
        self.is_desktop = self.client_screen_info.get('is_desktop', True)
        self.device_type = 'mobile' if self.is_mobile else 'tablet' if self.is_tablet else 'desktop'
        
        # Calculate aspect ratio
        self.aspect_ratio = self.screen_width / self.screen_height
        
        # Determine optimal video dimensions based on screen and device type
        self._calculate_optimal_dimensions()
        
        debug_print(f"[SCREEN_INFO] Screen: {self.screen_width}x{self.screen_height}")
        debug_print(f"[SCREEN_INFO] ViewPort: {self.ViewPort_width}x{self.ViewPort_height}")
        debug_print(f"[SCREEN_INFO] DPI: {self.dpi}, Pixel Ratio: {self.device_pixel_ratio}")
        debug_print(f"[SCREEN_INFO] Device: {self.device_type}")
    
    def _calculate_optimal_dimensions(self):
        """Calculate optimal video dimensions based on screen info and device capabilities"""
        # Base dimensions on ViewPort for full-screen coverage
        base_width = self.ViewPort_width
        base_height = self.ViewPort_height
        
        # Calculate total pixels (account for device pixel ratio to improve quality on HiDPI)
        dpr = max(1.0, float(self.device_pixel_ratio))
        total_pixels = base_width * base_height * dpr
        
        # Determine quality tier and adjust dimensions
        if total_pixels >= 3840 * 2160:  # 4K+
            self.quality_level = "4K"
            max_width, max_height = 3840, 2160
        elif total_pixels >= 2560 * 1440:  # 2K/QHD
            self.quality_level = "2K"
            max_width, max_height = 2560, 1440
        elif total_pixels >= 1920 * 1080:  # 1080p
            self.quality_level = "1080p"
            max_width, max_height = 1920, 1080
        elif total_pixels >= 1280 * 720:  # 720p
            self.quality_level = "720p"
            max_width, max_height = 1280, 720
        else:  # 480p
            self.quality_level = "480p"
            max_width, max_height = 854, 480
        
        # Apply device-specific optimizations
        if self.is_mobile:
            # Mobile devices - optimize for battery and bandwidth
            max_width = min(max_width, 1920)
            max_height = min(max_height, 1080)
        elif self.is_tablet:
            # Tablets - balance quality and performance
            max_width = min(max_width, 2560)
            max_height = min(max_height, 1440)
        # Desktop can handle full quality
        
        # Set target dimensions (don't exceed screen size or quality limits)
        self.target_width = min(base_width, max_width)
        self.target_height = min(base_height, max_height)
        
        # Apply device pixel ratio for high-DPI displays
        if self.device_pixel_ratio > 1.0 and not self.is_mobile:
            # Apply pixel ratio for desktop/tablet high-DPI displays
            self.target_width = min(int(self.target_width * self.device_pixel_ratio), max_width)
            self.target_height = min(int(self.target_height * self.device_pixel_ratio), max_height)
        
        # Ensure dimensions are even numbers (required for some codecs)
        self.target_width = self.target_width - (self.target_width % 2)
        self.target_height = self.target_height - (self.target_height % 2)
        
        # Ensure minimum dimensions
        self.target_width = max(self.target_width, 640)
        self.target_height = max(self.target_height, 480)
    
    def _initialize_camera(self, device_index):
        """Initialize camera with optimal settings based on screen dimensions"""
        # Try to open camera with different backends
        backends = [cv2.CAP_DSHOW, cv2.CAP_V4L2, cv2.CAP_ANY]
        self.cap = None
        
        for backend in backends:
            debug_print(f"Trying to open camera {device_index} with backend {backend}")
            self.cap = cv2.VideoCapture(device_index, backend)
            if self.cap.isOpened():
                debug_print(f"Successfully opened camera {device_index} with backend {backend}")
                break
        
        if self.cap is None or not self.cap.isOpened():
            debug_print(f"Failed to open camera {device_index}, trying alternative indices")
            for alt_index in [1, 2, 3]:
                for backend in backends:
                    self.cap = cv2.VideoCapture(alt_index, backend)
                    if self.cap.isOpened():
                        debug_print(f"Successfully opened camera {alt_index} with backend {backend}")
                        break
                if self.cap.isOpened():
                    break
        
        if self.cap is None or not self.cap.isOpened():
            debug_print("ERROR: Could not open any camera, will use test pattern")
            self.cap = None
            return
        
        # Set optimal camera properties based on target dimensions
        self._configure_camera_properties()
        
        # Test read a frame
        ret, test_frame = self.cap.read()
        if ret:
            debug_print(f"Camera test successful: {test_frame.shape}")
        else:
            debug_print("WARNING: Camera opened but cannot read frames")
    
    def _configure_camera_properties(self):
        """Configure camera properties for optimal quality"""
        if not self.cap:
            return
        
        # Request capture resolution that matches our computed target to minimize resampling
        high_res_width = int(self.target_width)
        high_res_height = int(self.target_height)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, high_res_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, high_res_height)
        
        # Set other camera properties for quality
        # Try 60fps first for smoother motion, fall back to ~30 if not supported
        try:
            self.cap.set(cv2.CAP_PROP_FPS, 60)
        except Exception:
            pass
        try:
            # MJPG often reduces CPU load on webcams
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        except Exception as e:
            debug_print(f"[CAMERA] Could not set MJPG format: {e}")
        
        # Get actual resolution from the camera
        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)

        # If the camera ignored 60fps request and returned too low, try setting 30fps explicitly
        try:
            if not actual_fps or actual_fps < 45:
                self.cap.set(cv2.CAP_PROP_FPS, 30)
                actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        except Exception:
            pass
        
        debug_print(f"[CAMERA] Requested {high_res_width}x{high_res_height}, Got {actual_width}x{actual_height} @ {actual_fps}fps")
        debug_print(f"[CAMERA] Target for ViewPort: {self.target_width}x{self.target_height}")

    def _initialize_video_file(self):
        """Initialize video file for playback"""
        try:
            # Check if the file exists first
            if not os.path.exists(self.video_file_path):
                try:
                    import glob
                    candidates = sorted(glob.glob(os.path.join(VIDEOS_DIR, "*.mp4")))
                    if candidates:
                        self.video_file_path = candidates[0]
                        debug_print(f"[VIDEO_FILE] Using fallback video: {self.video_file_path}")
                    else:
                        debug_print(f"ERROR: Video file does not exist: {self.video_file_path}")
                        self.video_cap = None
                        return
                except Exception:
                    debug_print(f"ERROR: Video file does not exist: {self.video_file_path}")
                    self.video_cap = None
                    return

            self.video_cap = cv2.VideoCapture(self.video_file_path)
            if self.video_cap.isOpened():
                self.video_total_frames = int(self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
                video_fps = self.video_cap.get(cv2.CAP_PROP_FPS)
                video_width = int(self.video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                video_height = int(self.video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                debug_print(f"Video file initialized: {self.video_file_path}")
                debug_print(f"Video properties: {video_width}x{video_height} @ {video_fps}fps, {self.video_total_frames} frames")
            else:
                debug_print(f"WARNING: Could not open video file: {self.video_file_path}")
                self.video_cap = None
        except Exception as e:
            debug_print(f"ERROR: Failed to initialize video file: {e}")
            self.video_cap = None

    def switch_video_source(self):
        """Switch between camera and video file"""
        self.use_camera = not self.use_camera
        
        if self.use_camera:
            source_name = "Camera"
            debug_print(f"[VIDEO_SOURCE] Switched to: {source_name}")
        else:
            if (self.video_cap is None) or (not self.video_cap.isOpened()):
                self._initialize_video_file()
            if self.video_cap and self.video_cap.isOpened():
                source_name = "Video File"
                # Reset video file to beginning when switching to it
                self.current_video_frame = 0
                self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                debug_print(f"[VIDEO_SOURCE] Switched to: {source_name} - {self.video_file_path}")
                debug_print(f"[VIDEO_SOURCE] Video reset to frame 0 of {self.video_total_frames}")
            else:
                # Video file not available, switch back to camera
                self.use_camera = True
                source_name = "Camera (Video file not available)"
                debug_print(f"[VIDEO_SOURCE] Video file not available, staying on: {source_name}")
        
        return source_name

    async def recv(self):
        self.frame_count += 1
        current_time = time.monotonic()
        
        # Generate timestamp for WebRTC
        pts, time_base = await self.next_timestamp()
        
        # Get frame from appropriate source
        if self.use_camera:
            # Camera mode
            if self.cap is not None and self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    debug_print(f"Warning: Failed to read frame {self.frame_count} from camera")
                    frame = self.create_test_frame("Camera Read Failed")
                else:
                    # Log frame info periodically (only if verbose logging enabled)
                    if self.verbose_frame_logging and self.frame_count % 60 == 0:  # Every 2 seconds at 30fps
                        debug_print(f"Camera Frame {self.frame_count}: {frame.shape}, dtype: {frame.dtype}")
                        debug_print(f"Frame stats - min: {frame.min()}, max: {frame.max()}, mean: {frame.mean():.2f}")
            else:
                frame = self.create_test_frame("No Camera Available")
        else:
            # Video file mode
            if self.video_cap is not None and self.video_cap.isOpened():
                ret, frame = self.video_cap.read()
                if not ret:
                    # End of video, loop back to beginning
                    debug_print(f"[VIDEO_FILE] End of video reached, looping back to beginning")
                    self.current_video_frame = 0
                    self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = self.video_cap.read()
                    if not ret:
                        debug_print(f"[VIDEO_FILE] ERROR: Cannot read from video file even after reset")
                        frame = self.create_test_frame("Video File Read Failed")
                    else:
                        debug_print(f"[VIDEO_FILE] Successfully looped back to beginning")
                
                if ret:
                    self.current_video_frame += 1
                    # Log frame info periodically (only if verbose logging enabled)
                    if self.verbose_frame_logging and self.frame_count % 120 == 0:  # Reduce logging frequency
                        progress = (self.current_video_frame / self.video_total_frames) * 100 if self.video_total_frames > 0 else 0
                        debug_print(f"Video Frame {self.frame_count} (file frame {self.current_video_frame}/{self.video_total_frames}, {progress:.1f}%): {frame.shape}")
            else:
                frame = self.create_test_frame("Video File Not Available")
        
        # Resize frame to cover client dimensions, cropping excess
        cam_h, cam_w = frame.shape[:2]
        target_w, target_h = self.target_width, self.target_height

        if cam_w != target_w or cam_h != target_h:
            cam_aspect = cam_w / cam_h
            target_aspect = target_w / target_h

            # Choose interpolation for best quality
            def choose_interp(new_w, new_h):
                if new_w < cam_w or new_h < cam_h:
                    return cv2.INTER_AREA  # best for downscale
                else:
                    return cv2.INTER_CUBIC  # higher quality upscale

            if abs(cam_aspect - target_aspect) > 0.01:
                # Aspect ratios differ, need to crop to cover
                if cam_aspect > target_aspect:
                    # Camera is wider than target (e.g., 16:9 cam, 4:3 target). Fit to height, crop width.
                    new_h = target_h
                    new_w = int(new_h * cam_aspect)
                    resized = cv2.resize(frame, (new_w, new_h), interpolation=choose_interp(new_w, new_h))
                    x_offset = (new_w - target_w) // 2
                    frame = resized[:, x_offset:x_offset + target_w]
                else:
                    # Camera is narrower than target (e.g., 4:3 cam, 16:9 target). Fit to width, crop height.
                    new_w = target_w
                    new_h = int(new_w / cam_aspect)
                    resized = cv2.resize(frame, (new_w, new_h), interpolation=choose_interp(new_w, new_h))
                    y_offset = (new_h - target_h) // 2
                    frame = resized[y_offset:y_offset + target_h, :]
            else:
                # Aspect ratios are the same, just resize.
                frame = cv2.resize(frame, (target_w, target_h), interpolation=choose_interp(target_w, target_h))
        
        # Apply object detection to the frame
        if self.detector:
            self.detector.update_frame(frame.copy())
            frame = self.detector.draw_detections_on_frame(frame)
        
        # Convert BGR to RGB (OpenCV uses BGR, WebRTC expects RGB)
        if len(frame.shape) == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Create VideoFrame
        try:
            video_frame = VideoFrame.from_ndarray(frame, format="rgb24")
            video_frame.pts = pts
            video_frame.time_base = time_base
            
            # Log frame creation info periodically (only if verbose logging enabled)
            if self.verbose_frame_logging and self.frame_count % 60 == 0:
                debug_print(f"VideoFrame created: {video_frame.width}x{video_frame.height}, pts: {pts}")
            
            return video_frame
        except Exception as e:
            debug_print(f"Error creating VideoFrame: {e}")
            # Return a simple test frame
            test_frame = self.create_test_frame("VideoFrame Error")
            test_frame = cv2.cvtColor(test_frame, cv2.COLOR_BGR2RGB)
            video_frame = VideoFrame.from_ndarray(test_frame, format="rgb24")
            video_frame.pts = pts
            video_frame.time_base = time_base
            return video_frame
    
    def create_test_frame(self, message="Test Pattern"):
        """Create a test pattern frame when camera/video is not available"""
        # Create a 640x480 test pattern with moving elements
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Animated background color based on frame count
        color_intensity = int(127 + 127 * np.sin(self.frame_count * 0.1))
        frame[:] = (color_intensity, 50, 100)  # BGR format
        
        # Add some text
        cv2.putText(frame, message, (50, 200), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Add current source info
        source_text = f"Source: {'Camera' if self.use_camera else 'Video File'}"
        cv2.putText(frame, source_text, (50, 150), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add frame counter
        cv2.putText(frame, f"Frame: {self.frame_count}", (50, 250), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add switching instructions
        cv2.putText(frame, "Press '3' to switch source", (50, 300), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Add moving circle
        center_x = int(320 + 100 * np.sin(self.frame_count * 0.05))
        center_y = int(240 + 50 * np.cos(self.frame_count * 0.05))
        cv2.circle(frame, (center_x, center_y), 20, (0, 255, 0), -1)
        
        return frame
    
    def cleanup(self):
            """Clean up resources when track is closed"""
            if hasattr(self, 'detector') and self.detector:
                self.detector.stop()
            if hasattr(self, 'cap') and self.cap:
                self.cap.release()
            if hasattr(self, 'video_cap') and self.video_cap:
                self.video_cap.release()

@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the client HTML."""
    # Print interactive instructions to terminal instead of rendering HTML text
    try:
        _print_runtime_instructions()
    except Exception:
        pass
    html = '''
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8" />
  <title>WebRTC OpenCV + DataChannel Demo</title>   
  <style>
    :root {
        --hud-red: rgb(255, 64, 64);
        --hud-red-stroke: rgba(255, 64, 64, 0.35);
        --hud-red-bg-1: rgba(48, 12, 12, 0.92);
        --hud-red-bg-2: rgba(22, 10, 10, 0.86);
        --hud-yellow: rgb(255, 220, 90);
        --hud-cyan: rgb(0, 220, 255);
        --hud-muted: rgba(231, 246, 255, 0.85);
    }
    body {
        margin: 0;
        font-family: sans-serif;
        overflow: hidden; /* Prevent scrollbars from appearing */
    }
    video#video {
        position: fixed;
        top: 0;
        left: 0;
        width: 100vw;
        height: 100vh;
        object-fit: cover;
        z-index: -1; /* Place it in the background */
        cursor: crosshair;
    }
     /* Folder-style entity card anchored middle-right */
     .entity-stack {
         position: fixed;
         top: 50%;
         right: 2vw;
         transform: translateY(-50%);
         display: flex;
         flex-direction: column;
         align-items: stretch;
         gap: 1.2vh;
         z-index: 10;
         pointer-events: none;
         max-height: 96vh;
         max-width: 40vw;
         /* Enable 3D space so children can rotate in perspective */
         perspective: 1000px;
         perspective-origin: right center;
         transform-style: preserve-3d;
     }
     .entity-card {
         /* Cyberpunk folder styling: red theme container, teal/yellow semantic text */
         --folder-bg: linear-gradient(180deg, var(--hud-red-bg-1) 0%, var(--hud-red-bg-2) 100%);
         --folder-stroke: var(--hud-red-stroke);
         position: relative;
         display: flex;
         flex-direction: column;
         align-items: stretch;
         width: clamp(280px, 10vw, 544px);
         background: var(--folder-bg);
         border: 1px solid var(--folder-stroke);
         border-radius: 8px;
         box-shadow: 0 8px 18px rgba(0,0,0,0.28);
         color: #e7f6ff;
         overflow: visible; /* allow tab to sit above the folder top */
         pointer-events: auto;
          /* Anchor the right edge like a door hinge */
          transform-origin: right center;
          /* Tilt the left side away from the viewer for 3D immersion */
          transform: rotateY(-20deg);
          backface-visibility: hidden;
          transform-style: preserve-3d;
         animation: folder-enter 220ms ease;
         z-index: 0; /* create a stacking context for pseudo elements */
     }
     /* Folder tab (DOM element) with cyberpunk translucent accent and label */
     .folder-tab {
         position: absolute;
         top: -16px;
         left: 14px;
         width: min(34%, 160px);
         height: 24px;
         display: flex;
         align-items: center;
         justify-content: center;
          background: linear-gradient(180deg, rgba(255,64,64,0.20) 0%, rgba(255,64,64,0.06) 100%);
          border: 1px solid var(--hud-red-stroke);
         border-bottom: none;
         border-top-left-radius: 8px;
         border-top-right-radius: 8px;
          color: var(--hud-cyan);
         font-weight: 700;
         font-size: 11px;
         letter-spacing: 1px;
         text-transform: uppercase;
         box-shadow: 0 4px 10px rgba(0,0,0,0.15);
         pointer-events: auto;
         z-index: 3; /* above content and grain */
     }
      .folder-tab .close-btn{
         position: absolute;
         right: 6px;
         top: 3px;
         width: 16px;
         height: 16px;
         display: inline-flex;
         align-items: center;
         justify-content: center;
         font-weight: 800;
         font-size: 11px;
          color: var(--hud-cyan);
          border: 1px solid var(--hud-red-stroke);
         border-radius: 3px;
          background: rgba(10,18,20,0.6);
         cursor: pointer;
         user-select: none;
     }
      .folder-tab .close-btn:hover{ background: rgba(255,64,64,0.18); }
     /* Removed paper grain lines for cleaner transparency */
      .entity-card .content {
         padding: clamp(10px, 1.2vh, 14px) clamp(14px, 1.4vw, 18px);
         font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
          font-size: __ENTITY_CARD_FONT_SIZE__px;
         line-height: 1.32; /* slightly tighter for title-to-text spacing */
          color: var(--hud-cyan);
         white-space: pre-wrap;
         overflow-wrap: anywhere;
         max-height: 72vh; /* expands until this, then scrolls */
         overflow-y: auto;
         position: relative;
         z-index: 2; /* ensure text is in front of borders/grain */
     }
      /* Tabbed entity card layout */
      .tabbed-entity-card { position: relative; }
      .tabbed-entity-card .tab-bar {
          position: absolute;
          top: -16px;
          left: 14px;
          right: 14px;
          display: flex;
          align-items: center;
          gap: 6px;
          padding: 0;
          background: transparent;
          overflow-x: auto;
          scrollbar-width: thin;
          z-index: 3;
      }
      .tabbed-entity-card .tab {
          position: relative;
          display: inline-flex;
          align-items: center;
          justify-content: center;
          gap: 8px;
          height: 24px;
          padding: 0 10px;
          background: linear-gradient(180deg, rgba(255,64,64,0.20) 0%, rgba(255,64,64,0.06) 100%);
          border: 1px solid var(--hud-red-stroke);
          border-bottom: none;
          border-top-left-radius: 8px;
          border-top-right-radius: 8px;
          color: var(--hud-cyan);
          font-weight: 700;
          font-size: 11px;
          letter-spacing: 1px;
          text-transform: uppercase;
          cursor: pointer;
          user-select: none;
          box-shadow: 0 4px 10px rgba(0,0,0,0.15);
      }
      .tabbed-entity-card .tab.active {
          background: linear-gradient(180deg, rgba(255,64,64,0.26) 0%, rgba(255,64,64,0.12) 100%);
          border-color: rgba(255,64,64,0.55);
      }
      .tabbed-entity-card .tab .close-btn {
          display: inline-flex;
          align-items: center;
          justify-content: center;
          width: 16px;
          height: 16px;
          border: 1px solid var(--hud-red-stroke);
          border-radius: 3px;
          font-size: 11px;
          font-weight: 800;
          color: var(--hud-cyan);
          background: rgba(10,18,20,0.6);
      }
      .tabbed-entity-card .tab .close-btn:hover { background: rgba(255,64,64,0.18); }
      .tabbed-entity-card .tab-panels { padding: 12px 12px 12px 12px; }
      .tabbed-entity-card .panel {
          display: none;
          white-space: pre-wrap;
          overflow-wrap: anywhere;
          color: var(--hud-cyan);
          font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
          font-size: __ENTITY_CARD_FONT_SIZE__px;
          line-height: 1.32;
          max-height: 72vh;
          overflow-y: auto;
      }
      .tabbed-entity-card .panel.active { display: block; }
      .tabbed-entity-card .panel.streaming { outline: 1px dashed rgba(57,255,186,0.25); color: var(--hud-cyan); }
      .tabbed-entity-card .panel.streaming::first-line { color: var(--hud-interact); }
      .tabbed-entity-card .panel.done { outline: 1px solid rgba(0,0,0,0.12); }

      /* Semantic HUD text colors */
      .hud-scan { color: var(--hud-cyan); }
      .hud-interact { color: var(--hud-yellow); }
      .hud-threat { color: var(--hud-red); }
      .hud-muted { color: var(--hud-muted); }
     .entity-card.streaming { outline: 1px solid rgba(0,0,0,0.08); }
     .entity-card.done { outline: 1px solid rgba(0,0,0,0.12); }

      @keyframes folder-enter {
         from { transform: rotateY(-20deg) scaleY(0.8); opacity: 0; }
         to { transform: rotateY(-20deg) scaleY(1); opacity: 1; }
      }

      /* Minimalistic query bar styled to match entity card */
      .query-bar {
          position: fixed;
          bottom: 20%; /* Position in lower area like ultrawide monitor */
          left: 50%;
          transform: translate(-50%, 0) perspective(1000px) rotateX(15deg) rotateY(-2deg);
          transform-style: preserve-3d;
          display: flex;
          align-items: center;
          gap: 6px;
          width: min(55vw, 800px); /* Smaller ultrawide width */
          height: 45px;
          background: linear-gradient(135deg, rgba(10,18,20,0.6) 0%, rgba(5,10,15,0.8) 50%, rgba(10,18,20,0.6) 100%);
          border: 1px solid var(--hud-red-stroke);
          border-radius: 30px; /* Smaller curved edges */
          padding: 8px 18px;
          z-index: 11;
          pointer-events: auto;
          box-shadow: 
              0 12px 40px rgba(0,0,0,0.8),
              0 6px 20px rgba(0,0,0,0.6),
              inset 0 2px 4px rgba(255,255,255,0.05),
              0 0 0 1px rgba(0,220,255,0.1);
          backdrop-filter: blur(12px);
      }
      
      /* Create curved screen effect with pseudo-element */
      .query-bar::before {
          content: '';
          position: absolute;
          top: -2px;
          left: -2px;
          right: -2px;
          bottom: -2px;
          background: linear-gradient(90deg, 
              rgba(0,220,255,0.1) 0%, 
              rgba(0,220,255,0.05) 25%,
              rgba(0,220,255,0.02) 50%, 
              rgba(0,220,255,0.05) 75%,
              rgba(0,220,255,0.1) 100%);
          border-radius: 32px;
          z-index: -1;
          transform: scaleX(1.02) rotateX(5deg);
          filter: blur(1px);
      }
      
      /* Additional curved distortion effect */
      .query-bar::after {
          content: '';
          position: absolute;
          top: 50%;
          left: 50%;
          width: 60%;
          height: 120%;
          background: radial-gradient(ellipse at center, rgba(0,0,0,0.2) 0%, transparent 70%);
          transform: translate(-50%, -50%) perspective(500px) rotateX(25deg);
          z-index: 1;
          pointer-events: none;
          border-radius: 50%;
      }
      .query-bar input[type="text"]{
          flex: 1; /* Take up remaining space in curved container */
          background: rgba(0,0,0,0.3);
          border: 1px solid rgba(0,220,255,0.4);
          color: var(--hud-cyan);
          border-radius: 20px; /* Smaller curved radius */
          padding: 6px 14px;
          outline: none;
          font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
          font-size: 13px;
          transition: all 0.3s ease;
          z-index: 2;
          position: relative;
      }
      .query-bar input[type="text"]::placeholder { color: rgba(0,220,255,0.55); }
      .query-bar input[type="text"]:focus {
          border-color: var(--hud-cyan);
          box-shadow: 0 0 20px rgba(0,220,255,0.3);
          background: rgba(0,0,0,0.3);
      }
      .query-bar button{
          height: 32px;
          padding: 0 12px;
          background: rgba(255,64,64,0.15);
          border: 1px solid var(--hud-red-stroke);
          color: var(--hud-cyan);
          border-radius: 16px; /* Smaller curved button */
          font-weight: 600;
          font-size: 12px;
          letter-spacing: 0.3px;
          cursor: pointer;
          transition: all 0.3s ease;
          white-space: nowrap;
          z-index: 2;
          position: relative;
      }
      .query-bar button:hover{ 
          background: rgba(255,64,64,0.2); 
          box-shadow: 0 0 15px rgba(255,64,64,0.3);
          transform: scale(1.05);
      }

      /* Agent process trace under left side of curved search bar */
      .process-trace {
          position: fixed;
          bottom: calc(20% - 60px); /* positioned above the smaller curved bar */
          left: calc(50% - 27.5vw); /* align with left edge of smaller ultrawide bar */
          width: min(25vw, 350px);
          height: 24px; /* smaller to match proportions */
          overflow: hidden;
          background: rgba(10,18,20,0.4);
          border: 1px solid var(--hud-red-stroke);
          border-radius: 12px; /* smaller curve to match design */
          z-index: 11;
          pointer-events: none;
          backdrop-filter: blur(6px);
      }
      .trace-slot {
          display: flex;
          flex-direction: column;
          will-change: transform;
          transition: transform 260ms ease;
      }
      .trace-item {
          height: 24px;
          display: flex;
          align-items: center;
          padding: 0 8px;
          color: var(--hud-cyan);
          font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
          font-size: 12px;
          white-space: nowrap;
      }
      .settings-toggle {
          position: fixed;
          bottom: 6%;
          right: 2vw;
          height: 32px;
          padding: 0 12px;
          background: rgba(255,64,64,0.15);
          border: 1px solid var(--hud-red-stroke);
          color: var(--hud-cyan);
          border-radius: 16px;
          font-weight: 600;
          font-size: 12px;
          letter-spacing: 0.3px;
          cursor: pointer;
          z-index: 12;
          box-shadow: 0 6px 18px rgba(0,0,0,0.5);
      }
      .settings-panel {
          position: fixed;
          bottom: 12%;
          right: 2vw;
          width: 240px;
          max-height: 50vh;
          overflow-y: auto;
          background: linear-gradient(180deg, var(--hud-red-bg-1) 0%, var(--hud-red-bg-2) 100%);
          border: 1px solid var(--hud-red-stroke);
          border-radius: 10px;
          box-shadow: 0 10px 24px rgba(0,0,0,0.6);
          z-index: 12;
          display: none;
      }
      .settings-panel.open { display: block; }
      .settings-header {
          padding: 10px 12px;
          font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
          font-size: 12px;
          letter-spacing: 1px;
          text-transform: uppercase;
          color: var(--hud-cyan);
          border-bottom: 1px solid var(--hud-red-stroke);
      }
      .settings-list { padding: 8px 12px 10px 12px; }
      .settings-item {
          display: flex;
          align-items: center;
          justify-content: space-between;
          gap: 10px;
          padding: 6px 0;
          color: var(--hud-cyan);
          font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
          font-size: 12px;
      }
      .settings-item input[type="checkbox"] {
          width: 14px;
          height: 14px;
          accent-color: var(--hud-cyan);
      }
  </style>
</head>
<body>
  <video id="video" autoplay playsinline muted></video>
   <div class="entity-stack" id="entityStack"></div>
    <div class="query-bar" id="queryBar">
      <input type="text" id="queryInput" placeholder="Ask a question..." />
      <button id="querySend">ASK</button>
    </div>
    <div class="process-trace" id="processTrace" style="display:none;">
      <div class="trace-slot" id="traceSlot"></div>
    </div>
    <button class="settings-toggle" id="settingsToggle">SETTINGS</button>
    <div class="settings-panel" id="settingsPanel">
      <div class="settings-header">Settings</div>
      <div class="settings-list" id="settingsList"></div>
    </div>

  <script>
  const SETTINGS_DEFAULTS = __SETTINGS_JSON__;
  const SETTINGS_SCHEMA = [
    { key: 'DEBUG_MODE', label: 'Debug mode' },
    { key: 'ENHANCED_DEBUG', label: 'Enhanced debug' },
    { key: 'VERBOSE_FRAME_LOGGING', label: 'Verbose frame logging' },
    { key: 'SHOW_OBJECT_TITLES', label: 'Show object titles' },
    { key: 'SHOW_CONFIDENCE_SCORES', label: 'Show confidence' },
    { key: 'USE_LIGHT_OBJECT_TITLES', label: 'Light title style' },
    { key: 'SHOW_DETECTION_ARROWS', label: 'Show arrows' },
    { key: 'SHOW_CORNER_MARKERS', label: 'Show corner markers' },
    { key: 'ENABLE_BASE_EDGE_BACKGROUND', label: 'Base edge background' },
    { key: 'ENABLE_HEATMAP_MODE', label: 'Heatmap haze' },
    { key: 'SHOW_QUERY_BAR', label: 'Show query bar' },
    { key: 'ENABLE_POSE_ESTIMATION', label: 'Pose estimation' },
    { key: 'USE_PRIORITY_CLASSES', label: 'Priority classes' }
  ];
  const pc = new RTCPeerConnection({
    iceServers: [{ urls: 'stun:stun.l.google.com:19302' }]
  });
  
  pc.addTransceiver('video', { direction: 'recvonly' });
  const dc = pc.createDataChannel("chat");
  const video = document.getElementById('video');
  const entityStack = document.getElementById('entityStack');

  // Map mouse coordinates to video pixel coordinates with object-fit: cover handling
  function toVideoCoords(e){
    const rect = video.getBoundingClientRect();
    const vw = video.videoWidth || 1;
    const vh = video.videoHeight || 1;
    const elW = rect.width || 1;
    const elH = rect.height || 1;
    // For object-fit: cover, scale is the max that fills the element
    const scale = Math.max(elW / vw, elH / vh);
    const dispW = vw * scale;
    const dispH = vh * scale;
    const offX = (elW - dispW) / 2;
    const offY = (elH - dispH) / 2;
    const x = Math.round(((e.clientX - rect.left) - offX) / dispW * vw);
    const y = Math.round(((e.clientY - rect.top) - offY) / dispH * vh);
    return [Math.max(0, Math.min(vw - 1, x)), Math.max(0, Math.min(vh - 1, y))];
  }

  function ensureTabbedCard(){
    let container = document.getElementById('tabbedEntityCard');
    if (!container){
      container = document.createElement('div');
      container.className = 'entity-card tabbed-entity-card';
      container.id = 'tabbedEntityCard';
      container.innerHTML = '<div class="tab-bar" id="tabBar"></div><div class="tab-panels" id="tabPanels"></div>';
      entityStack.prepend(container);
    }
    return container;
  }

  function setActiveTab(tabId){
    const container = ensureTabbedCard();
    const tabBar = container.querySelector('#tabBar');
    const panels = container.querySelector('#tabPanels');
    const tabs = tabBar.querySelectorAll('.tab');
    const panelEls = panels.querySelectorAll('.panel');
    tabs.forEach(t => t.classList.remove('active'));
    panelEls.forEach(p => p.classList.remove('active'));
    const tab = tabBar.querySelector(`.tab[data-id="${tabId}"]`);
    const panel = panels.querySelector(`.panel[data-id="${tabId}"]`);
    if (tab) tab.classList.add('active');
    if (panel) panel.classList.add('active');
  }

  // Colorize entity text using Kiroshi HUD semantics
  function colorizeText(raw){
    try {
      const safe = String(raw || '')
        .replace(/&/g,'&amp;')
        .replace(/</g,'&lt;')
        .replace(/>/g,'&gt;');
      const reHeading = /^(POSTURE|LIGHTING|FOCUS|CONDITION|MUSCLE|AMBIENCE)\\s*:/gm;
      const reThreat = /\\b(CIVILIAN|ENEMY|ALLY)\\b/g;
      const reScan = /\\b(SCAN RESULTS|AFFILIATION|DATABASE|BACKGROUND|EARLY LIFE|SIGNIFICANT EVENTS)\\b/g;
      const reInteract = /\\b(HACK|INTERACT|ACCESS)\\b/g;
      // First line is the title: color it yellow
      const parts = safe.split(/\\n/);
      if (parts.length > 0) {
        parts[0] = `<span class="hud-interact">${parts[0]}</span>`;
      }
      const body = parts.slice(1).join('\\n')
        .replace(reHeading, '<span class="hud-scan">$1</span>:')
        .replace(reThreat, '<span class="hud-threat">$1</span>')
        .replace(reScan, '<span class="hud-scan">$1</span>')
        .replace(reInteract, '<span class="hud-interact">$1</span>');
      return parts.slice(0,1).join('\\n') + (body ? '\\n' + body : '');
    } catch (_) {
      return String(raw || '');
    }
  }

  function addLog(message) {
    try { if (dc && dc.readyState === 'open') { dc.send(JSON.stringify({ type: 'client_log', message })); } } catch (e) {}
    console.log(message);
  }

  dc.onopen = () => addLog('[DC] Opened');
  const settingsToggle = document.getElementById('settingsToggle');
  const settingsPanel = document.getElementById('settingsPanel');
  const settingsList = document.getElementById('settingsList');
  const queryBar = document.getElementById('queryBar');

  function renderSettings(){
    settingsList.innerHTML = '';
    SETTINGS_SCHEMA.forEach((item) => {
      const row = document.createElement('div');
      row.className = 'settings-item';
      const label = document.createElement('label');
      label.textContent = item.label;
      const input = document.createElement('input');
      input.type = 'checkbox';
      input.dataset.key = item.key;
      input.checked = !!SETTINGS_DEFAULTS[item.key];
      input.addEventListener('change', sendSettings);
      row.appendChild(label);
      row.appendChild(input);
      settingsList.appendChild(row);
    });
    applyQueryBarVisibility(SETTINGS_DEFAULTS.SHOW_QUERY_BAR !== false);
  }

  function applyQueryBarVisibility(show){
    if (!queryBar) return;
    queryBar.style.display = show ? 'flex' : 'none';
    if (!show) {
      processTraceEl.style.display = 'none';
    }
  }

  function sendSettings(){
    const payload = {};
    SETTINGS_SCHEMA.forEach((item) => {
      const el = settingsList.querySelector(`input[data-key="${item.key}"]`);
      if (el) payload[item.key] = !!el.checked;
    });
    applyQueryBarVisibility(payload.SHOW_QUERY_BAR !== false);
    if (dc && dc.readyState === 'open'){
      dc.send(JSON.stringify({ type: 'settings_update', settings: payload }));
      addLog('[Settings] Updated');
    } else {
      addLog('[Settings] Data channel not ready');
    }
  }

  settingsToggle.addEventListener('click', () => {
    settingsPanel.classList.toggle('open');
  });
  function animateCardToContentHeight(card){
    try {
      const prev = card.offsetHeight;
      // Lock current height to enable transition from fixed height
      card.style.height = prev + 'px';
      card.style.willChange = 'height';
      requestAnimationFrame(() => {
        const target = card.scrollHeight;
        card.style.transition = 'height 220ms ease';
        card.style.height = target + 'px';
        const cleanup = (e) => {
          if (!e || e.propertyName === 'height'){
            card.style.transition = '';
            card.style.height = '';
            card.style.willChange = '';
            card.removeEventListener('transitionend', cleanup);
          }
        };
        card.addEventListener('transitionend', cleanup);
      });
    } catch (_) {}
  }

  function upsertEntityCard(id, status, text){
    const key = String(id);
    const container = ensureTabbedCard();
    const tabBar = container.querySelector('#tabBar');
    const panels = container.querySelector('#tabPanels');

    let tab = tabBar.querySelector(`.tab[data-id="${key}"]`);
    let panel = panels.querySelector(`.panel[data-id="${key}"]`);

    if (!tab){
      tab = document.createElement('div');
      tab.className = 'tab';
      tab.dataset.id = key;
      tab.innerHTML = `<span class="label hud-scan">DATA ${key}</span> <span class="close-btn" title="Close">X</span>`;
      const closeBtn = tab.querySelector('.close-btn');
      closeBtn.addEventListener('click', (ev) => {
        ev.stopPropagation();
        const idToClose = tab.dataset.id;
        const panelToClose = panels.querySelector(`.panel[data-id="${idToClose}"]`);
        tab.remove();
        if (panelToClose) panelToClose.remove();
        // If no tabs left, remove the whole container
        if (!tabBar.querySelector('.tab')){
          container.remove();
        } else {
          // Activate the first remaining tab
          const firstTab = tabBar.querySelector('.tab');
          if (firstTab){
            setActiveTab(firstTab.dataset.id);
          }
        }
      });
      tab.addEventListener('click', () => setActiveTab(key));
      tabBar.appendChild(tab);
    }

    if (!panel){
      panel = document.createElement('div');
      panel.className = 'panel';
      panel.dataset.id = key;
      panels.appendChild(panel);
    }

    panel.classList.remove('streaming','done');
    if (status === 'streaming') panel.classList.add('streaming');
    if (status === 'done') panel.classList.add('done');

    if (status === 'done'){
      panel.innerHTML = colorizeText(text);
      animateCardToContentHeight(container);
    } else {
      // Stream with minimal processing to avoid heavy DOM updates per token
      panel.textContent = text || '';
    }

    // Activate this tab
    setActiveTab(key);
  }

  // Query bar handlers
  function sendTextQuery(){
    try{
      const input = document.getElementById('queryInput');
      const text = (input.value || '').trim();
      if (!text) return;
      if (dc.readyState === 'open'){
        dc.send(JSON.stringify({ type: 'text_query', text }));
        addLog(`[Query] Sent: ${text}`);
        input.value = '';
      } else {
        addLog('[Query] Data channel not ready');
      }
    } catch (e) {}
  }
  document.getElementById('querySend').addEventListener('click', sendTextQuery);
  document.getElementById('queryInput').addEventListener('keydown', (e) => {
    if (e.key === 'Enter') sendTextQuery();
  });

  // Agent process trace UI
  const processTraceEl = document.getElementById('processTrace');
  const traceSlotEl = document.getElementById('traceSlot');
  let traceCount = 0;
  function traceStart(){
    try{
      traceSlotEl.innerHTML = '';
      traceCount = 0;
      processTraceEl.style.display = 'block';
    }catch(_){ }
  }
  function traceStep(text){
    try{
      const item = document.createElement('div');
      item.className = 'trace-item';
      item.textContent = text;
      traceSlotEl.appendChild(item);
      traceCount += 1;
      const dy = -24 * (traceCount - 1);
      traceSlotEl.style.transform = `translateY(${dy}px)`;
    }catch(_){ }
  }
  function traceDone(){
    try{
      setTimeout(() => { processTraceEl.style.display = 'none'; }, 1200);
    }catch(_){ }
  }

  dc.onmessage = e => {
    try {
      const data = JSON.parse(e.data);
      if (data.type === 'entity_card'){
        upsertEntityCard(data.id, data.status, data.text);
        return;
      }
      if (data.type === 'text_query_ack') {
        addLog('[Query] Accepted by server');
        return;
      }
      if (data.type === 'agent_trace') {
        if (data.status === 'start') { traceStart(); return; }
        if (data.status === 'done') { traceStep('Done'); traceDone(); return; }
        if (data.event === 'step' && data.text) { traceStep(String(data.text)); return; }
      }
      if (data.type === 'settings_ack') {
        addLog('[Settings] Applied');
        return;
      }
      if (data.type === 'click_response') {
        addLog(`[DC] Click processed at (${data.x}, ${data.y}) - ${data.status}`);
      } else if (data.type === 'video_source_switched') {
        if (data.status === 'success') {
          addLog(`[Video] Switched to: ${data.source}`);
        } else {
          addLog(`[Video] Switch failed: ${data.message}`);
        }
      } else {
        addLog(`[DC] Server: ${e.data}`);
      }
    } catch {
      addLog(`[DC] Server: ${e.data}`);
    }
  };

  pc.onconnectionstatechange = () => {
    console.log('Connection state:', pc.connectionState);
    addLog(`[WebRTC] Connection state: ${pc.connectionState}`);
  };

  pc.oniceconnectionstatechange = () => {
    console.log('ICE connection state:', pc.iceConnectionState);
    addLog(`[WebRTC] ICE state: ${pc.iceConnectionState}`);
  };

  pc.ontrack = ({ streams: [stream] }) => {
    console.log('Track received:', stream);
    addLog(`[WebRTC] Track received, ${stream.getTracks().length} tracks`);
    
    const videoTrack = stream.getVideoTracks()[0];
    if (videoTrack) {
      addLog(`[Video] Track settings: ${JSON.stringify(videoTrack.getSettings())}`);
      
      videoTrack.onended = () => addLog('[Video] Track ended');
      videoTrack.onmute = () => addLog('[Video] Track muted');
      videoTrack.onunmute = () => addLog('[Video] Track unmuted');
    }
    
    video.srcObject = stream;
  };

  // Video element events
  video.onloadstart = () => addLog('[Video] Load start');
  video.onloadedmetadata = () => addLog(`[Video] Metadata loaded: ${video.videoWidth}x${video.videoHeight}`);
  video.oncanplay = () => addLog('[Video] Can play');
  video.onplaying = () => addLog('[Video] Playing');
  video.onerror = (e) => addLog(`[Video] Error: ${e.message}`);
  
  // Mouse handlers for interactive drawing
  let isDrawing = false;
  let lastMoveTime = 0;
  const throttleInterval = 50; // ms
  
  video.onmousedown = (e) => {
    isDrawing = true;
    const [x, y] = toVideoCoords(e);
    
      const eventData = { type: 'mouse_down', x: x, y: y, timestamp: Date.now() };
      if (dc.readyState === 'open') { dc.send(JSON.stringify(eventData)); }
    addLog(`[Draw] Start: (${x}, ${y})`);
  };
  
  video.onmousemove = (e) => {
    if (isDrawing) {
      const now = Date.now();
      if (now - lastMoveTime < throttleInterval) return;
      lastMoveTime = now;
      const [x, y] = toVideoCoords(e);
      
      const eventData = { type: 'mouse_move', x: x, y: y, timestamp: Date.now() };
      if (dc.readyState === 'open') { dc.send(JSON.stringify(eventData)); }
    }
  };
  
  video.onmouseup = (e) => {
    if (isDrawing) {
      isDrawing = false;
      const [x, y] = toVideoCoords(e);
      
      const eventData = { type: 'mouse_up', x: x, y: y, timestamp: Date.now() };
      if (dc.readyState === 'open') { dc.send(JSON.stringify(eventData)); }
      addLog(`[Draw] End: (${x}, ${y})`);
    }
  };
  
  video.onmouseleave = (e) => {
    if (isDrawing) {
      isDrawing = false;
      const [x, y] = toVideoCoords(e);
      
      const eventData = { type: 'mouse_up', x: x, y: y, timestamp: Date.now() };
      if (dc.readyState === 'open') { dc.send(JSON.stringify(eventData)); }
      addLog(`[Draw] End (mouseleave): (${x}, ${y})`);
    }
  };

  async function start() {
    try {
      addLog('[WebRTC] Creating offer...');
      const offer = await pc.createOffer();
      await pc.setLocalDescription(offer);
      addLog('[WebRTC] Offer created and set as local description');
      
      const screen_info = {
          screen_width: window.screen.width,
          screen_height: window.screen.height,
          available_width: window.screen.availWidth,
          available_height: window.screen.availHeight,
          ViewPort_width: document.documentElement.clientWidth,
          ViewPort_height: document.documentElement.clientHeight,
          dpi: window.devicePixelRatio || 1,
          is_mobile: /Mobi|Android/i.test(navigator.userAgent),
          is_tablet: /Tablet|iPad/i.test(navigator.userAgent),
          is_desktop: !/Mobi|Android/i.test(navigator.userAgent) && !/Tablet|iPad/i.test(navigator.userAgent)
      };
      
      addLog(`[Screen] Sending screen info: ${screen_info.ViewPort_width}x${screen_info.ViewPort_height}`);
      
      const resp = await fetch('/offer', {
        method: 'POST',
        body: JSON.stringify({ 
            sdp: pc.localDescription.sdp, 
            type: pc.localDescription.type,
            screen_info: screen_info
        }),
        headers: {'Content-Type':'application/json'}
      });
      const answer = await resp.json();
      addLog('[WebRTC] Answer received from server');
      
      await pc.setRemoteDescription(answer);
      addLog('[WebRTC] Remote description set - signaling complete');
    } catch (error) {
      console.error('Error in start():', error);
      addLog(`[ERROR] ${error.message}`);
    }
  }
  
  // Keyboard event handler for video source switching
  document.addEventListener('keydown', (event) => {
    if (event.key === '3') {
      if (dc.readyState === 'open') {
        const switchData = {
          type: 'switch_video_source',
          timestamp: Date.now()
        };
        
        dc.send(JSON.stringify(switchData));
        addLog('[Keyboard] Pressed "3" - switching video source...');
      } else {
        addLog('[Keyboard] Data channel not ready for video switch');
      }
    }
  });

  start();
  renderSettings();

  document.getElementById('sendBtn').onclick = () => {
    const msg = document.getElementById('msg').value;
    if (dc.readyState === 'open') {
      dc.send(msg);
      addLog(`[DC] You: ${msg}`);
      document.getElementById('msg').value = '';
    } else {
      addLog('[DC] Not connected');
    }
  };

  document.getElementById('debugBtn').onclick = () => {
    addLog(`[Debug] Video element: ${video.videoWidth}x${video.videoHeight}`);
    addLog(`[Debug] Video ready state: ${video.readyState}`);
    addLog(`[Debug] Video paused: ${video.paused}`);
    addLog(`[Debug] PC state: ${pc.connectionState}`);
    addLog(`[Debug] PC ice state: ${pc.iceConnectionState}`);
  };
  </script>
</body>
</html>
    '''
    # Inject constant font size into the CSS
    try:
        html = html.replace('__ENTITY_CARD_FONT_SIZE__', str(ENTITY_CARD_FONT_SIZE_PX))
    except Exception:
        pass
    try:
        settings_defaults = {
            "DEBUG_MODE": DEBUG_MODE,
            "ENHANCED_DEBUG": ENHANCED_DEBUG,
            "VERBOSE_FRAME_LOGGING": VERBOSE_FRAME_LOGGING,
            "SHOW_OBJECT_TITLES": SHOW_OBJECT_TITLES,
            "SHOW_CONFIDENCE_SCORES": SHOW_CONFIDENCE_SCORES,
            "SHOW_DETECTION_ARROWS": SHOW_DETECTION_ARROWS,
            "SHOW_CORNER_MARKERS": SHOW_CORNER_MARKERS,
            "ENABLE_BASE_EDGE_BACKGROUND": ENABLE_BASE_EDGE_BACKGROUND,
            "USE_LIGHT_OBJECT_TITLES": USE_LIGHT_OBJECT_TITLES,
            "SHOW_QUERY_BAR": SHOW_QUERY_BAR,
            "ENABLE_POSE_ESTIMATION": ENABLE_POSE_ESTIMATION,
            "USE_PRIORITY_CLASSES": USE_PRIORITY_CLASSES,
            "ENABLE_HEATMAP_MODE": ENABLE_HEATMAP_MODE,
        }
        html = html.replace('__SETTINGS_JSON__', json.dumps(settings_defaults, ensure_ascii=True))
    except Exception:
        pass
    return HTMLResponse(html)

@app.get("/healthz")
async def healthz():
    with ui_channels_lock:
        ui_count = len(ui_channels)
    return {
        "ok": True,
        "mode": RUN_MODE,
        "openai": async_client is not None,
        "pcs": len(pcs),
        "ui_channels": ui_count,
    }

@app.post("/offer")
async def offer(request: Request):
    """
    Handle SDP offer from browser, create answer with a video track
    and a data channel listener for two-way interaction.
    """
    try:
        params = await request.json()
        offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])
        debug_print(f"Received offer: {offer.type}")

        client_screen_info = params.get("screen_info")
        if client_screen_info:
            debug_print(f"[OFFER] Received client screen info: {client_screen_info}")
        else:
            debug_print(f"[OFFER] No client screen info received, using defaults.")

        pc = RTCPeerConnection()
        pcs.add(pc)

        # Add connection state logging
        @pc.on("connectionstatechange")
        async def on_connectionstatechange():
            debug_print(f"Connection state: {pc.connectionState}")

        @pc.on("iceconnectionstatechange")
        async def on_iceconnectionstatechange():
            debug_print(f"ICE connection state: {pc.iceConnectionState}")

        # 1) Add video track (OpenCV webcam)
        video_track = OpenCVVideoTrack(device_index=0, client_screen_info=client_screen_info)
        video_tracks.add(video_track)
        sender = pc.addTrack(video_track)
        try:
            params = sender.getParameters()
            # Dynamic bitrate/framerate based on computed quality level
            quality = getattr(video_track, 'quality_level', '1080p')
            bitrate_map = {
                '4K': 25_000_000,     # 25 Mbps for UHD
                '2K': 12_000_000,     # 12 Mbps for QHD
                '1080p': 8_000_000,   # 8 Mbps for Full HD
                '720p': 5_000_000,    # 5 Mbps for HD
                '480p': 2_500_000     # 2.5 Mbps for SD
            }
            max_framerate_map = {
                '4K': 30,
                '2K': 60,
                '1080p': 60,
                '720p': 60,
                '480p': 60
            }
            target_bitrate = bitrate_map.get(quality, 8_000_000)
            target_max_fps = max_framerate_map.get(quality, 30)

            enc = RTCRtpEncodingParameters(maxBitrate=target_bitrate)
            # Some aiortc versions support maxFramerate; set if available
            try:
                setattr(enc, 'maxFramerate', target_max_fps)
            except Exception:
                pass

            params.encodings = [enc]
            sender.setParameters(params)
            debug_print(f"[WebRTC] Applied encoding params - Quality: {quality}, MaxBitrate: {target_bitrate/1_000_000:.1f} Mbps, MaxFramerate: {target_max_fps}fps")
        except Exception as e:
            debug_print(f"[WebRTC] Could not set dynamic encoding params: {e}")
        debug_print("Video track added to peer connection")

        # 2) Data channel handler
        @pc.on("datachannel")
        def on_datachannel(channel):
            debug_print(f"Data channel created: {channel.label}")
            with ui_channels_lock:
                ui_channels.add(channel)

            @channel.on("message")
            def on_message(message):
                debug_print(f"[DataChannel] Received: {message}")
                
                try:
                    # Try to parse as JSON for click coordinates
                    import json
                    data = json.loads(message)
                    
                    # Forward client logs to terminal instead of HTML
                    if data.get('type') == 'client_log':
                        debug_print(f"[CLIENT] {data.get('message')}")
                        return

                    if data.get('type') == 'click':
                        x, y = data.get('x'), data.get('y')
                        debug_print(f"[DataChannel] Processing click at ({x}, {y})")
                        
                        # Forward click to the video track's detector
                        if hasattr(video_track, 'detector') and video_track.detector:
                            video_track.detector.handle_click(x, y)
                            
                        # Send confirmation back to client
                        channel.send(json.dumps({'type': 'click_response','x': x,'y': y,'status': 'processed'}))
                    elif data.get('type') == 'mouse_down':
                        x, y = data.get('x'), data.get('y')
                        debug_print(f"[DataChannel] Processing mouse down at ({x}, {y})")
                        
                        # Start rectangle drawing
                        if hasattr(video_track, 'detector') and video_track.detector:
                            video_track.detector.handle_mouse_down(x, y)
                            
                        # Send confirmation back to client
                        channel.send(json.dumps({'type': 'mouse_down_response','x': x,'y': y,'status': 'processed'}))
                    elif data.get('type') == 'mouse_move':
                        x, y = data.get('x'), data.get('y')
                        debug_print(f"[DataChannel] Processing mouse move at ({x}, {y})")
                        
                        # Update current rectangle being drawn
                        if hasattr(video_track, 'detector') and video_track.detector:
                            video_track.detector.handle_mouse_move(x, y)
                    elif data.get('type') == 'mouse_up':
                        x, y = data.get('x'), data.get('y')
                        debug_print(f"[DataChannel] Processing mouse up at ({x}, {y})")
                        
                        # Complete rectangle drawing and trigger segmentation
                        if hasattr(video_track, 'detector') and video_track.detector:
                            video_track.detector.handle_mouse_up(x, y)
                            
                        # Send confirmation back to client
                        channel.send(json.dumps({'type': 'mouse_up_response','x': x,'y': y,'status': 'processed'}))
                    elif data.get('type') == 'switch_video_source':
                        debug_print(f"[DataChannel] Switching video source")
                        
                        # Switch video source
                        if hasattr(video_track, 'switch_video_source'):
                            source_name = video_track.switch_video_source()
                            
                            # Send confirmation back to client
                            channel.send(json.dumps({'type': 'video_source_switched','source': source_name,'status': 'success'}))
                        else:
                            response = {
                                'type': 'video_source_switched',
                                'status': 'error',
                                'message': 'Video track not available'
                            }
                            channel.send(json.dumps(response))
                    elif data.get('type') == 'settings_update':
                        settings = data.get('settings', {})
                        apply_settings_update(settings, video_track)
                        try:
                            channel.send(json.dumps({'type': 'settings_ack', 'settings': settings}))
                        except Exception:
                            pass
                    elif data.get('type') == 'text_query':
                        # Database-style text query to LLM; stream bulleted result to entity card and emit process trace
                        user_text = str(data.get('text') or '').strip()
                        channel.send(json.dumps({'type': 'text_query_ack'}))
                        if user_text:
                            try:
                                # Create a new entity card session for this query
                                if hasattr(video_track, 'detector') and video_track.detector and video_track.detector.segmentation_engine:
                                    seg = video_track.detector.segmentation_engine
                                    seg.start_new_input_session()
                                    entity_id = seg.current_entity_card_id or 0
                                    # Notify client: agent process started
                                    try:
                                        channel.send(json.dumps({'type': 'agent_trace', 'status': 'start'}))
                                    except Exception:
                                        pass
                                    # Kick off async LLM stream in the background
                                    async def _run_text_query():
                                        try:
                                            if hasattr(seg, 'llm_breaker') and not seg.llm_breaker.allow():
                                                seg.push_entity_card(entity_id, "ANALYSIS TEMPORARILY DISABLED (cooldown)", status='done')
                                                return
                                            # Database search agent: enable web access and enforce concise bullet list output
                                            db_system_instructions = (
                                                "You are a single Database Search agent with live web access enabled. "
                                                "Use web search as needed to retrieve up-to-date, authoritative information. "
                                                "Provide the best possible answer strictly as a concise bulleted list. "
                                                "Rules:\n"
                                                "- Start each item with '- ' (dash + space).\n"
                                                "- No headings, preamble, or closing text.\n"
                                                "- Focus on facts and high-signal guidance.\n"
                                                "- Combine duplicates; avoid redundancy.\n"
                                                "- Put each bullet on its own line; never chain multiple bullets on a single line.\n"
                                                "- Maximum 10 bullets.\n"
                                                "- Citations: include only the origin name in square brackets (e.g., [whitehouse.gov] or [Wikipedia]); never include URLs or parentheses around citations.\n"
                                                "- If nothing reliable is found, output '- No reliable information found.'"
                                            )
                                            title_prefix = "DB SEARCH RESULT\n"
                                            model = ModelPayLoad()
                                            def _emit_terminal_result(text: str) -> None:
                                                if RUN_MODE == "terminal":
                                                    print(f"\n[TEXT_QUERY {entity_id}]\n{text}\n", flush=True)
                                            if async_client and ('SUPPORTS_RESPONSES_API' in globals() and SUPPORTS_RESPONSES_API):
                                                full_text = ""
                                                # Sanitize streaming text to remove URLs and keep only [origin]
                                                def _sanitize_citations(text: str) -> str:
                                                    try:
                                                        # Replace markdown links [label](url) -> [label]
                                                        text = re.sub(r"\[([^\]]+)\]\((?:https?:)?//[^)]+\)", r"[\1]", text)
                                                        # Replace parenthesized URL or domain -> [domain]
                                                        text = re.sub(r"\((?:https?:\/\/)?(?:www\.)?([A-Za-z0-9.-]+)(?:\/[^)]*)?\)", r"[\1]", text)
                                                        # Remove any remaining raw URLs
                                                        text = re.sub(r"https?://\S+", "", text)
                                                        # Collapse extra spaces
                                                        text = re.sub(r"\s{2,}", " ", text)
                                                        return text
                                                    except Exception:
                                                        return text
                                                def _format_bulleted(text: str) -> str:
                                                    try:
                                                        # Normalize whitespace then insert line breaks before probable bullets
                                                        t = text.replace("\r", " ")
                                                        t = re.sub(r"\s+", " ", t)
                                                        t = re.sub(r"\s-\s+", "\n- ", t)
                                                        # Collect lines starting with '- '
                                                        bullets = [ln.strip() for ln in t.split("\n") if ln.strip().startswith("- ")]
                                                        if not bullets:
                                                            # Fallback: split into sentences and bullet them
                                                            sentences = re.split(r"(?<=[.!?])\s+", t.strip())
                                                            bullets = [f"- {s.strip()}" for s in sentences if s.strip()]
                                                        # Limit to 10
                                                        return "\n".join(bullets[:10])
                                                    except Exception:
                                                        return text
                                                async with async_client.responses.stream(
                                                    model=model,
                                                    input=[
                                                        {
                                                            "role":"user",
                                                            "content":[
                                                                {"type":"input_text","text": db_system_instructions},
                                                                {"type":"input_text","text": f"Query:\n{user_text}"}
                                                            ]
                                                        }
                                                    ],
                                                    tools=[{"type":"web_search"}],
                                                    temperature=0.2
                                                ) as stream:
                                                    # Client trace: planning
                                                    try:
                                                        channel.send(json.dumps({'type': 'agent_trace', 'event': 'step', 'text': 'Planning search...'}))
                                                    except Exception:
                                                        pass
                                                    async for event in stream:
                                                        et = getattr(event, 'type', '')
                                                        if et == 'response.tool_call.started':
                                                            try:
                                                                channel.send(json.dumps({'type': 'agent_trace', 'event': 'step', 'text': 'Searching web...'}))
                                                            except Exception:
                                                                pass
                                                        if et in ("response.output_text.delta", "response.delta"):
                                                            delta_text = getattr(event, 'delta', '') or ''
                                                            if delta_text:
                                                                full_text += delta_text
                                                                seg.push_entity_card(entity_id, title_prefix + _format_bulleted(_sanitize_citations(full_text)), status='streaming')
                                                try:
                                                    final = await stream.get_final_response()
                                                except Exception:
                                                    final = None
                                                formatted = title_prefix + _format_bulleted(_sanitize_citations(full_text))
                                                seg.push_entity_card(entity_id, formatted, status='done')
                                                _emit_terminal_result(formatted)
                                            elif async_client:
                                                # Sanitize helper for this branch as well
                                                def _sanitize_citations_cc(text: str) -> str:
                                                    try:
                                                        text = re.sub(r"\[([^\]]+)\]\((?:https?:)?//[^)]+\)", r"[\1]", text)
                                                        text = re.sub(r"\((?:https?:\/\/)?(?:www\.)?([A-Za-z0-9.-]+)(?:\/[^)]*)?\)", r"[\1]", text)
                                                        text = re.sub(r"https?://\S+", "", text)
                                                        text = re.sub(r"\s{2,}", " ", text)
                                                        return text
                                                    except Exception:
                                                        return text
                                                def _format_bulleted_cc(text: str) -> str:
                                                    try:
                                                        t = text.replace("\r", " ")
                                                        t = re.sub(r"\s+", " ", t)
                                                        t = re.sub(r"\s-\s+", "\n- ", t)
                                                        bullets = [ln.strip() for ln in t.split("\n") if ln.strip().startswith("- ")]
                                                        if not bullets:
                                                            sentences = re.split(r"(?<=[.!?])\s+", t.strip())
                                                            bullets = [f"- {s.strip()}" for s in sentences if s.strip()]
                                                        return "\n".join(bullets[:10])
                                                    except Exception:
                                                        return text
                                                stream = await async_client.chat.completions.create(
                                                    model=model,
                                                    messages=[
                                                        {"role":"system","content": db_system_instructions},
                                                        {"role":"user","content":[{"type":"text","text": f"Query:\n{user_text}"}]}
                                                    ],
                                                    stream=True,
                                                    stream_options={"include_usage": True},
                                                    temperature=0.2
                                                )
                                                try:
                                                    channel.send(json.dumps({'type': 'agent_trace', 'event': 'step', 'text': 'Generating answer...'}))
                                                except Exception:
                                                    pass
                                                full_text = ""
                                                async for chunk in stream:
                                                    if chunk.choices:
                                                        delta = chunk.choices[0].delta
                                                        piece = getattr(delta, 'content', None) or ''
                                                        if piece:
                                                            full_text += piece
                                                            seg.push_entity_card(entity_id, title_prefix + _format_bulleted_cc(_sanitize_citations_cc(full_text)), status='streaming')
                                                formatted = title_prefix + _format_bulleted_cc(_sanitize_citations_cc(full_text))
                                                seg.push_entity_card(entity_id, formatted, status='done')
                                                _emit_terminal_result(formatted)
                                            else:
                                                fallback_text = "LLM unavailable. Configure OpenAI API key."
                                                seg.push_entity_card(entity_id, fallback_text, status='done')
                                                _emit_terminal_result(fallback_text)
                                            if hasattr(seg, 'llm_breaker'):
                                                seg.llm_breaker.record_success()
                                            # Notify client: agent process done
                                            try:
                                                channel.send(json.dumps({'type': 'agent_trace', 'status': 'done'}))
                                            except Exception:
                                                pass
                                        except Exception as ee:
                                            debug_print(f"[TEXT_QUERY] Error: {ee}")
                                            try:
                                                error_text = "Error processing query."
                                                seg.push_entity_card(entity_id, error_text, status='done')
                                                _emit_terminal_result(error_text)
                                                if hasattr(seg, 'llm_breaker'):
                                                    seg.llm_breaker.record_failure()
                                            except Exception:
                                                pass
                                    # schedule on the same async loop used by segmentation engine
                                    try:
                                        future = asyncio.run_coroutine_threadsafe(_run_text_query(), seg.async_loop)
                                        seg.streaming_futures[entity_id] = future
                                    except Exception as loop_err:
                                        debug_print(f"[TEXT_QUERY] Scheduling error: {loop_err}")
                            except Exception as e2:
                                debug_print(f"[TEXT_QUERY] Failed to start: {e2}")
                        
                except json.JSONDecodeError:
                    # Handle non-JSON messages as regular text
                    if message.lower() in ('clear', 'c'):
                        # Clear segmentation data
                        if hasattr(video_track, 'detector') and video_track.detector and video_track.detector.segmentation_engine:
                            video_track.detector.segmentation_engine.clear()
                            channel.send("Segmentation data cleared")
                        else:
                            channel.send("No segmentation engine available")
                    else:
                        channel.send(f"Echo from server: {message}")

                except Exception as e:
                    debug_print(f"[DataChannel] Error processing message: {e}")
                    channel.send(f"Error processing message: {str(e)}")

        # Complete signaling handshake
        await pc.setRemoteDescription(offer)
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)
        debug_print("Answer created and local description set")
        debug_print(f"Local description SDP length: {len(pc.localDescription.sdp)}")

        return {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}
    
    except Exception as e:
        debug_print(f"Error in offer handler: {e}")
        import traceback
        traceback.print_exc()
        raise

@app.on_event("shutdown")
async def on_shutdown():
    debug_print("Shutting down, closing peer connections...")
    
    # Clean up video tracks first
    for track in video_tracks:
        track.cleanup()
    video_tracks.clear()
    
    # Close peer connections
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()
    debug_print("All connections closed")

def _parse_args():
    parser = argparse.ArgumentParser(description="WebRTC OpenCV server")
    parser.add_argument(
        "--mode",
        choices=["ui", "terminal"],
        default=RUN_MODE,
        help="Run mode: 'ui' shows LLM output in the browser; 'terminal' prints LLM output to the terminal."
    )
    return parser.parse_args()

if __name__ == '__main__':
    args = _parse_args()
    RUN_MODE = args.mode
    prompt_runtime_settings()
    debug_print(f"Starting WebRTC OpenCV server (mode: {RUN_MODE})...")
    uvicorn.run(app, host='0.0.0.0', port=8000)

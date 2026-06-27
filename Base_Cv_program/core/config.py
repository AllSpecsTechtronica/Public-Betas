# Configuration Constants for Modular CV System
# ═══════════════════════════════════════════════════════════════════════════════

import os

# Paths anchored to the cvLayer directory for self-contained assets.
BASE_CV_PROGRAM_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CVLAYER_DIR = os.path.abspath(os.path.join(BASE_CV_PROGRAM_DIR, ".."))
ASSETS_DIR = os.path.join(CVLAYER_DIR, "assets")
MODELS_DIR = os.path.join(ASSETS_DIR, "models")
VIDEOS_DIR = os.path.join(ASSETS_DIR, "videos")

# Global Debug Configuration
DEBUG_MODE = True  # Set to False to disable all print statements
ENHANCED_DEBUG = False  # Additional debug info for segmentation issues
VERBOSE_FRAME_LOGGING = False  # Set to True to enable detailed frame stats every 2 seconds

# Double-click Detection Constants
DOUBLE_CLICK_BASE_WIDTH = 420
DOUBLE_CLICK_BASE_HEIGHT = 420

# Detection Constants
MODEL_PATH = os.path.join(MODELS_DIR, "yolo11n_Humans.pt")
IMG_SIZE = 640
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

# Entity Card UI
ENTITY_CARD_FONT_SIZE_PX = 11

# Pose Estimation Options
ENABLE_POSE_ESTIMATION = False
POSE_MIN_BBOX_SIZE = 100

# Priority Classes
USE_PRIORITY_CLASSES = False
PRIORITY_CLASSES = {"person", "car"}
PRIORITY_CONF = 0.10

# Class Categories for Object Classification
CLASS_CATEGORIES = {
    "person": "Humans", "cat": "Pets", "dog": "Pets", "bird": "Wild Animals", 
    "horse": "Wild Animals", "sheep": "Wild Animals", "cow": "Wild Animals", 
    "elephant": "Wild Animals", "bear": "Wild Animals", "zebra": "Wild Animals", 
    "giraffe": "Wild Animals", "potted plant": "Plants", "bicycle": "Vehicles", 
    "car": "Vehicles", "motorcycle": "Vehicles", "airplane": "Vehicles", 
    "bus": "Vehicles", "train": "Vehicles", "truck": "Vehicles", "boat": "Vehicles", 
    "tv": "Technology", "laptop": "Technology", "mouse": "Technology", 
    "remote": "Technology", "keyboard": "Technology", "cell phone": "Technology", 
    "microwave": "Appliances", "oven": "Appliances", "toaster": "Appliances", 
    "refrigerator": "Appliances", "sink": "Plumbing Fixtures", "toilet": "Plumbing Fixtures", 
    "bench": "Furniture", "chair": "Furniture", "couch": "Furniture", "bed": "Furniture", 
    "dining table": "Furniture", "book": "Furniture", "clock": "Furniture", 
    "vase": "Furniture", "bottle": "Kitchenware", "wine glass": "Kitchenware", 
    "cup": "Kitchenware", "fork": "Kitchenware", "knife": "Kitchenware", 
    "spoon": "Kitchenware", "bowl": "Kitchenware", "banana": "Food", "apple": "Food", 
    "sandwich": "Food", "orange": "Food", "broccoli": "Food", "carrot": "Food", 
    "hot dog": "Food", "pizza": "Food", "donut": "Food", "cake": "Food", 
    "frisbee": "Sports", "skis": "Sports", "snowboard": "Sports", "sports ball": "Sports", 
    "kite": "Sports", "baseball bat": "Sports", "baseball glove": "Sports", 
    "skateboard": "Sports", "surfboard": "Sports", "tennis racket": "Sports", 
    "backpack": "Accessories", "umbrella": "Accessories", "handbag": "Accessories", 
    "tie": "Accessories", "suitcase": "Accessories", "scissors": "Accessories", 
    "teddy bear": "Accessories", "hair drier": "Accessories", "toothbrush": "Accessories", 
    "traffic light": "Urban Infrastructure (Electronic)", 
    "parking meter": "Urban Infrastructure (Electronic)", 
    "fire hydrant": "Urban Infrastructure (Static)", 
    "stop sign": "Urban Infrastructure (Static)"
}

TECH_CATEGORIES = {"Vehicles", "Technology", "Appliances", "Urban Infrastructure (Electronic)"}

# Helper Functions
def get_category(name: str) -> str:
    """Get the category for a given object name."""
    return CLASS_CATEGORIES.get(name, "Unknown")

def get_type(name: str) -> str:
    """Get the type (Tech/Organic) for a given object name."""
    return "Tech" if get_category(name) in TECH_CATEGORIES or name == "hair drier" else "Organic"

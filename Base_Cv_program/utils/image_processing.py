# Image Processing Utilities
# ═══════════════════════════════════════════════════════════════════════════════

import cv2
import base64
import numpy as np
from utils.debug import debug_print

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
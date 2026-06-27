# Debug Utility Functions
# ═══════════════════════════════════════════════════════════════════════════════

from core.config import DEBUG_MODE, ENHANCED_DEBUG

def debug_print(*args, **kwargs):
    """A wrapper for print() that only prints if DEBUG_MODE is True."""
    if DEBUG_MODE:
        print(*args, **kwargs)

def enhanced_debug_print(*args, **kwargs):
    """Enhanced debug print for segmentation issues."""
    if DEBUG_MODE and ENHANCED_DEBUG:
        print("[ENHANCED_DEBUG]", *args, **kwargs)

class DebugLogger:
    """Enhanced debug logging with categorization and filtering capabilities."""
    
    def __init__(self, enabled=True):
        self.enabled = enabled and DEBUG_MODE
        self.debug_categories = {
            'detection': True,
            'tracking': True,
            'pose': True,
            'ui': True,
            'session': True,
            'ai': True
        }
    
    def log(self, category, message, *args, **kwargs):
        """Log message if category is enabled."""
        if not self.enabled or not self.debug_categories.get(category, False):
            return
        
        prefix = f"[DEBUG_{category.upper()}]"
        print(prefix, message, *args, **kwargs)
    
    def enable_category(self, category):
        """Enable logging for specific category."""
        if category in self.debug_categories:
            self.debug_categories[category] = True
    
    def disable_category(self, category):
        """Disable logging for specific category."""
        if category in self.debug_categories:
            self.debug_categories[category] = False
    
    def toggle_category(self, category):
        """Toggle logging for specific category."""
        if category in self.debug_categories:
            self.debug_categories[category] = not self.debug_categories[category]

# Global debug logger instance
debug_logger = DebugLogger()
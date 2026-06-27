# Modular PyQt Computer Vision System

This is a modular PyQt-based computer vision system converted from the original monolithic `cv.py` web-based implementation. The system provides real-time object detection, tracking, and AI analysis through a native desktop interface.

## Features

- **YOLO Object Detection**: Real-time object detection using Ultralytics YOLO
- **Object Tracking**: Kalman filter-based tracking with highlight system
- **PyQt5 GUI**: Native desktop interface replacing web-based GUI
- **AI Integration**: OpenAI integration for image analysis (optional)
- **Session Tracking**: Cost monitoring and usage tracking
- **Pose Estimation**: Human pose estimation support
- **Modular Architecture**: Clean separation of concerns for maintainability

## Architecture

```
Base_Cv_program/
├── main.py                    # Standalone application entry point
├── core/
│   ├── config.py              # Configuration constants
│   ├── controller.py          # Main application controller
│   └── session_tracker.py     # Session tracking and billing
├── detection/
│   ├── yolo_detector.py       # YOLO object detection
│   └── trackers.py            # Kalman & Highlight tracking
├── ai_integration/
│   └── openai_client.py       # OpenAI integration
├── gui/
│   ├── main_window.py         # Main PyQt window
│   ├── video_widget.py        # Video display widget
│   └── entity_panel.py        # AI analysis results panel
└── utils/
    ├── debug.py               # Debug utilities
    └── image_processing.py    # Image processing helpers
```

## Installation

1. **Install Python Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Install PyQt5** (if not already installed):
   ```bash
   pip install PyQt5
   ```

3. **Download YOLO Model**:
   - Place your YOLO model file in the specified path or update the `MODEL_PATH` in `core/config.py`
   - Default path: `../assets/models/yolo11n_Humans.pt` (relative to `Base_Cv_program/`)

4. **Configure OpenAI** (optional):
   - Update your API key in the parent directory's `LibBinaries.py`
   - Or disable AI integration in the configuration

## Usage

### Quick Start

```bash
# Activate your virtual environment first
source /path/to/your/venv/bin/activate

# Navigate to the application directory  
cd /path/to/Base_Cv_program

# Launch with the convenient launcher script
./launch.sh

# Or run directly with Python
python main.py
```

### Command Line Options

```bash
# Run with default settings (camera 0)
python main.py

# Use a different camera
python main.py --camera 1

# Use a custom YOLO model
python main.py --model path/to/your/model.pt

# Enable debug mode
python main.py --debug

# Show help
python main.py --help
```

### Alternative Launchers

```bash
# Simple launcher with dependency checks
./launch.sh --debug

# Test system components
python test_system.py

# Setup dependencies
./setup.sh
```

### GUI Controls

- **Start/Stop Detection**: Toggle object detection on/off
- **Camera Selection**: Choose which camera to use
- **Detection Settings**: Adjust confidence and IOU thresholds
- **Display Options**: Toggle visualization elements
- **Mouse Interaction**: Click and drag to select regions for AI analysis

### Key Differences from Original

| Original (cv.py) | Modular (Base_Cv_program) |
|------------------|---------------------------|
| Web-based GUI (FastAPI + HTML) | Native PyQt5 desktop GUI |
| Monolithic 56k+ lines | Modular architecture |
| WebRTC video streaming | Direct OpenCV video capture |
| Single file | Multiple organized modules |
| Hard to maintain | Easy to extend and modify |

## Configuration

Main configuration options are in `core/config.py`:

```python
# Detection settings
MODEL_PATH = "path/to/your/model.pt"
CONF = 0.25          # Confidence threshold
IOU = 0.10           # IOU threshold
IMG_SIZE = 320       # Input image size

# Display options
SHOW_OBJECT_TITLES = False
SHOW_CONFIDENCE_SCORES = False
SHOW_DETECTION_ARROWS = True

# Pose estimation
ENABLE_POSE_ESTIMATION = False
```

## Dependencies

- **PyQt5**: Desktop GUI framework
- **OpenCV**: Computer vision operations
- **NumPy**: Numerical computations
- **Ultralytics**: YOLO implementation
- **OpenAI**: AI analysis (optional)
- **PyTorch**: Deep learning backend

## Troubleshooting

### Common Issues

1. **Camera not opening**:
   - Try different camera indices (0, 1, 2, etc.)
   - Check camera permissions
   - Ensure camera is not in use by another application

2. **Model file not found**:
   - Verify the model path in configuration
   - Download the required YOLO model
   - Use `--model` argument to specify custom path

3. **PyQt5 installation issues**:
   - On macOS: `brew install pyqt5`
   - On Ubuntu: `sudo apt-get install python3-pyqt5`
   - On Windows: Use pip or conda

4. **AI integration not working**:
   - Check OpenAI API key configuration
   - Verify internet connection
   - Check API usage limits

## Performance Optimization

- **GPU Acceleration**: Ensure PyTorch CUDA is available for GPU processing
- **Frame Rate**: Adjust `inference_interval` in detector for better performance
- **Resolution**: Lower camera resolution for faster processing
- **Model Size**: Use smaller YOLO models (yolo11n vs yolo11x) for speed

## Development

### Adding New Features

1. **New Detection Algorithm**: Add to `detection/` module
2. **Custom GUI Components**: Add to `gui/` module  
3. **AI Integration**: Extend `ai_integration/` module
4. **Configuration Options**: Update `core/config.py`

### Code Style

- Follow PEP 8 conventions
- Use type hints where possible
- Add proper error handling
- Include debug logging
- Document public methods

## License

This modular system maintains the same functionality as the original cv.py while providing better organization, maintainability, and a native desktop experience.

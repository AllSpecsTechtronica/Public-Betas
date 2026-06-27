# Quick Start Guide

## ✅ Your Modular CV System is Ready!

The monolithic `cv.py` has been successfully converted to a modular PyQt desktop application.

### 🚀 How to Run

```bash
# 1. Activate your virtual environment (example uses the local .venv in cvLayer)
source ../.venv/bin/activate

# 2. Navigate to the application directory
cd /path/to/cvLayer/Base_Cv_program

# 3. Run the application
python main.py --debug
```

### 🎯 Alternative Methods

**Using the launcher script** (recommended):
```bash
./launch.sh --debug
```

**Test the system first**:
```bash
python test_system.py
```

### 📋 What You'll Get

- **Native PyQt5 desktop interface** (no more web browser)
- **Real-time video processing** with YOLO object detection
- **Mouse interaction** for region selection and analysis
- **AI integration** for image analysis (if OpenAI configured)
- **Session tracking** for monitoring usage and costs
- **Modular architecture** for easy maintenance and extension

### 🔧 If You Get a Model Error

The system will try to download a default YOLO model automatically. If that fails:

```bash
# Use a different model
python main.py --model yolo11n.pt --debug
```

### 🎮 Using the GUI

1. **Start Detection**: Click the "Start Detection" button
2. **Select Camera**: Choose camera index from dropdown
3. **Adjust Settings**: Use sliders for confidence/IOU thresholds  
4. **Interact**: Click and drag on video to select regions for analysis
5. **View Results**: AI analysis results appear in the right panel

### 🆘 Troubleshooting

**Import errors**: Make sure you're in the correct directory
**Camera issues**: Try different camera indices (0, 1, 2, etc.)
**Model issues**: The system will auto-download fallback models
**GUI issues**: Ensure PyQt5 is installed: `pip install PyQt5`

### 🎉 Success!

Your modular PyQt CV system is now running with all the same capabilities as the original monolithic web-based system, but with better organization and a native desktop interface!

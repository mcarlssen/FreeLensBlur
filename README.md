# Depth Blur Filter

A Python desktop application that applies realistic depth-of-field blur effects to JPEG images using AI-generated depth maps.

## Features

- Click-to-focus interface for setting focal points
- Adjustable blur strength with live preview
- GPU/CPU acceleration support
- Handles high-resolution images (7-10MP)
- Realistic depth-of-field effects

## Installation

1. Install Python 3.8 or higher
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Run the application:
   ```bash
   python main.py
   ```

2. Open a JPEG image using File → Open or drag-and-drop
3. Click on the preview to set the focal point
4. Adjust blur strength using the slider
5. Save the result using File → Save

## Requirements

- Python 3.8+
- CUDA-compatible GPU (optional, for faster processing)
- ~400MB disk space for MiDaS model weights (downloaded on first use)

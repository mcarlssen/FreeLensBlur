"""
Main GUI application for the depth blur filter.
PyQt5-based interface with live preview and interactive controls.
"""

import sys
import os
from typing import Optional, Tuple
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QLabel, QSlider, QPushButton, QFileDialog,
                            QMessageBox, QProgressBar, QStatusBar, QMenuBar, QAction,
                            QSplitter, QFrame)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QSize
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen, QColor, QFont
import cv2

from depth_estimator import DepthEstimator
from blur_processor import BlurProcessor
from utils.image_loader import load_jpeg_image, save_jpeg_image, resize_for_preview
from preferences import PreferencesManager


class ImageDisplayWidget(QLabel):
    """Custom widget for displaying images with click handling."""
    
    clicked = pyqtSignal(int, int)  # Signal emitted when image is clicked
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(400, 300)
        self.setStyleSheet("border: 1px solid gray; background-color: #f0f0f0;")
        self.setAlignment(Qt.AlignCenter)
        self.setScaledContents(False)  # Don't stretch to fill widget
        self.focal_point = None
        self.scale_factor = 1.0
        self.image_offset_x = 0
        self.image_offset_y = 0
        self.original_image_size = None
        self.preview_image_size = None
    
    def mousePressEvent(self, event):
        """Handle mouse click events."""
        if event.button() == Qt.LeftButton and self.preview_image_size is not None:
            click_x = event.x()
            click_y = event.y()
            
            preview_width, preview_height = self.preview_image_size
            scaled_width = int(preview_width * self.scale_factor)
            scaled_height = int(preview_height * self.scale_factor)
            
            image_left = self.image_offset_x
            image_right = self.image_offset_x + scaled_width
            image_top = self.image_offset_y
            image_bottom = self.image_offset_y + scaled_height
            
            if (image_left <= click_x <= image_right and image_top <= click_y <= image_bottom):
                relative_x = click_x - image_left
                relative_y = click_y - image_top
                
                preview_x = int(relative_x / self.scale_factor)
                preview_y = int(relative_y / self.scale_factor)
                
                preview_x = max(0, min(preview_x, preview_width - 1))
                preview_y = max(0, min(preview_y, preview_height - 1))
                
                orig_width, orig_height = self.original_image_size
                scale_x = orig_width / preview_width
                scale_y = orig_height / preview_height
                
                image_x = int(preview_x * scale_x)
                image_y = int(preview_y * scale_y)
                
                image_x = max(0, min(image_x, orig_width - 1))
                image_y = max(0, min(image_y, orig_height - 1))
                
                self.focal_point = (image_x, image_y)
                self.clicked.emit(image_x, image_y)
                self.update()
    
    def set_image(self, image_array: np.ndarray):
        """Set the image to display."""
        try:
            if image_array is None:
                self.clear()
                self.original_image_size = None
                self.preview_image_size = None
                return
            
            height, width, channel = image_array.shape
            self.preview_image_size = (width, height)
            
            bytes_per_line = 3 * width
            q_image = QImage(image_array.data, width, height, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image)
            
            self._recalculate_image_positioning()
            
            preview_width, preview_height = self.preview_image_size
            scaled_width = int(preview_width * self.scale_factor)
            scaled_height = int(preview_height * self.scale_factor)
            
            scaled_pixmap = pixmap.scaled(scaled_width, scaled_height, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.setPixmap(scaled_pixmap)
        except Exception as e:
            print(f"Error in set_image: {e}")
    
    def set_original_image_size(self, width: int, height: int):
        """Set the original full-size image dimensions."""
        self.original_image_size = (width, height)
    
    def paintEvent(self, event):
        super().paintEvent(event)
        
        # Draw focal point indicator
        if self.focal_point is not None and self.original_image_size is not None:
            painter = QPainter(self)
            painter.setPen(QPen(QColor(255, 0, 0), 3))
            
            # Convert original image coordinates back to preview coordinates for display
            orig_x, orig_y = self.focal_point
            orig_width, orig_height = self.original_image_size
            preview_width, preview_height = self.preview_image_size
            
            scale_x = preview_width / orig_width
            scale_y = preview_height / orig_height
            
            preview_x = orig_x * scale_x
            preview_y = orig_y * scale_y
            
            # Calculate display coordinates
            display_x = int(self.image_offset_x + preview_x * self.scale_factor)
            display_y = int(self.image_offset_y + preview_y * self.scale_factor)
            
            # Draw circle
            painter.drawEllipse(display_x - 10, display_y - 10, 20, 20)
            painter.drawLine(display_x - 15, display_y, display_x + 15, display_y)
            painter.drawLine(display_x, display_y - 15, display_x, display_y + 15)
    
    def resizeEvent(self, event):
        """Handle widget resize to recalculate image positioning."""
        super().resizeEvent(event)
        # Recalculate image positioning if we have an image
        if self.preview_image_size is not None:
            # Recalculate scale factor and offsets
            self._recalculate_image_positioning()
            # Trigger a repaint to update focal point position
            self.update()
    
    def _recalculate_image_positioning(self):
        """Recalculate image positioning after resize."""
        if self.preview_image_size is None:
            return
        
        preview_width, preview_height = self.preview_image_size
        
        # Calculate scale factor to fit image in widget while preserving aspect ratio
        widget_size = self.size()
        
        scale_x = widget_size.width() / preview_width
        scale_y = widget_size.height() / preview_height
        self.scale_factor = min(scale_x, scale_y)  # Use min to fit within widget
        
        # Calculate scaled image size
        scaled_width = int(preview_width * self.scale_factor)
        scaled_height = int(preview_height * self.scale_factor)
        
        # Calculate offset to center the image
        self.image_offset_x = (widget_size.width() - scaled_width) // 2
        self.image_offset_y = (widget_size.height() - scaled_height) // 2


class ProcessingThread(QThread):
    """Thread for processing depth estimation and blur application."""
    
    # Signals for communication with main thread
    depth_completed = pyqtSignal(np.ndarray)  # depth map
    blur_completed = pyqtSignal(np.ndarray)  # blurred image
    
    def __init__(self, parent=None):
        super().__init__(parent)
        # Initialize with default values - will be updated from preferences
        self.depth_estimator = DepthEstimator(
            detail_level="high",
            hole_filling_enabled=True,
            background_correction_enabled=True,
            smoothing_strength="medium"
        )
        self.blur_processor = BlurProcessor()
        self.current_image = None
        self.depth_map = None
        self.focal_point = None
        self.blurred_result = None
        self.focal_length = 50.0
        self.force_cpu_mode = False  # Flag to force CPU processing
        
    def has_blur_result(self):
        """Check if blur processing has completed."""
        return self.blurred_result is not None
    
    def get_blur_result(self):
        """Get the blur result and clear it."""
        result = self.blurred_result
        self.blurred_result = None
        return result
        
    def estimate_depth(self, image_array: np.ndarray):
        """Estimate depth map for the image."""
        print(f"🔄 ProcessingThread: Starting depth estimation for new image")
        self.current_image = image_array
        self.depth_map = None  # Clear previous depth map
        self.focal_point = None
        self.blurred_result = None  # Clear previous blur result
        print(f"✅ ProcessingThread: Cleared previous depth_map and blur_result")
        self.start()
    
    def apply_blur(self, focal_point: Tuple[int, int], blur_strength: float, focal_length: float = 50.0):
        """Apply blur effect with given parameters."""
        if self.isRunning():
            self.terminate()
        self.wait(1000)
        
        self.focal_point = focal_point
        self.blur_strength = blur_strength
        self.focal_length = focal_length
        self.start()
    
    def run(self):
        """Main thread execution."""
        try:
            if self.current_image is None:
                return
            
            # Estimate depth if not already done
            if self.depth_map is None:
                print(f"🔄 ProcessingThread: Generating new depth map")
                self.depth_map = self.depth_estimator.estimate_depth(self.current_image)
                # Signal that depth estimation is complete
                self.depth_completed.emit(self.depth_map)
            else:
                print(f"⚠️ ProcessingThread: Skipping depth estimation - depth_map already exists")
                print(f"   This should not happen for new images!")
            
            # Apply blur if focal point is set
            if self.focal_point is not None:
                # Force CPU mode for blur processing if requested
                if self.force_cpu_mode:
                    print("🖥️ Forcing CPU mode for blur processing")
                    blurred = self.blur_processor.apply_focal_blur_simple(
                        self.current_image, self.depth_map, 
                        self.focal_point, self.blur_strength, self.focal_length
                    )
                else:
                    blurred = self.blur_processor.apply_focal_blur_gpu(
                        self.current_image, self.depth_map, 
                        self.focal_point, self.blur_strength, self.focal_length
                    )
                
                self.blurred_result = blurred
                # Signal that blur processing is complete
                self.blur_completed.emit(blurred)
                
        except Exception as e:
            print(f"Error in ProcessingThread.run(): {e}")
            import traceback
            traceback.print_exc()


class DepthBlurApp(QMainWindow):
    """Main application window."""
    
    def __init__(self):
        super().__init__()
        self.current_image = None
        self.depth_map = None
        self.blurred_image = None
        self.processing_thread = None
        self.blur_timer = QTimer()
        self.blur_timer.setSingleShot(True)
        self.blur_timer.timeout.connect(self._apply_blur_debounced)
        self.dark_mode = False
        self.showing_depth_map = False  # Track if we're showing depth map
        self.focal_length = 50.0  # Default focal length in mm
        
        # Initialize preferences manager
        self.preferences = PreferencesManager()
        
        self.init_ui()
        self.setup_processing_thread()
        self.load_preferences()
    
    def update_depth_settings(self, hole_filling_enabled: bool = None, 
                            background_correction_enabled: bool = None,
                            smoothing_strength: str = None):
        """
        Update depth estimator settings.
        
        Args:
            hole_filling_enabled: Whether to enable hole filling
            background_correction_enabled: Whether to enable background correction
            smoothing_strength: Smoothing strength ("low", "medium", "high")
        """
        if hole_filling_enabled is not None:
            self.processing_thread.depth_estimator.hole_filling_enabled = hole_filling_enabled
        if background_correction_enabled is not None:
            self.processing_thread.depth_estimator.background_correction_enabled = background_correction_enabled
        if smoothing_strength is not None:
            self.processing_thread.depth_estimator.smoothing_strength = smoothing_strength
        
        # Clear depth cache since settings changed
        self.processing_thread.depth_estimator.clear_cache()
        
        # If we have a current image, regenerate the depth map
        if self.current_image is not None:
            print(f"🔄 Regenerating depth map with updated settings...")
            self.status_bar.showMessage("Regenerating depth map with updated settings...")
            self.depth_map = None  # Clear cached depth map
            self.blurred_image = None  # Clear blurred result
            self.progress_bar.setVisible(True)
            self.progress_bar.setRange(0, 0)  # Indeterminate progress
            self.processing_thread.estimate_depth(self.current_image)
    
    def init_ui(self):
        """Initialize the user interface."""
        self.setWindowTitle("Depth Blur Filter")
        self.setGeometry(100, 100, 1200, 800)
        
        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Create main layout
        main_layout = QVBoxLayout(central_widget)
        
        # Create menu bar
        self.create_menu_bar()
        
        # Create image display area
        self.create_image_display(main_layout)
        
        # Create controls
        self.create_controls(main_layout)
        
        # Create status bar
        self.create_status_bar()
    
    def create_menu_bar(self):
        """Create the menu bar."""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu('File')
        
        open_action = QAction('Open', self)
        open_action.setShortcut('Ctrl+O')
        open_action.triggered.connect(self.open_image)
        file_menu.addAction(open_action)
        
        save_action = QAction('Save', self)
        save_action.setShortcut('Ctrl+S')
        save_action.triggered.connect(self.save_image)
        file_menu.addAction(save_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction('Exit', self)
        exit_action.setShortcut('Ctrl+Q')
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # View menu
        view_menu = menubar.addMenu('View')
        
        self.dark_mode_action = QAction('Dark Mode', self)
        self.dark_mode_action.setCheckable(True)
        self.dark_mode_action.setShortcut('Ctrl+D')
        self.dark_mode_action.triggered.connect(self.toggle_dark_mode)
        view_menu.addAction(self.dark_mode_action)
        
        view_menu.addSeparator()
        
        self.depth_map_action = QAction('Show Depth Map', self)
        self.depth_map_action.setCheckable(True)
        self.depth_map_action.setShortcut('Ctrl+M')
        self.depth_map_action.triggered.connect(self.toggle_depth_map)
        view_menu.addAction(self.depth_map_action)
        
        view_menu.addSeparator()
        
        # Depth detail submenu
        detail_menu = view_menu.addMenu('Depth Detail Level')
        
        self.detail_high_action = QAction('High Detail', self)
        self.detail_high_action.setCheckable(True)
        self.detail_high_action.setChecked(True)  # Default to high detail
        self.detail_high_action.triggered.connect(lambda: self.set_detail_level("high"))
        detail_menu.addAction(self.detail_high_action)
        
        self.detail_medium_action = QAction('Medium Detail', self)
        self.detail_medium_action.setCheckable(True)
        self.detail_medium_action.triggered.connect(lambda: self.set_detail_level("medium"))
        detail_menu.addAction(self.detail_medium_action)
        
        self.detail_low_action = QAction('Low Detail', self)
        self.detail_low_action.setCheckable(True)
        self.detail_low_action.triggered.connect(lambda: self.set_detail_level("low"))
        detail_menu.addAction(self.detail_low_action)
        
        view_menu.addSeparator()
        
        # CPU mode toggle
        self.cpu_mode_action = QAction('Force CPU Mode', self)
        self.cpu_mode_action.setCheckable(True)
        self.cpu_mode_action.setShortcut('Ctrl+C')
        self.cpu_mode_action.triggered.connect(self.toggle_cpu_mode)
        view_menu.addAction(self.cpu_mode_action)
        
        view_menu.addSeparator()
        
        # Depth processing submenu
        depth_menu = view_menu.addMenu('Depth Processing')
        
        # Hole filling toggle
        self.hole_filling_action = QAction('Enable Hole Filling', self)
        self.hole_filling_action.setCheckable(True)
        self.hole_filling_action.setChecked(True)  # Default enabled
        self.hole_filling_action.triggered.connect(self.toggle_hole_filling)
        depth_menu.addAction(self.hole_filling_action)
        
        # Background correction toggle
        self.background_correction_action = QAction('Enable Background Correction', self)
        self.background_correction_action.setCheckable(True)
        self.background_correction_action.setChecked(True)  # Default enabled
        self.background_correction_action.triggered.connect(self.toggle_background_correction)
        depth_menu.addAction(self.background_correction_action)
        
        depth_menu.addSeparator()
        
        # Smoothing strength submenu
        smoothing_menu = depth_menu.addMenu('Smoothing Strength')
        
        self.smoothing_low_action = QAction('Low Smoothing', self)
        self.smoothing_low_action.setCheckable(True)
        self.smoothing_low_action.triggered.connect(lambda: self.set_smoothing_strength("low"))
        smoothing_menu.addAction(self.smoothing_low_action)
        
        self.smoothing_medium_action = QAction('Medium Smoothing', self)
        self.smoothing_medium_action.setCheckable(True)
        self.smoothing_medium_action.setChecked(True)  # Default
        self.smoothing_medium_action.triggered.connect(lambda: self.set_smoothing_strength("medium"))
        smoothing_menu.addAction(self.smoothing_medium_action)
        
        self.smoothing_high_action = QAction('High Smoothing', self)
        self.smoothing_high_action.setCheckable(True)
        self.smoothing_high_action.triggered.connect(lambda: self.set_smoothing_strength("high"))
        smoothing_menu.addAction(self.smoothing_high_action)
    
    def create_image_display(self, parent_layout):
        """Create the image display area."""
        # Create single preview panel (removed original view)
        preview_frame = QFrame()
        preview_frame.setFrameStyle(QFrame.StyledPanel)
        preview_layout = QVBoxLayout(preview_frame)
        
        preview_label = QLabel("Preview (Click to set focal point)")
        preview_label.setAlignment(Qt.AlignCenter)
        preview_label.setFont(QFont("Arial", 12, QFont.Bold))
        preview_label.setStyleSheet("color: #0066cc; padding: 4px;")
        preview_label.setMaximumHeight(30)  # Fixed height
        self.preview_label = preview_label  # Store reference for updates
        preview_layout.addWidget(preview_label)
        
        self.preview_display = ImageDisplayWidget()
        self.preview_display.clicked.connect(self.on_focal_point_clicked)
        self.preview_display.setStyleSheet("border: 2px solid #0066cc; background-color: #f0f0f0;")
        preview_layout.addWidget(self.preview_display)
        
        parent_layout.addWidget(preview_frame)
    
    def create_controls(self, parent_layout):
        """Create the control panel."""
        controls_layout = QHBoxLayout()
        
        # Blur strength label - compact
        blur_label = QLabel("Blur Strength:")
        blur_label.setMaximumHeight(25)  # Fixed height
        blur_label.setStyleSheet("padding: 2px;")
        controls_layout.addWidget(blur_label)
        
        # Blur strength slider - compact
        self.blur_slider = QSlider(Qt.Horizontal)
        self.blur_slider.setMinimum(0)
        self.blur_slider.setMaximum(10)
        self.blur_slider.setValue(1)
        self.blur_slider.setMaximumHeight(25)  # Fixed height
        self.blur_slider.valueChanged.connect(self.on_blur_strength_value_changed)
        self.blur_slider.sliderReleased.connect(self.on_blur_strength_released)
        controls_layout.addWidget(self.blur_slider)
        
        # Blur value label - compact
        self.blur_value_label = QLabel("1.0")
        self.blur_value_label.setMinimumWidth(30)
        self.blur_value_label.setMaximumHeight(25)  # Fixed height
        self.blur_value_label.setStyleSheet("padding: 2px;")
        controls_layout.addWidget(self.blur_value_label)
        
        # Add some spacing
        controls_layout.addSpacing(20)
        
        # Focal length slider
        focal_label = QLabel("Focal Length:")
        focal_label.setMaximumHeight(25)  # Fixed height
        focal_label.setStyleSheet("padding: 2px;")
        controls_layout.addWidget(focal_label)
        
        self.focal_slider = QSlider(Qt.Horizontal)
        self.focal_slider.setMinimum(24)  # Wide angle
        self.focal_slider.setMaximum(200)  # Telephoto
        self.focal_slider.setValue(50)  # Normal lens
        self.focal_slider.setMaximumHeight(25)  # Fixed height
        self.focal_slider.valueChanged.connect(self.on_focal_length_value_changed)
        self.focal_slider.sliderReleased.connect(self.on_focal_length_released)
        controls_layout.addWidget(self.focal_slider)
        
        self.focal_value_label = QLabel("50mm")
        self.focal_value_label.setMinimumWidth(40)
        self.focal_value_label.setMaximumHeight(25)  # Fixed height
        self.focal_value_label.setStyleSheet("padding: 2px;")
        controls_layout.addWidget(self.focal_value_label)
        
        controls_layout.addStretch()
        
        # Progress bar - compact
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setMinimumWidth(200)
        self.progress_bar.setMaximumHeight(25)  # Fixed height
        controls_layout.addWidget(self.progress_bar)
        
        parent_layout.addLayout(controls_layout)
    
    def create_status_bar(self):
        """Create the status bar."""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")
        
        # Add processing mode indicator
        self.mode_label = QLabel("GPU")
        self.mode_label.setStyleSheet("color: #0078d4; font-weight: bold; padding: 2px 8px;")
        self.status_bar.addPermanentWidget(self.mode_label)
    
    def setup_processing_thread(self):
        """Setup the processing thread."""
        self.processing_thread = ProcessingThread()
        # Connect signals
        self.processing_thread.depth_completed.connect(self.on_depth_completed)
        self.processing_thread.blur_completed.connect(self.on_blur_completed)
    
    def load_preferences(self):
        """Load preferences and apply them to the UI."""
        print("📁 Loading application preferences...")
        
        # Load dark mode preference
        dark_mode = self.preferences.get_preference('ui', 'dark_mode', False)
        if dark_mode != self.dark_mode:
            self.dark_mode = dark_mode
            self.dark_mode_action.setChecked(dark_mode)
            self.apply_theme()
            print(f"🌙 Dark mode: {'enabled' if dark_mode else 'disabled'}")
        
        # Load CPU mode preference
        cpu_mode = self.preferences.get_preference('processing', 'cpu_mode', False)
        if cpu_mode:
            self.cpu_mode_action.setChecked(True)
            self.toggle_cpu_mode()
            print(f"🖥️ CPU mode: {'enabled' if cpu_mode else 'disabled'}")
        
        # Load detail level preference
        detail_level = self.preferences.get_preference('processing', 'detail_level', 'high')
        self.set_detail_level(detail_level)
        print(f"🎨 Detail level: {detail_level}")
        
        # Load depth processing preferences
        hole_filling_enabled = self.preferences.get_preference('depth', 'hole_filling_enabled', True)
        background_correction_enabled = self.preferences.get_preference('depth', 'background_correction_enabled', True)
        smoothing_strength = self.preferences.get_preference('depth', 'smoothing_strength', 'medium')
        
        self.hole_filling_action.setChecked(hole_filling_enabled)
        self.background_correction_action.setChecked(background_correction_enabled)
        
        # Update smoothing strength checkboxes
        self.smoothing_low_action.setChecked(smoothing_strength == "low")
        self.smoothing_medium_action.setChecked(smoothing_strength == "medium")
        self.smoothing_high_action.setChecked(smoothing_strength == "high")
        
        # Update depth estimator settings
        self.update_depth_settings(hole_filling_enabled, background_correction_enabled, smoothing_strength)
        print(f"🔧 Depth processing: hole_filling={hole_filling_enabled}, background_correction={background_correction_enabled}, smoothing={smoothing_strength}")
        
        # Load blur settings
        blur_strength = self.preferences.get_preference('blur', 'blur_strength', 1)
        focal_length = self.preferences.get_preference('blur', 'focal_length', 50)
        
        self.blur_slider.setValue(blur_strength)
        self.blur_value_label.setText(f"{blur_strength}.0")
        
        self.focal_slider.setValue(focal_length)
        self.focal_length = float(focal_length)
        self.focal_value_label.setText(f"{focal_length}mm")
        
        print(f"🎚️ Blur settings: strength={blur_strength}, focal_length={focal_length}mm")
        print("✅ Preferences loaded successfully")
    
    def save_preferences(self):
        """Save current preferences to file."""
        try:
            # Get current preferences
            preferences = {
                'ui': {
                    'dark_mode': self.dark_mode,
                    'show_depth_map': self.showing_depth_map
                },
                'processing': {
                    'cpu_mode': self.processing_thread.force_cpu_mode if self.processing_thread else False,
                    'detail_level': self.processing_thread.depth_estimator.detail_level if self.processing_thread else 'high'
                },
                'depth': {
                    'hole_filling_enabled': self.processing_thread.depth_estimator.hole_filling_enabled if self.processing_thread else True,
                    'background_correction_enabled': self.processing_thread.depth_estimator.background_correction_enabled if self.processing_thread else True,
                    'smoothing_strength': self.processing_thread.depth_estimator.smoothing_strength if self.processing_thread else 'medium'
                },
                'blur': {
                    'blur_strength': self.blur_slider.value(),
                    'focal_length': self.focal_slider.value()
                }
            }
            
            # Save to file
            success = self.preferences.save_preferences(preferences)
            if success:
                print("💾 Preferences saved successfully")
            else:
                print("❌ Failed to save preferences")
                
        except Exception as e:
            print(f"❌ Error saving preferences: {e}")
    
    def open_image(self):
        """Open an image file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "", "JPEG Files (*.jpg *.jpeg)"
        )
        
        if file_path:
            self.load_image(file_path)
    
    def load_image(self, file_path: str):
        """Load an image from file."""
        self.status_bar.showMessage("Loading image...")
        
        image_array, error = load_jpeg_image(file_path)
        if error:
            QMessageBox.critical(self, "Error", f"Failed to load image: {error}")
            return
        
        self.current_image = image_array
        self.depth_map = None
        self.blurred_image = None
        
        # Display preview image
        preview_image = resize_for_preview(image_array)
        self.preview_display.set_image(preview_image)
        
        # Set original image size for coordinate mapping
        orig_height, orig_width = image_array.shape[:2]
        self.preview_display.set_original_image_size(orig_width, orig_height)
        
        # Estimate processing time
        estimated_time = 5.0  # Simple estimate
        self.status_bar.showMessage(f"Estimating depth map... (estimated {estimated_time:.1f}s)")
        
        # Start depth estimation
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate progress
        self.processing_thread.estimate_depth(image_array)
    
    def save_image(self):
        """Save the processed image."""
        if self.blurred_image is None:
            QMessageBox.warning(self, "Warning", "No processed image to save")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Image", "", "JPEG Files (*.jpg *.jpeg)"
        )
        
        if file_path:
            error = save_jpeg_image(self.blurred_image, file_path)
            if error:
                QMessageBox.critical(self, "Error", f"Failed to save image: {error}")
            else:
                self.status_bar.showMessage(f"Image saved to {file_path}")
    
    def on_focal_point_clicked(self, x: int, y: int):
        """Handle focal point click."""
        if self.depth_map is None:
            self.status_bar.showMessage("Depth map not ready yet")
            return
            
        self.status_bar.showMessage(f"Focal point set at ({x}, {y}) - applying blur effect...")
        self._apply_blur_debounced()
    
    def on_blur_strength_value_changed(self, value: int):
        """Handle blur strength slider value change (for display only)."""
        self.blur_value_label.setText(f"{value}.0")
    
    def on_blur_strength_released(self):
        """Handle blur strength slider release (trigger calculation)."""
        blur_value = self.blur_slider.value()
        print(f"🎚️ Blur strength slider released at value: {blur_value}")
        
        # Save preference
        self.preferences.set_preference('blur', 'blur_strength', blur_value)
        self.save_preferences()
        
        self.blur_timer.start(100)  # Short delay to ensure UI is updated
    
    def on_focal_length_value_changed(self, value: int):
        """Handle focal length slider value change (for display only)."""
        self.focal_length = float(value)
        self.focal_value_label.setText(f"{value}mm")
    
    def on_focal_length_released(self):
        """Handle focal length slider release (trigger calculation)."""
        focal_value = self.focal_slider.value()
        print(f"🎚️ Focal length slider released at value: {focal_value}")
        
        # Save preference
        self.preferences.set_preference('blur', 'focal_length', focal_value)
        self.save_preferences()
        
        self.blur_timer.start(100)  # Short delay to ensure UI is updated
    
    def _apply_blur_debounced(self):
        """Apply blur effect with debouncing."""
        if self.depth_map is None or self.current_image is None:
            return
        
        focal_point = self.preview_display.focal_point
        if focal_point is None:
            return
        
        # Check image size - if too large, skip processing
        height, width = self.current_image.shape[:2]
        total_pixels = height * width
        
        if total_pixels > 15_000_000:  # 15 megapixels
            self.status_bar.showMessage("Image too large for blur processing")
            return
        
        blur_strength = self.blur_slider.value()
        focal_length = self.focal_length
        self.status_bar.showMessage(f"Applying blur effect... (strength: {blur_strength}, focal: {focal_length}mm)")
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate progress
        self.processing_thread.apply_blur(focal_point, blur_strength, focal_length)
    
    def on_depth_completed(self, depth_map: np.ndarray):
        """Handle depth estimation completion."""
        self.depth_map = depth_map
        self.status_bar.showMessage("Depth map generated. Click on the Preview panel to set focal point.")
        self.progress_bar.setVisible(False)
    
    def on_blur_completed(self, blurred_image: np.ndarray):
        """Handle blur application completion."""
        try:
            print(f"🔍 DEBUG: on_blur_completed called with image shape: {blurred_image.shape}")
            print(f"🔍 DEBUG: blurred_image mean: {blurred_image.mean():.3f}")
            
            self.blurred_image = blurred_image
            
            # Only update preview if we're not showing depth map
            if not self.showing_depth_map:
                print(f"🔍 DEBUG: Updating preview display...")
                # Display preview
                preview_image = resize_for_preview(blurred_image)
                print(f"🔍 DEBUG: Preview image shape: {preview_image.shape}")
                self.preview_display.set_image(preview_image)
                print(f"🔍 DEBUG: Preview display updated")
            else:
                print(f"🔍 DEBUG: Skipping preview update - showing depth map")
            
            self.status_bar.showMessage("Blur effect applied successfully")
            self.progress_bar.setVisible(False)
            
        except Exception as e:
            print(f"Error in on_blur_completed: {e}")
            self.status_bar.showMessage(f"Error updating preview: {str(e)}")
            self.progress_bar.setVisible(False)
    
    def toggle_dark_mode(self):
        """Toggle between light and dark mode."""
        self.dark_mode = not self.dark_mode
        self.dark_mode_action.setChecked(self.dark_mode)
        
        # Update menu text
        if self.dark_mode:
            self.dark_mode_action.setText('Light Mode')
        else:
            self.dark_mode_action.setText('Dark Mode')
        
        self.apply_theme()
        
        # Save preference
        self.preferences.set_preference('ui', 'dark_mode', self.dark_mode)
        self.save_preferences()
    
    def toggle_depth_map(self):
        """Toggle between image preview and depth map display."""
        if self.depth_map is None:
            QMessageBox.information(self, "Information", "No depth map available. Please load an image first.")
            self.depth_map_action.setChecked(False)
            return
        
        self.showing_depth_map = not self.showing_depth_map
        self.depth_map_action.setChecked(self.showing_depth_map)
        
        if self.showing_depth_map:
            self.preview_label.setText("Depth Map (Click to set focal point)")
            self.preview_label.setStyleSheet("color: #ff6600; padding: 4px;")  # Orange for depth map
            self.show_depth_map()
        else:
            self.preview_label.setText("Preview (Click to set focal point)")
            self.preview_label.setStyleSheet("color: #0066cc; padding: 4px;")  # Blue for preview
            self.show_image_preview()
    
    def set_detail_level(self, level: str):
        """Set the depth detail level and regenerate depth map if needed."""
        print(f"🔄 Detail level changed to: {level}")
        
        # Update checkboxes
        self.detail_high_action.setChecked(level == "high")
        self.detail_medium_action.setChecked(level == "medium")
        self.detail_low_action.setChecked(level == "low")
        
        # Update depth estimator
        self.processing_thread.depth_estimator.detail_level = level
        print(f"✅ Updated depth estimator detail_level to: {level}")
        
        # Save preference
        self.preferences.set_preference('processing', 'detail_level', level)
        self.save_preferences()
        
        # If we have a current image, regenerate the depth map
        if self.current_image is not None:
            print(f"🔄 Regenerating depth map with {level} detail...")
            self.status_bar.showMessage(f"Regenerating depth map with {level} detail...")
            self.depth_map = None  # Clear cached depth map
            self.blurred_image = None  # Clear blurred result
            self.progress_bar.setVisible(True)
            self.progress_bar.setRange(0, 0)  # Indeterminate progress
            self.processing_thread.estimate_depth(self.current_image)
        else:
            print(f"ℹ️ No current image loaded, detail level set to {level} for next image")
    
    def toggle_cpu_mode(self):
        """Toggle CPU mode and regenerate processing if needed."""
        cpu_mode = self.cpu_mode_action.isChecked()
        
        if cpu_mode:
            self.status_bar.showMessage("Switching to CPU mode...")
            print("🖥️ CPU Mode enabled - forcing CPU processing")
            self.mode_label.setText("CPU")
            self.mode_label.setStyleSheet("color: #ff6600; font-weight: bold; padding: 2px 8px;")
        else:
            self.status_bar.showMessage("Switching to GPU mode...")
            print("🚀 GPU Mode enabled - using GPU acceleration")
            self.mode_label.setText("GPU")
            self.mode_label.setStyleSheet("color: #0078d4; font-weight: bold; padding: 2px 8px;")
        
        # Update processing thread to use CPU mode
        self.processing_thread.force_cpu_mode = cpu_mode
        self.processing_thread.depth_estimator.set_cpu_mode(cpu_mode)
        self.processing_thread.blur_processor.set_cpu_mode(cpu_mode)
        
        # Save preference
        self.preferences.set_preference('processing', 'cpu_mode', cpu_mode)
        self.save_preferences()
        
        # If we have a current image, regenerate everything
        if self.current_image is not None:
            self.status_bar.showMessage(f"Regenerating with {'CPU' if cpu_mode else 'GPU'} processing...")
            self.depth_map = None  # Clear cached depth map
            self.blurred_image = None  # Clear blurred result
            self.progress_bar.setVisible(True)
            self.progress_bar.setRange(0, 0)  # Indeterminate progress
            self.processing_thread.estimate_depth(self.current_image)
    
    def toggle_hole_filling(self):
        """Toggle hole filling and regenerate depth map if needed."""
        hole_filling_enabled = self.hole_filling_action.isChecked()
        
        print(f"🔧 Hole filling: {'enabled' if hole_filling_enabled else 'disabled'}")
        
        # Update depth estimator setting
        self.update_depth_settings(hole_filling_enabled=hole_filling_enabled)
        
        # Save preference
        self.preferences.set_preference('depth', 'hole_filling_enabled', hole_filling_enabled)
        self.save_preferences()
    
    def toggle_background_correction(self):
        """Toggle background correction and regenerate depth map if needed."""
        background_correction_enabled = self.background_correction_action.isChecked()
        
        print(f"🔧 Background correction: {'enabled' if background_correction_enabled else 'disabled'}")
        
        # Update depth estimator setting
        self.update_depth_settings(background_correction_enabled=background_correction_enabled)
        
        # Save preference
        self.preferences.set_preference('depth', 'background_correction_enabled', background_correction_enabled)
        self.save_preferences()
    
    def set_smoothing_strength(self, strength: str):
        """Set smoothing strength and regenerate depth map if needed."""
        print(f"🔧 Smoothing strength changed to: {strength}")
        
        # Update checkboxes
        self.smoothing_low_action.setChecked(strength == "low")
        self.smoothing_medium_action.setChecked(strength == "medium")
        self.smoothing_high_action.setChecked(strength == "high")
        
        # Update depth estimator setting
        self.update_depth_settings(smoothing_strength=strength)
        
        # Save preference
        self.preferences.set_preference('depth', 'smoothing_strength', strength)
        self.save_preferences()
    
    def show_depth_map(self):
        """Display the depth map visualization."""
        if self.depth_map is None:
            return
        
        # Create depth map visualization
        depth_vis = self.create_depth_visualization(self.depth_map)
        
        # Resize for preview
        preview_depth = resize_for_preview(depth_vis)
        
        # Display the depth map
        self.preview_display.set_image(preview_depth)
    
    def show_image_preview(self):
        """Display the image preview (original or blurred)."""
        if self.blurred_image is not None:
            # Show blurred image
            preview_image = resize_for_preview(self.blurred_image)
        elif self.current_image is not None:
            # Show original image
            preview_image = resize_for_preview(self.current_image)
        else:
            return
        
        self.preview_display.set_image(preview_image)
    
    def create_depth_visualization(self, depth_map: np.ndarray) -> np.ndarray:
        """Create a colorized visualization of the depth map."""
        # Normalize to 0-255
        depth_vis = (depth_map * 255).astype(np.uint8)
        
        # Apply colormap (jet colormap for depth visualization)
        depth_colored = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
        
        return depth_colored
    
    def apply_theme(self):
        """Apply the current theme (light or dark)."""
        if self.dark_mode:
            # Dark theme
            self.setStyleSheet("""
                QMainWindow {
                    background-color: #2b2b2b;
                    color: #ffffff;
                }
                QMenuBar {
                    background-color: #3c3c3c;
                    color: #ffffff;
                    border-bottom: 1px solid #555555;
                }
                QMenuBar::item {
                    background-color: transparent;
                    padding: 4px 8px;
                }
                QMenuBar::item:selected {
                    background-color: #555555;
                }
                QMenu {
                    background-color: #3c3c3c;
                    color: #ffffff;
                    border: 1px solid #555555;
                }
                QMenu::item:selected {
                    background-color: #555555;
                }
                QLabel {
                    color: #ffffff;
                }
                QSlider::groove:horizontal {
                    border: 1px solid #555555;
                    height: 8px;
                    background: #3c3c3c;
                    border-radius: 4px;
                }
                QSlider::handle:horizontal {
                    background: #0078d4;
                    border: 1px solid #555555;
                    width: 18px;
                    margin: -2px 0;
                    border-radius: 9px;
                }
                QSlider::handle:horizontal:hover {
                    background: #106ebe;
                }
                QProgressBar {
                    border: 1px solid #555555;
                    border-radius: 4px;
                    text-align: center;
                    background-color: #3c3c3c;
                    color: #ffffff;
                }
                QProgressBar::chunk {
                    background-color: #0078d4;
                    border-radius: 3px;
                }
                QStatusBar {
                    background-color: #3c3c3c;
                    color: #ffffff;
                    border-top: 1px solid #555555;
                }
                QFrame {
                    background-color: #2b2b2b;
                    border: 1px solid #555555;
                }
                QSplitter::handle {
                    background-color: #555555;
                }
            """)
            
            # Update image display widget
            self.preview_display.setStyleSheet("border: 2px solid #0078d4; background-color: #2b2b2b;")
            
            # Update label colors for dark mode
            if self.showing_depth_map:
                self.preview_label.setStyleSheet("color: #ff8800; padding: 4px;")  # Orange for depth map in dark mode
            else:
                self.preview_label.setStyleSheet("color: #0078d4; padding: 4px;")  # Blue for preview in dark mode
            
            # Update mode label for dark mode
            if hasattr(self, 'mode_label'):
                if self.mode_label.text() == "CPU":
                    self.mode_label.setStyleSheet("color: #ff8800; font-weight: bold; padding: 2px 8px;")
                else:
                    self.mode_label.setStyleSheet("color: #0078d4; font-weight: bold; padding: 2px 8px;")
        else:
            # Light theme (default)
            self.setStyleSheet("")
            
            # Reset image display widget to light theme
            self.preview_display.setStyleSheet("border: 2px solid #0066cc; background-color: #f0f0f0;")
            
            # Update label colors for light mode
            if self.showing_depth_map:
                self.preview_label.setStyleSheet("color: #ff6600; padding: 4px;")  # Orange for depth map in light mode
            else:
                self.preview_label.setStyleSheet("color: #0066cc; padding: 4px;")  # Blue for preview in light mode
            
            # Update mode label for light mode
            if hasattr(self, 'mode_label'):
                if self.mode_label.text() == "CPU":
                    self.mode_label.setStyleSheet("color: #ff6600; font-weight: bold; padding: 2px 8px;")
                else:
                    self.mode_label.setStyleSheet("color: #0066cc; font-weight: bold; padding: 2px 8px;")
    
    def closeEvent(self, event):
        """Handle application close."""
        if self.processing_thread is not None:
            self.processing_thread.terminate()
            self.processing_thread.wait()
            
            # Clean up blur processor
            if hasattr(self.processing_thread, 'blur_processor'):
                self.processing_thread.blur_processor.cleanup()
        
        event.accept()


def exception_handler(exc_type, exc_value, exc_traceback):
    """Global exception handler for debugging."""
    print(f"Uncaught exception: {exc_type.__name__}: {exc_value}")
    import traceback
    traceback.print_exc()


def main():
    """Main application entry point."""
    try:
        # Set up global exception handler
        sys.excepthook = exception_handler
        
        print("Starting application...")
        app = QApplication(sys.argv)
        app.setApplicationName("Depth Blur Filter")
        app.setApplicationVersion("1.0")
        
        print("Creating main window...")
        window = DepthBlurApp()
        print("Showing window...")
        window.show()
        
        print("Starting event loop...")
        sys.exit(app.exec_())
        
    except Exception as e:
        print(f"Error in main(): {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
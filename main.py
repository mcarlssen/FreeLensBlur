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
from utils.performance import PerformanceMonitor, optimize_image_for_processing, estimate_processing_time


class ImageDisplayWidget(QLabel):
    """Custom widget for displaying images with click handling."""
    
    clicked = pyqtSignal(int, int)  # x, y coordinates
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(400, 300)
        self.setStyleSheet("border: 1px solid gray; background-color: #f0f0f0;")
        self.setAlignment(Qt.AlignCenter)
        self.setScaledContents(True)
        self.focal_point = None
        self.scale_factor = 1.0
        
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            # Calculate actual image coordinates
            x = int(event.x() / self.scale_factor)
            y = int(event.y() / self.scale_factor)
            self.focal_point = (x, y)
            self.clicked.emit(x, y)
            self.update()
    
    def set_image(self, image_array: np.ndarray):
        """Set the image to display."""
        if image_array is None:
            self.clear()
            return
        
        # Convert numpy array to QImage
        height, width, channel = image_array.shape
        bytes_per_line = 3 * width
        q_image = QImage(image_array.data, width, height, bytes_per_line, QImage.Format_RGB888)
        
        # Convert to QPixmap
        pixmap = QPixmap.fromImage(q_image)
        
        # Calculate scale factor for coordinate mapping
        widget_size = self.size()
        pixmap_size = pixmap.size()
        
        scale_x = pixmap_size.width() / widget_size.width()
        scale_y = pixmap_size.height() / widget_size.height()
        self.scale_factor = max(scale_x, scale_y)
        
        self.setPixmap(pixmap)
    
    def paintEvent(self, event):
        super().paintEvent(event)
        
        # Draw focal point indicator
        if self.focal_point is not None:
            painter = QPainter(self)
            painter.setPen(QPen(QColor(255, 0, 0), 3))
            
            # Calculate display coordinates
            x = int(self.focal_point[0] * self.scale_factor)
            y = int(self.focal_point[1] * self.scale_factor)
            
            # Draw circle
            painter.drawEllipse(x - 10, y - 10, 20, 20)
            painter.drawLine(x - 15, y, x + 15, y)
            painter.drawLine(x, y - 15, x, y + 15)


class ProcessingThread(QThread):
    """Thread for processing depth estimation and blur application."""
    
    progress_updated = pyqtSignal(int, int)  # current, total
    depth_completed = pyqtSignal(np.ndarray)  # depth map
    blur_completed = pyqtSignal(np.ndarray)  # blurred image
    error_occurred = pyqtSignal(str)  # error message
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.depth_estimator = DepthEstimator()
        self.blur_processor = BlurProcessor()
        self.current_image = None
        self.focal_point = None
        self.blur_strength = 5.0
        
    def estimate_depth(self, image_array: np.ndarray):
        """Estimate depth map for the image."""
        self.current_image = image_array
        self.focal_point = None
        self.start()
    
    def apply_blur(self, focal_point: Tuple[int, int], blur_strength: float):
        """Apply blur effect with given parameters."""
        self.focal_point = focal_point
        self.blur_strength = blur_strength
        self.start()
    
    def run(self):
        """Main thread execution."""
        try:
            if self.current_image is None:
                return
            
            # Estimate depth if not already done
            if not hasattr(self, 'depth_map') or self.depth_map is None:
                self.progress_updated.emit(0, 100)
                self.depth_map = self.depth_estimator.estimate_depth(self.current_image)
                self.depth_completed.emit(self.depth_map)
            
            # Apply blur if focal point is set
            if self.focal_point is not None:
                self.progress_updated.emit(50, 100)
                
                # Use simple blur for preview
                blurred = self.blur_processor.apply_focal_blur_simple(
                    self.current_image, self.depth_map, 
                    self.focal_point, self.blur_strength
                )
                
                self.progress_updated.emit(100, 100)
                self.blur_completed.emit(blurred)
                
        except Exception as e:
            self.error_occurred.emit(str(e))


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
        self.performance_monitor = PerformanceMonitor()
        
        self.init_ui()
        self.setup_processing_thread()
    
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
    
    def create_image_display(self, parent_layout):
        """Create the image display area."""
        # Create splitter for side-by-side display
        splitter = QSplitter(Qt.Horizontal)
        parent_layout.addWidget(splitter)
        
        # Original image panel
        original_frame = QFrame()
        original_frame.setFrameStyle(QFrame.StyledPanel)
        original_layout = QVBoxLayout(original_frame)
        
        original_label = QLabel("Original")
        original_label.setAlignment(Qt.AlignCenter)
        original_label.setFont(QFont("Arial", 12, QFont.Bold))
        original_layout.addWidget(original_label)
        
        self.original_display = ImageDisplayWidget()
        original_layout.addWidget(self.original_display)
        
        # Preview image panel
        preview_frame = QFrame()
        preview_frame.setFrameStyle(QFrame.StyledPanel)
        preview_layout = QVBoxLayout(preview_frame)
        
        preview_label = QLabel("Preview")
        preview_label.setAlignment(Qt.AlignCenter)
        preview_label.setFont(QFont("Arial", 12, QFont.Bold))
        preview_layout.addWidget(preview_label)
        
        self.preview_display = ImageDisplayWidget()
        self.preview_display.clicked.connect(self.on_focal_point_clicked)
        preview_layout.addWidget(self.preview_display)
        
        # Add panels to splitter
        splitter.addWidget(original_frame)
        splitter.addWidget(preview_frame)
        splitter.setSizes([600, 600])
    
    def create_controls(self, parent_layout):
        """Create the control panel."""
        controls_layout = QHBoxLayout()
        
        # Blur strength slider
        blur_label = QLabel("Blur Strength:")
        controls_layout.addWidget(blur_label)
        
        self.blur_slider = QSlider(Qt.Horizontal)
        self.blur_slider.setMinimum(0)
        self.blur_slider.setMaximum(10)
        self.blur_slider.setValue(5)
        self.blur_slider.valueChanged.connect(self.on_blur_strength_changed)
        controls_layout.addWidget(self.blur_slider)
        
        self.blur_value_label = QLabel("5.0")
        self.blur_value_label.setMinimumWidth(30)
        controls_layout.addWidget(self.blur_value_label)
        
        controls_layout.addStretch()
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        controls_layout.addWidget(self.progress_bar)
        
        parent_layout.addLayout(controls_layout)
    
    def create_status_bar(self):
        """Create the status bar."""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")
    
    def setup_processing_thread(self):
        """Setup the processing thread."""
        self.processing_thread = ProcessingThread()
        self.processing_thread.progress_updated.connect(self.update_progress)
        self.processing_thread.depth_completed.connect(self.on_depth_completed)
        self.processing_thread.blur_completed.connect(self.on_blur_completed)
        self.processing_thread.error_occurred.connect(self.on_error)
    
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
        
        # Display original image
        preview_image = resize_for_preview(image_array)
        self.original_display.set_image(preview_image)
        self.preview_display.set_image(preview_image)
        
        # Estimate processing time
        estimated_time = estimate_processing_time(image_array.shape[:2], 5.0)
        self.status_bar.showMessage(f"Estimating depth map... (estimated {estimated_time:.1f}s)")
        
        # Start depth estimation
        self.progress_bar.setVisible(True)
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
            return
        
        self.status_bar.showMessage(f"Focal point set at ({x}, {y})")
        self.blur_timer.start(300)  # Debounce blur application
    
    def on_blur_strength_changed(self, value: int):
        """Handle blur strength slider change."""
        self.blur_value_label.setText(f"{value}.0")
        self.blur_timer.start(300)  # Debounce blur application
    
    def _apply_blur_debounced(self):
        """Apply blur effect with debouncing."""
        if self.depth_map is None or self.current_image is None:
            return
        
        focal_point = self.preview_display.focal_point
        if focal_point is None:
            return
        
        blur_strength = self.blur_slider.value()
        
        self.status_bar.showMessage("Applying blur effect...")
        self.progress_bar.setVisible(True)
        self.processing_thread.apply_blur(focal_point, blur_strength)
    
    def update_progress(self, current: int, total: int):
        """Update progress bar."""
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(current)
    
    def on_depth_completed(self, depth_map: np.ndarray):
        """Handle depth estimation completion."""
        self.depth_map = depth_map
        self.status_bar.showMessage("Depth map generated. Click to set focal point.")
        self.progress_bar.setVisible(False)
    
    def on_blur_completed(self, blurred_image: np.ndarray):
        """Handle blur application completion."""
        self.blurred_image = blurred_image
        
        # Display preview
        preview_image = resize_for_preview(blurred_image)
        self.preview_display.set_image(preview_image)
        
        self.status_bar.showMessage("Blur effect applied")
        self.progress_bar.setVisible(False)
    
    def on_error(self, error_message: str):
        """Handle processing errors."""
        QMessageBox.critical(self, "Error", f"Processing error: {error_message}")
        self.status_bar.showMessage("Error occurred")
        self.progress_bar.setVisible(False)
    
    def closeEvent(self, event):
        """Handle application close."""
        if self.processing_thread and self.processing_thread.isRunning():
            self.processing_thread.terminate()
            self.processing_thread.wait()
        
        if hasattr(self.processing_thread, 'blur_processor'):
            self.processing_thread.blur_processor.cleanup()
        
        event.accept()


def main():
    """Main application entry point."""
    app = QApplication(sys.argv)
    app.setApplicationName("Depth Blur Filter")
    app.setApplicationVersion("1.0")
    
    window = DepthBlurApp()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

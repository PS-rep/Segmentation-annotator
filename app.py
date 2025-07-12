import sys
import os
import cv2
import numpy as np
from pathlib import Path
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QPushButton, QComboBox, 
                             QFileDialog, QMessageBox, QSpinBox, QCheckBox,
                             QGroupBox, QGridLayout, QTextEdit, QSplitter,
                             QFrame, QInputDialog, QProgressBar, QStatusBar,
                             QScrollArea)
from PyQt5.QtCore import Qt, QPoint, QTimer, pyqtSignal, QObject
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen, QColor, QFont

class ImageCanvas(QLabel):
    """Custom QLabel for displaying and annotating images"""
    
    annotation_changed = pyqtSignal()  # Signal emitted when annotations change
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(800, 600)
        self.setStyleSheet("border: 2px solid gray; background-color: white;")
        self.setAlignment(Qt.AlignCenter)
        self.setScaledContents(False)
        
        # Image data
        self.original_image = None  # Original loaded image
        self.display_image = None   # Image displayed on the canvas
        self.scale_factor = 1.0     # Scale factor for resizing image
        self.offset_x = 0           # X offset for centering image
        self.offset_y = 0           # Y offset for centering image
        
        # Annotation data
        self.current_points = []    # Points for the current polygon being drawn
        self.all_annotations = []   # List of all completed annotations
        self.current_class_id = 0   # Current class selected for annotation
        self.class_colors = []      # Colors for each class
        self.class_names = []       # Names of annotation classes
        
        # Drawing state
        self.drawing_enabled = False  # Whether drawing mode is enabled
        self.drawing_active = False   # Whether a polygon is being drawn
        
        # Mouse tracking
        self.setMouseTracking(True)
        self.preview_point = None     # Preview point for drawing line to mouse

    def set_image(self, image_path):
        """Load and display an image"""
        if not os.path.exists(image_path):
            return False
            
        self.original_image = cv2.imread(image_path)
        if self.original_image is None:
            return False
            
        self.all_annotations = []
        self.current_points = []
        self.drawing_active = False
        
        self._update_display()
        return True

    def set_classes(self, class_names):
        """Set class names and generate colors"""
        self.class_names = class_names
        self.class_colors = self._generate_colors(len(class_names))
        self.current_class_id = 0

    def _generate_colors(self, num_classes):
        """Generate distinct colors for each class"""
        colors = []
        for i in range(num_classes):
            hue = int(180 * i / num_classes)
            hsv_color = np.array([[[hue, 255, 255]]], dtype=np.uint8)
            bgr_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0][0]
            colors.append(tuple(map(int, bgr_color)))
        return colors

    def _update_display(self):
        """Update the display image"""
        if self.original_image is None:
            return
            
        # Calculate scale and offset for centering
        h, w = self.original_image.shape[:2]
        widget_w, widget_h = self.width(), self.height()
        
        scale_w = widget_w / w
        scale_h = widget_h / h
        self.scale_factor = min(scale_w, scale_h, 1.0)  # Don't scale up
        
        display_w = int(w * self.scale_factor)
        display_h = int(h * self.scale_factor)
        
        self.offset_x = (widget_w - display_w) // 2
        self.offset_y = (widget_h - display_h) // 2
        
        # Resize image
        self.display_image = cv2.resize(self.original_image, (display_w, display_h))
        
        # Draw annotations
        self._draw_annotations()
        
        # Convert to QPixmap and display
        self._show_image()

    def _draw_annotations(self):
        """Draw all annotations on the display image"""
        if self.display_image is None:
            return
            
        # Create overlay for semi-transparent polygons
        overlay = self.display_image.copy()
        
        # Draw completed annotations
        for ann in self.all_annotations:
            if not ann['points']:
                continue
                
            # Scale points to display coordinates
            scaled_points = []
            for px, py in ann['points']:
                scaled_x = int(px * self.scale_factor)
                scaled_y = int(py * self.scale_factor)
                scaled_points.append((scaled_x, scaled_y))
            
            if len(scaled_points) >= 3:
                points_array = np.array(scaled_points, dtype=np.int32)
                color = self.class_colors[ann['class_id']] if ann['class_id'] < len(self.class_colors) else (0, 255, 0)
                
                # Fill polygon
                cv2.fillPoly(overlay, [points_array], color)
                
                # Draw outline
                cv2.polylines(self.display_image, [points_array], True, color, 2)
        
        # Blend overlay with original
        alpha = 0.3
        self.display_image = cv2.addWeighted(self.display_image, 1-alpha, overlay, alpha, 0)
        
        # Draw current polygon being created
        if len(self.current_points) > 0:
            color = self.class_colors[self.current_class_id] if self.current_class_id < len(self.class_colors) else (0, 255, 0)
            
            # Draw points
            for px, py in self.current_points:
                scaled_x = int(px * self.scale_factor)
                scaled_y = int(py * self.scale_factor)
                cv2.circle(self.display_image, (scaled_x, scaled_y), 4, color, -1)
            
            # Draw lines between points
            if len(self.current_points) > 1:
                scaled_points = []
                for px, py in self.current_points:
                    scaled_x = int(px * self.scale_factor)
                    scaled_y = int(py * self.scale_factor)
                    scaled_points.append((scaled_x, scaled_y))
                
                points_array = np.array(scaled_points, dtype=np.int32)
                cv2.polylines(self.display_image, [points_array], False, color, 2)
            
            # Draw preview line to mouse position
            if self.preview_point and self.drawing_active:
                last_point = self.current_points[-1]
                scaled_last_x = int(last_point[0] * self.scale_factor)
                scaled_last_y = int(last_point[1] * self.scale_factor)
                cv2.line(self.display_image, (scaled_last_x, scaled_last_y), 
                        self.preview_point, color, 2)

    def _show_image(self):
        """Convert OpenCV image to QPixmap and display"""
        if self.display_image is None:
            return
            
        h, w, ch = self.display_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(self.display_image.data, w, h, bytes_per_line, QImage.Format_BGR888)
        
        pixmap = QPixmap.fromImage(qt_image)
        self.setPixmap(pixmap)

    def _image_to_original_coords(self, x, y):
        """Convert widget coordinates to original image coordinates"""
        if self.scale_factor == 0:
            return None, None
            
        # Adjust for offset and scale
        img_x = (x - self.offset_x) / self.scale_factor
        img_y = (y - self.offset_y) / self.scale_factor
        
        # Check bounds
        if (img_x < 0 or img_y < 0 or 
            img_x >= self.original_image.shape[1] or 
            img_y >= self.original_image.shape[0]):
            return None, None
            
        return int(img_x), int(img_y)

    def mousePressEvent(self, event):
        """Handle mouse press events"""
        if event.button() == Qt.LeftButton and self.drawing_enabled:
            img_x, img_y = self._image_to_original_coords(event.x(), event.y())
            if img_x is not None and img_y is not None:
                if not self.drawing_active:
                    # Start new polygon
                    self.drawing_active = True
                    self.current_points = []
                
                # Add point
                self.current_points.append((img_x, img_y))
                self._update_display()
                
        elif event.button() == Qt.RightButton and self.drawing_active:
            # Finish current polygon
            self._finish_polygon()

    def mouseMoveEvent(self, event):
        """Handle mouse move events"""
        if self.drawing_active and len(self.current_points) > 0:
            # Update preview point
            display_x = event.x() - self.offset_x
            display_y = event.y() - self.offset_y
            
            if (0 <= display_x <= self.display_image.shape[1] and 
                0 <= display_y <= self.display_image.shape[0]):
                self.preview_point = (display_x, display_y)
                self._update_display()

    def _finish_polygon(self):
        """Finish the current polygon"""
        if len(self.current_points) >= 3:
            annotation = {
                'class_id': self.current_class_id,
                'points': self.current_points.copy()
            }
            self.all_annotations.append(annotation)
            self.annotation_changed.emit()
        
        self.current_points = []
        self.drawing_active = False
        self.preview_point = None
        self._update_display()

    def start_drawing(self):
        """Enable drawing mode"""
        self.drawing_enabled = True
        self.setCursor(Qt.CrossCursor)

    def stop_drawing(self):
        """Disable drawing mode"""
        self.drawing_enabled = False
        self.drawing_active = False
        self.current_points = []
        self.preview_point = None
        self.setCursor(Qt.ArrowCursor)
        self._update_display()

    def set_current_class(self, class_id):
        """Set the current class for annotation"""
        self.current_class_id = class_id
        self._update_display()

    def undo_last_annotation(self):
        """Remove the last annotation"""
        if self.all_annotations:
            self.all_annotations.pop()
            self.annotation_changed.emit()
            self._update_display()

    def clear_all_annotations(self):
        """Clear all annotations"""
        self.all_annotations = []
        self.current_points = []
        self.drawing_active = False
        self.preview_point = None
        self.annotation_changed.emit()
        self._update_display()

    def get_annotations(self):
        """Get all annotations"""
        return self.all_annotations

    def set_annotations(self, annotations):
        """Set annotations"""
        self.all_annotations = annotations
        self._update_display()

    def resizeEvent(self, event):
        """Handle resize events"""
        super().resizeEvent(event)
        if self.original_image is not None:
            self._update_display()


class YOLOSegmentationGUI(QMainWindow):
    """Main GUI window for YOLO segmentation annotation"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLO Segmentation Annotator")
        self.setGeometry(100, 100, 1400, 900)
        
        # Data
        self.input_dir = ""
        self.output_dir = ""
        self.image_files = []
        self.current_image_index = 0
        self.class_names = ["object"]
        
        # UI Setup
        self.setup_ui()
        self.setup_connections()
        
        # Status
        self.update_status("Ready")
    
    def setup_ui(self):
        """Setup the user interface"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Create main layout
        main_layout = QHBoxLayout(central_widget)
        
        # Create splitter
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)
        
        # Left panel - Controls
        left_panel = self.create_left_panel()
        splitter.addWidget(left_panel)
        
        # Right panel - Image canvas
        self.canvas = ImageCanvas()
        splitter.addWidget(self.canvas)
        
        # Set splitter proportions
        splitter.setSizes([300, 1100])
        
        # Status bar
        self.statusBar().showMessage("Ready")
    
    def create_left_panel(self):
        """Create the left control panel"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Directory selection
        dir_group = QGroupBox("Directories")
        dir_layout = QVBoxLayout(dir_group)
        
        self.input_dir_label = QLabel("Input: Not selected")
        self.input_dir_label.setWordWrap(True)
        self.input_dir_btn = QPushButton("Select Input Directory")
        
        self.output_dir_label = QLabel("Output: Not selected")
        self.output_dir_label.setWordWrap(True)
        self.output_dir_btn = QPushButton("Select Output Directory")
        
        dir_layout.addWidget(self.input_dir_label)
        dir_layout.addWidget(self.input_dir_btn)
        dir_layout.addWidget(self.output_dir_label)
        dir_layout.addWidget(self.output_dir_btn)
        
        layout.addWidget(dir_group)
        
        # Class management
        class_group = QGroupBox("Classes")
        class_layout = QVBoxLayout(class_group)
        
        class_btn_layout = QHBoxLayout()
        self.add_class_btn = QPushButton("Add Class")
        self.edit_classes_btn = QPushButton("Edit Classes")
        class_btn_layout.addWidget(self.add_class_btn)
        class_btn_layout.addWidget(self.edit_classes_btn)
        
        self.class_combo = QComboBox()
        self.class_combo.addItem("0: object")
        
        class_layout.addLayout(class_btn_layout)
        class_layout.addWidget(QLabel("Current Class:"))
        class_layout.addWidget(self.class_combo)
        
        layout.addWidget(class_group)
        
        # Image navigation
        nav_group = QGroupBox("Image Navigation")
        nav_layout = QVBoxLayout(nav_group)
        
        self.image_info_label = QLabel("No images loaded")
        nav_layout.addWidget(self.image_info_label)
        
        nav_btn_layout = QHBoxLayout()
        self.prev_btn = QPushButton("Previous")
        self.next_btn = QPushButton("Next")
        nav_btn_layout.addWidget(self.prev_btn)
        nav_btn_layout.addWidget(self.next_btn)
        
        nav_layout.addLayout(nav_btn_layout)
        
        layout.addWidget(nav_group)
        
        # Annotation controls
        ann_group = QGroupBox("Annotation Controls")
        ann_layout = QVBoxLayout(ann_group)
        
        self.drawing_btn = QPushButton("Start Drawing")
        self.drawing_btn.setCheckable(True)
        self.drawing_btn.setStyleSheet("""
            QPushButton:checked {
                background-color: #4CAF50;
                color: white;
            }
        """)
        
        ann_btn_layout = QHBoxLayout()
        self.undo_btn = QPushButton("Undo")
        self.clear_btn = QPushButton("Clear All")
        ann_btn_layout.addWidget(self.undo_btn)
        ann_btn_layout.addWidget(self.clear_btn)
        
        self.ann_count_label = QLabel("Annotations: 0")
        
        ann_layout.addWidget(self.drawing_btn)
        ann_layout.addLayout(ann_btn_layout)
        ann_layout.addWidget(self.ann_count_label)
        
        layout.addWidget(ann_group)
        
        # Save controls
        save_group = QGroupBox("Save")
        save_layout = QVBoxLayout(save_group)
        
        self.save_current_btn = QPushButton("Save Current")
        self.save_all_btn = QPushButton("Save All")
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        
        save_layout.addWidget(self.save_current_btn)
        save_layout.addWidget(self.save_all_btn)
        save_layout.addWidget(self.progress_bar)
        
        layout.addWidget(save_group)
        
        # Instructions
        instructions = QTextEdit()
        instructions.setMaximumHeight(200)
        instructions.setReadOnly(True)
        instructions.setHtml("""
        <h4>Instructions:</h4>
        <ul>
        <li><b>Select directories:</b> Choose input (images) and output folders</li>
        <li><b>Start Drawing:</b> Click button to enable annotation mode</li>
        <li><b>Left Click:</b> Add points to polygon</li>
        <li><b>Right Click:</b> Finish current polygon</li>
        <li><b>Navigation:</b> Use Previous/Next buttons</li>
        <li><b>Classes:</b> Select class before drawing</li>
        <li><b>Save:</b> Save current image or all annotations</li>
        </ul>
        """)
        
        layout.addWidget(instructions)
        
        # Add stretch
        layout.addStretch()
        
        return panel
    
    def setup_connections(self):
        """Setup signal connections"""
        # Directory selection
        self.input_dir_btn.clicked.connect(self.select_input_directory)
        self.output_dir_btn.clicked.connect(self.select_output_directory)
        
        # Class management
        self.add_class_btn.clicked.connect(self.add_class)
        self.edit_classes_btn.clicked.connect(self.edit_classes)
        self.class_combo.currentIndexChanged.connect(self.on_class_changed)
        
        # Navigation
        self.prev_btn.clicked.connect(self.previous_image)
        self.next_btn.clicked.connect(self.next_image)
        
        # Annotation controls
        self.drawing_btn.clicked.connect(self.toggle_drawing)
        self.undo_btn.clicked.connect(self.undo_annotation)
        self.clear_btn.clicked.connect(self.clear_annotations)
        
        # Save controls
        self.save_current_btn.clicked.connect(self.save_current_annotations)
        self.save_all_btn.clicked.connect(self.save_all_annotations)
        
        # Canvas signals
        self.canvas.annotation_changed.connect(self.on_annotation_changed)
    
    def select_input_directory(self):
        """Select input directory"""
        directory = QFileDialog.getExistingDirectory(self, "Select Input Directory")
        if directory:
            self.input_dir = directory
            self.input_dir_label.setText(f"Input: {directory}")
            self.load_images()
    
    def select_output_directory(self):
        """Select output directory"""
        directory = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if directory:
            self.output_dir = directory
            self.output_dir_label.setText(f"Output: {directory}")
            self.create_output_directories()
    
    def load_images(self):
        """Load image files from input directory"""
        if not self.input_dir:
            return
        
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
        self.image_files = []
        
        for ext in extensions:
            self.image_files.extend(Path(self.input_dir).glob(f'*{ext}'))
            self.image_files.extend(Path(self.input_dir).glob(f'*{ext.upper()}'))
        
        self.image_files = sorted(self.image_files)
        self.current_image_index = 0
        
        if self.image_files:
            self.load_current_image()
            self.update_image_info()
            self.update_status(f"Loaded {len(self.image_files)} images")
        else:
            self.update_status("No images found in directory")
    
    def load_current_image(self):
        """Load the current image"""
        if not self.image_files or self.current_image_index >= len(self.image_files):
            return
        
        image_path = str(self.image_files[self.current_image_index])
        if self.canvas.set_image(image_path):
            self.canvas.set_classes(self.class_names)
            self.load_existing_annotations()
            self.update_image_info()
            self.on_annotation_changed()
    
    def create_output_directories(self):
        """Create output directories"""
        if not self.output_dir:
            return
        
        images_dir = os.path.join(self.output_dir, 'images')
        labels_dir = os.path.join(self.output_dir, 'labels')
        
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)
        
        self.create_dataset_yaml()
    
    def create_dataset_yaml(self):
        """Create dataset.yaml file"""
        if not self.output_dir:
            return
        
        yaml_content = f"""# YOLO Dataset Configuration
# Generated by YOLO Segmentation Annotator

# Dataset paths
path: {self.output_dir}
train: images
val: images

# Classes
nc: {len(self.class_names)}
names: {self.class_names}
"""
        
        yaml_path = os.path.join(self.output_dir, 'dataset.yaml')
        with open(yaml_path, 'w') as f:
            f.write(yaml_content)
    
    def add_class(self):
        """Add a new class"""
        class_name, ok = QInputDialog.getText(self, "Add Class", "Enter class name:")
        if ok and class_name.strip():
            self.class_names.append(class_name.strip())
            self.update_class_combo()
            self.canvas.set_classes(self.class_names)
            self.create_dataset_yaml()
    
    def edit_classes(self):
        """Edit all classes"""
        current_classes = ", ".join(self.class_names)
        classes_str, ok = QInputDialog.getText(
            self, "Edit Classes", 
            "Enter class names (comma-separated):", 
            text=current_classes
        )
        if ok and classes_str.strip():
            new_classes = [name.strip() for name in classes_str.split(',') if name.strip()]
            if new_classes:
                self.class_names = new_classes
                self.update_class_combo()
                self.canvas.set_classes(self.class_names)
                self.create_dataset_yaml()
    
    def update_class_combo(self):
        """Update class combo box"""
        self.class_combo.clear()
        for i, name in enumerate(self.class_names):
            self.class_combo.addItem(f"{i}: {name}")
    
    def on_class_changed(self):
        """Handle class selection change"""
        self.canvas.set_current_class(self.class_combo.currentIndex())
    
    def toggle_drawing(self):
        """Toggle drawing mode"""
        if self.drawing_btn.isChecked():
            self.canvas.start_drawing()
            self.drawing_btn.setText("Stop Drawing")
        else:
            self.canvas.stop_drawing()
            self.drawing_btn.setText("Start Drawing")
    
    def undo_annotation(self):
        """Undo last annotation"""
        self.canvas.undo_last_annotation()
    
    def clear_annotations(self):
        """Clear all annotations"""
        reply = QMessageBox.question(self, "Clear Annotations", 
                                   "Are you sure you want to clear all annotations?",
                                   QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.canvas.clear_all_annotations()
    
    def previous_image(self):
        """Go to previous image"""
        if self.current_image_index > 0:
            self.save_current_annotations()
            self.current_image_index -= 1
            self.load_current_image()
    
    def next_image(self):
        """Go to next image"""
        if self.current_image_index < len(self.image_files) - 1:
            self.save_current_annotations()
            self.current_image_index += 1
            self.load_current_image()
    
    def update_image_info(self):
        """Update image information display"""
        if self.image_files:
            total = len(self.image_files)
            current = self.current_image_index + 1
            filename = self.image_files[self.current_image_index].name
            self.image_info_label.setText(f"Image {current}/{total}\n{filename}")
        else:
            self.image_info_label.setText("No images loaded")
    
    def on_annotation_changed(self):
        """Handle annotation changes"""
        count = len(self.canvas.get_annotations())
        self.ann_count_label.setText(f"Annotations: {count}")
    
    def update_status(self, message):
        """Update status bar"""
        self.statusBar().showMessage(message)
    
    def save_current_annotations(self):
        """Save annotations for current image"""
        if not self.output_dir or not self.image_files:
            return
        
        current_image_path = self.image_files[self.current_image_index]
        image_name = current_image_path.stem
        
        # Copy image to output directory
        images_dir = os.path.join(self.output_dir, 'images')
        output_image_path = os.path.join(images_dir, f"{image_name}.jpg")
        
        try:
            original_image = cv2.imread(str(current_image_path))
            if original_image is not None:
                cv2.imwrite(output_image_path, original_image)
        except Exception as e:
            self.update_status(f"Error saving image: {e}")
            return
        
        # Save annotations
        labels_dir = os.path.join(self.output_dir, 'labels')
        label_path = os.path.join(labels_dir, f"{image_name}.txt")
        
        annotations = self.canvas.get_annotations()
        
        try:
            original_image = cv2.imread(str(current_image_path))
            h, w = original_image.shape[:2]
            
            with open(label_path, 'w') as f:
                for ann in annotations:
                    class_id = ann['class_id']
                    points = ann['points']
                    
                    # Normalize coordinates
                    normalized_points = []
                    for x, y in points:
                        norm_x = x / w
                        norm_y = y / h
                        normalized_points.extend([norm_x, norm_y])
                    
                    # Write in YOLO format
                    line = f"{class_id} " + " ".join(map(str, normalized_points))
                    f.write(line + "\n")
            
            self.update_status(f"Saved annotations for {image_name}")
            
        except Exception as e:
            self.update_status(f"Error saving annotations: {e}")
    
    def save_all_annotations(self):
        """Save annotations for all images"""
        if not self.output_dir or not self.image_files:
            QMessageBox.warning(self, "Warning", "Please select directories and load images first")
            return
        
        reply = QMessageBox.question(self, "Save All", 
                                   "This will save annotations for all images. Continue?",
                                   QMessageBox.Yes | QMessageBox.No)
        if reply != QMessageBox.Yes:
            return
        
        # Show progress bar
        self.progress_bar.setVisible(True)
        self.progress_bar.setMaximum(len(self.image_files))
        
        saved_count = 0
        current_index = self.current_image_index
        
        for i, image_path in enumerate(self.image_files):
            self.current_image_index = i
            self.load_current_image()
            
            if self.canvas.get_annotations():
                self.save_current_annotations()
                saved_count += 1
            
            self.progress_bar.setValue(i + 1)
            QApplication.processEvents()
        
        # Restore current image
        self.current_image_index = current_index
        self.load_current_image()
        
        self.progress_bar.setVisible(False)
        self.update_status(f"Saved annotations for {saved_count} images")
        QMessageBox.information(self, "Save Complete", 
                              f"Saved annotations for {saved_count} images")
    
    def load_existing_annotations(self):
        """Load existing annotations if they exist"""
        if not self.output_dir or not self.image_files:
            return
        
        current_image_path = self.image_files[self.current_image_index]
        image_name = current_image_path.stem
        labels_dir = os.path.join(self.output_dir, 'labels')
        label_path = os.path.join(labels_dir, f"{image_name}.txt")
        
        if not os.path.exists(label_path):
            return
        
        try:
            original_image = cv2.imread(str(current_image_path))
            if original_image is None:
                return
            
            h, w = original_image.shape[:2]
            annotations = []
            
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 7:  # At least class_id + 3 points (6 coordinates)
                        class_id = int(parts[0])
                        coords = list(map(float, parts[1:]))
                        
                        # Convert normalized coordinates back to image coordinates
                        points = []
                        for i in range(0, len(coords), 2):
                            if i + 1 < len(coords):
                                x = int(coords[i] * w)
                                y = int(coords[i+1] * h)
                                points.append((x, y))
                        
                        if len(points) >= 3:  # Valid polygon
                            annotation = {
                                'class_id': class_id,
                                'points': points,
                                'class_name': self.class_names[class_id] if class_id < len(self.class_names) else "unknown"
                            }
                            annotations.append(annotation)
            
            self.canvas.set_annotations(annotations)
            
        except Exception as e:
            print(f"Error loading annotations: {e}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = YOLOSegmentationGUI()
    window.show()
    sys.exit(app.exec_())
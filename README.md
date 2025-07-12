# YOLO Segmentation Annotator

A desktop application for annotating images with polygon masks for YOLO segmentation tasks. Built with PyQt5 and OpenCV.

## Features
- Load images from a directory
- Draw polygon masks for segmentation
- Manage multiple annotation classes
- Save annotations in YOLO segmentation format
- Edit, undo, and clear annotations
- Responsive and user-friendly GUI
- Progress bar for batch saving

## Installation
1. Clone this repository or copy the files to your workspace.
2. Create and activate a Python virtual environment (recommended).
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   Required packages:
   - PyQt5
   - opencv-python
   - numpy

## Usage
1. Run the application:
   ```bash
   python app.py
   ```
2. Select the input directory containing images.
3. Select the output directory for saving annotations.
4. Add or edit annotation classes as needed.
5. Click "Start Drawing" to begin annotating polygons on images.
6. Use left-click to add points, right-click to finish a polygon.
7. Save annotations for the current image or all images.

## Annotation Format
- Annotations are saved in YOLO segmentation format:
  - Each line: `<class_id> <x1> <y1> <x2> <y2> ... <xn> <yn>` (coordinates normalized to [0,1])
- Images and labels are saved in `images/` and `labels/` subfolders of the output directory.
- A `dataset.yaml` file is generated for YOLO training.

## Screenshots
*Add screenshots here if desired*

## License
MIT License

## Author
*Your Name*

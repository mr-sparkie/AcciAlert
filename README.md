# AcciAlert

AcciAlert is a video processing tool designed to detect accidents in real-time using the YOLO (You Only Look Once) object detection algorithm. This project utilizes OpenCV and YOLO to analyze video footage and alert users when an accident is detected.

## Features

- Real-time accident detection in video streams.
- Configurable to recognize different classes of objects.
- Alerts the user when an accident is detected.
- Visualizes detections with bounding boxes and labels.

## Table of Contents

- [Setup](#setup)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
  - [Running the Script](#running-the-script)
  - [Example Output](#example-output)
- [File Structure](#file-structure)
- [Project Structure](#project-structure)
- [License](#license)
- [Acknowledgements](#acknowledgements)
- [Contributing](#contributing)

## Setup

### Prerequisites

- Python 3.x
- OpenCV
- NumPy
- YOLOv3 weights and configuration files

### Installation

1. **Clone the repository:**

   ```sh
   git clone https://github.com/mr-sparkie/AcciAlert.git
   cd AcciAlert
   ```

2. **Install the required Python packages:**

   ```sh
   pip install opencv-python numpy
   ```

3. **Download YOLOv3 weights and configuration files:**

   - [YOLOv3 Weights](https://pjreddie.com/media/files/yolov3.weights)
   - [YOLOv3 Configuration](https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg)
   - [COCO Names](https://github.com/pjreddie/darknet/blob/master/data/coco.names)

   Place these files in the `model` directory within your project:
   ```
   AcciAlert/
   ├── model/
   │   ├── yolov3.weights
   │   ├── yolov3.cfg
   │   └── coco.names
   ├── main.py
   ├── yolo_utils.py
   └── alert_utils.py
   ```

## Usage

### Running the Script

1. **Adjust the paths in `main.py` to point to your downloaded YOLO files and video file:**

   ```python
   weights_path = "model/yolov3.weights"
   cfg_path = "model/yolov3.cfg"
   names_path = "model/coco.names"
   video_path = "path/to/your/video.mp4"
   ```

2. **Run the script:**

   ```sh
   python main.py
   ```

### Example Output

When the script runs, it processes the video and detects objects in real-time. If an accident is detected, it alerts the user and visualizes the detection with a red bounding box. Non-accident objects are shown with green bounding boxes.

![AcciAlert Screenshot](images/alert.jpg)

## File Structure

```
AcciAlert/
├── model/
│   ├── yolov3.weights
│   ├── yolov3.cfg
│   └── coco.names
├── images/
│   └── accialert_screenshot.png
├── main.py
├── yolo_utils.py
└── alert_utils.py
```

## Project Structure

### main.py

The main script to load the YOLO model and process the video.

### yolo_utils.py

Utility functions for loading the YOLO model and processing video frames.

### alert_utils.py

Functions for handling alerts when an accident is detected.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- [YOLO: Real-Time Object Detection](https://pjreddie.com/darknet/yolo/)
- [OpenCV](https://opencv.org/)
- [COCO Dataset](https://cocodataset.org/)

## Contributing

We welcome contributions to AcciAlert! Please read the [contribution guidelines](CONTRIBUTING.md) before making any pull requests.

```


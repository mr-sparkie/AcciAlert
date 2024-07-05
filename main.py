import cv2
from yolo_utils import load_yolo_model, process_video

# Load YOLO model
weights_path = r"C:\users\froze\Downloads\yolov3.weights"
cfg_path = r"C:\users\froze\Downloads\yolov3.cfg"
names_path = r"C:\users\froze\Downloads\coco.names"
net, output_layers, classes = load_yolo_model(weights_path, cfg_path, names_path)

# Specify the class ID for "Accident" (assumed to be 1 here, adjust as needed)
accident_class_id = 1  # Adjust this ID based on your coco.names file

# Example usage with a video file
video_path = r"C:\Users\froze\Downloads\Minnesota car crash compilation April 2024.mp4"
process_video(video_path, net, output_layers, classes, accident_class_id)

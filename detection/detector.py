from ultralytics import YOLO
import cv2
import numpy as np
import torch
import utils.config as cfg

"""
This script handles target detection using YOLOv8 and optional head detection within bounding boxes using SIFT.
It includes functions for performing inference, processing detections, and calculating head positions.
"""

# Load the YOLO model
model = YOLO(cfg.WEIGHTS_PATH)  # Lightweight model for fast inference

# Check if CUDA is available and move the model to the appropriate device
if torch.cuda.is_available():
    print("CUDA is available. Using GPU.")
    print(torch.version.cuda)

    model.to('cuda')  # Move the model to the GPU
else:
    print("CUDA is not available. Using CPU.")
    model.to('cpu')  # Keep the model on the CPU

# Log which device the model is on
print(f"Model is on device: {next(model.parameters()).device}")

def detect_targets(frame, screen_center=(320, 320)):
    """
    Detects targets in a frame using YOLOv8.

    Parameters:
    - frame (np.ndarray or torch.Tensor): Input image or tensor.
    - screen_center (tuple): (x, y) representing the reference point in the resized image.

    Returns:
    - detections (list): List of detections in the format:
                         ((x_min, y_min, x_max, y_max), confidence, class_label, distance, head_position).
    - inference_time (float): Time taken for model inference in seconds.
    """
    # Move the frame to the GPU if CUDA is available
    if torch.cuda.is_available():
        frame = frame.to('cuda')

    # Perform inference using the YOLO model
    results = model.predict(source=frame, conf=cfg.MODEL_CONFIDENCE_THRESHOLD, classes=cfg.MODEL_CLASSES, verbose=cfg.VERBOSE)
    stats = results[0].speed  # Extract processing times (ms)
    inference_time = stats['inference'] / 1000.0  # Convert milliseconds to seconds

    detections = []
    ref_x, ref_y = screen_center  # Center of reference in the resized image (640x640)

    # Separate detections into persons and heads
    person_boxes = []
    head_boxes = []

    # Parse detections and categorize them
    for box in results[0].boxes.data:
        x_min, y_min, x_max, y_max, conf, cls = box.tolist()
        if int(cls) == 0:  # Class: Person
            person_boxes.append(((int(x_min), int(y_min), int(x_max), int(y_max)), conf))
        elif int(cls) == 7:  # Class: Head
            head_boxes.append(((int(x_min), int(y_min), int(x_max), int(y_max)), conf))

    # Match persons with detected heads
    for (x_min, y_min, x_max, y_max), conf in person_boxes:
        # Calculate the center of the bounding box for the person
        center_x = (x_min + x_max) / 2
        center_y = (y_min + y_max) / 2

        # Compute the distance to the center of the resized screen (optional)
        distance = ((center_x - ref_x) ** 2 + (center_y - ref_y) ** 2) ** 0.5

        # Search for a head within the bounding box of the person
        head_position = None
        for (hx_min, hy_min, hx_max, hy_max), h_conf in head_boxes:
            if x_min <= hx_min <= x_max and y_min <= hy_min <= y_max:
                head_x = int((hx_min + hx_max) / 2)
                head_y = int((hy_min + hy_max) / 2)
                head_position = (head_x, head_y)
                break

        # Save the full detection (distance included but not used for sorting)
        detections.append(((x_min, y_min, x_max, y_max), conf, 'person', distance, head_position))

    # Return detections as they are (unsorted)
    return detections, inference_time

def detect_head(frame, bbox):
    """
    Detect the head within a bounding box using SIFT.

    Parameters:
    - frame (np.ndarray): Preprocessed image in NumPy format.
    - bbox (tuple): Bounding box in the format (x_min, y_min, x_max, y_max).

    Returns:
    - sift_position (tuple): Average coordinates of keypoints detected by SIFT or None if none detected.
    """
    x_min, y_min, x_max, y_max = map(int, bbox)

    # Adjust the bounding box to focus on the upper quarter
    y_min_adjusted = max(0, y_min)
    y_max_adjusted = min(frame.shape[0], y_min + (y_max - y_min) // 4)
    x_min_adjusted = max(0, x_min)
    x_max_adjusted = min(frame.shape[1], x_max)

    head_region = frame[y_min_adjusted:y_max_adjusted, x_min_adjusted:x_max_adjusted]

    # Validate that the region is not empty
    if head_region.size == 0 or len(head_region.shape) < 2 or head_region.shape[0] == 0 or head_region.shape[1] == 0:
        print("Error: head_region has invalid dimensions.")
        return None

    try:
        # Convert to grayscale for SIFT
        gray_region = cv2.cvtColor(head_region, cv2.COLOR_BGR2GRAY)
    except Exception as e:
        print(f"Error converting to grayscale in detect_head: {e}")
        return None

    # Detect keypoints using SIFT
    sift = cv2.SIFT_create()
    keypoints = sift.detect(gray_region, None)

    if not keypoints:
        return None

    # Calculate the average position of the keypoints
    avg_x = int(np.mean([kp.pt[0] for kp in keypoints]))
    avg_y = int(np.mean([kp.pt[1] for kp in keypoints]))

    # Transform the average position to the original frame space
    sift_x = x_min_adjusted + avg_x
    sift_y = y_min_adjusted + avg_y

    return (sift_x, sift_y)

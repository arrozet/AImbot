from ultralytics import YOLO  # YOLOv8 library for object detection.

# Load the YOLO model with a pre-trained weight file.
model = YOLO('yolov8s.pt')  # Replace 'yolov8n.pt' with your desired YOLO model.

def detect_targets(frame):
    """
    Performs object detection on a given frame using the YOLOv8 model.

    Parameters:
    - frame: A NumPy array representing the image to process.

    Returns:
    - detections: A list of tuples with detected objects in the format:
                  ((x_min, y_min, x_max, y_max), confidence, class_label).
    """
    # Perform detection on the input frame with a confidence threshold of 0.3.
    results = model.predict(source=frame, conf=0.3)
    detections = []

    for box in results[0].boxes.data:
        # Extract bounding box coordinates, confidence, and class index.
        x_min, y_min, x_max, y_max, conf, cls = box.tolist()

        # Filter for class "person" (class index 0) with confidence above 0.3.
        if int(cls) == 0 and conf > 0.3:
            width = x_max - x_min
            height = y_max - y_min
            aspect_ratio = width / height

            # Filter detections based on realistic aspect ratios and minimum sizes.
            if 0.3 < aspect_ratio < 3 and width > 20 and height > 40:
                detections.append(((int(x_min), int(y_min), int(x_max), int(y_max)), conf, model.names[int(cls)]))

    return detections
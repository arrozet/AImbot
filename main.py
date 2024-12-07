from screen.capture import capture_screen  # Module for screen capturing.
from detection.detector import detect_targets  # Module for object detection.
from control.mouse import aim_and_shoot  # Module to control the mouse actions.
import cv2  # OpenCV library for image processing and visualization.

def main():
    """
    Main function for running real-time object detection and interaction.
    - Captures frames from the screen in real-time.
    - Detects objects using a YOLO model.
    - Simulates aiming and shooting at detected targets.
    - Displays processed frames with visual overlays.
    """
    print("Starting real-time detection. Press 'q' to quit.")

    while True:
        # Capture the entire screen at full resolution.
        frame = capture_screen()  # Returns a frame in 1920x1080 resolution.

        # Detect targets within the captured frame.
        detections = detect_targets(frame)

        for detection in detections:
            # Extract bounding box coordinates, confidence, and class label.
            (x_min, y_min, x_max, y_max), conf, cls = detection

            # Simulate aiming and shooting at the detected target.
            aim_and_shoot(((x_min, y_min, x_max, y_max), conf))

            # Draw bounding box on the captured frame.
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            # Add class and confidence as text near the bounding box.
            label = f"{cls}: {conf:.2f}"
            cv2.putText(frame, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Display the frame with detections in a window.
        cv2.imshow("Real-Time Detection", frame)

        # Exit the program when 'q' is pressed.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Clean up windows once the loop ends.
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

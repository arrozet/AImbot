import mss  # Library for screen capturing.
import numpy as np  # Library for handling image arrays.
import cv2  # OpenCV library for image processing.

def capture_screen(region=None):
    """
    Captures a screenshot of the entire screen or a specified region.

    Parameters:
    - region: A dictionary defining the region to capture (e.g., {'top': 0, 'left': 0, 'width': 800, 'height': 600}).
              If None, it defaults to the second monitor.

    Returns:
    - frame: A NumPy array representing the captured screen image in BGR format for OpenCV.
    """
    with mss.mss() as sct:
        # Define the monitor or region to capture.
        monitor = region if region else sct.monitors[2]  # Defaults to monitor 2.
        screenshot = sct.grab(monitor)  # Capture the screen or region.

        # Convert the screenshot to a format usable by OpenCV.
        frame = np.array(screenshot)  # Convert to NumPy array.
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)  # Convert from BGRA to BGR.

        return frame

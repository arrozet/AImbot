import utils.config as cfg
import numpy as np
import cv2

"""
This script provides utility functions for homography transformations, coordinate mapping, and detection processing.
It allows mapping coordinates from a model's frame size to screen size, scaling weapon masks, and handling detections.
"""

def compute_homography(frame_size, screen_size):
    """
    Compute the homography matrix from frame size to screen size.

    Parameters:
    - frame_size (tuple): (width, height) of the input frame.
    - screen_size (tuple): (width, height) of the output screen.

    Returns:
    - homography_matrix (np.ndarray): 3x3 homography matrix.
    """
    frame_width, frame_height = frame_size
    screen_width, screen_height = screen_size

    # Corners of the frame (source)
    frame_points = np.array([
        [0, 0],                         # Top-left corner
        [frame_width, 0],               # Top-right corner
        [frame_width, frame_height],    # Bottom-right corner
        [0, frame_height]               # Bottom-left corner
    ], dtype=np.float32)

    # Corners of the screen (destination)
    screen_points = np.array([
        [0, 0],                         # Top-left corner
        [screen_width, 0],              # Top-right corner
        [screen_width, screen_height],  # Bottom-right corner
        [0, screen_height]              # Bottom-left corner
    ], dtype=np.float32)

    # Calculate homography
    homography_matrix, _ = cv2.findHomography(frame_points, screen_points)
    return homography_matrix

def map_coordinates(coords, homography_matrix):
    """
    Map coordinates from the frame resolution to the full screen resolution using a homography matrix.

    Parameters:
    - coords (tuple): (x, y) coordinates in the frame resolution.
    - homography_matrix (np.ndarray): 3x3 homography matrix.

    Returns:
    - tuple: (mapped_x, mapped_y) coordinates mapped to the screen resolution.
    """
    point = np.array([[coords[0], coords[1]]], dtype=np.float32).reshape(-1, 1, 2)
    transformed_point = cv2.perspectiveTransform(point, homography_matrix)
    mapped_x, mapped_y = transformed_point[0][0]
    return int(mapped_x), int(mapped_y)

def scale_coordinates(screen_size, reference_size, reference_mask):
    """
    Scale weapon mask coordinates dynamically based on screen resolution.

    Parameters:
    - screen_size (tuple): (width, height) of the current screen resolution.
    - reference_size (tuple): (width, height) of the reference resolution.
    - reference_mask (tuple): Coordinates of the mask (x_start, y_start, x_end, y_end).

    Returns:
    - tuple: Scaled coordinates (x_start, y_start, x_end, y_end).
    """
    x_scale = screen_size[0] / reference_size[0]
    y_scale = screen_size[1] / reference_size[1]

    x_start, y_start, x_end, y_end = reference_mask
    return (
        int(x_start * x_scale),
        int(y_start * y_scale),
        int(x_end * x_scale),
        int(y_end * y_scale),
    )

def map_detections(detections):
    """
    Map detections from the model's frame resolution to screen resolution and compute distances.

    Parameters:
    - detections (list): List of detections in the format [(x_min, y_min, x_max, y_max), conf, cls, _, head_position].

    Returns:
    - list: Mapped detections sorted by distance to the screen center.
    """
    mapped_detections = []

    homography_matrix = compute_homography(cfg.TARGET_SIZE, cfg.SCREEN_SIZE)

    for detection in detections:
        (x_min, y_min, x_max, y_max), conf, cls, _, head_position = detection

        # Map bounding box coordinates from model resolution to screen resolution
        mapped_x_min, mapped_y_min = map_coordinates((x_min, y_min), homography_matrix)
        mapped_x_max, mapped_y_max = map_coordinates((x_max, y_max), homography_matrix)

        # Calculate the center of the mapped bounding box
        center_x = (mapped_x_min + mapped_x_max) // 2
        center_y = (mapped_y_min + mapped_y_max) // 2

        # Calculate the distance to the screen center
        distance = ((center_x - cfg.SCREEN_CENTER[0]) ** 2 + (center_y - cfg.SCREEN_CENTER[1]) ** 2) ** 0.5

        mapped_head_position = None
        if head_position:
            mapped_head_position = map_coordinates(head_position, homography_matrix)

        # Add mapped detection with its distance
        mapped_detections.append(((mapped_x_min, mapped_y_min, mapped_x_max, mapped_y_max), conf, cls, distance, mapped_head_position))

    # Sort detections by distance (index 3)
    mapped_detections.sort(key=lambda d: d[3])

    return mapped_detections

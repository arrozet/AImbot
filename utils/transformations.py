import utils.config as cfg
import numpy as np
import cv2

def compute_homography(frame_size, screen_size):
    """
    Compute the homography matrix from frame size to screen size.

    Parameters:
    - frame_size: Tuple (width, height) of the input frame.
    - screen_size: Tuple (width, height) of the output screen.

    Returns:
    - homography_matrix: 3x3 homography matrix.
    """
    frame_width, frame_height = frame_size
    screen_width, screen_height = screen_size

    # Esquinas del frame (fuente)
    frame_points = np.array([
        [0, 0],                         # Esquina superior izquierda
        [frame_width, 0],               # Esquina superior derecha
        [frame_width, frame_height],    # Esquina inferior derecha
        [0, frame_height]               # Esquina inferior izquierda
    ], dtype=np.float32)

    # Esquinas del screen (destino)
    screen_points = np.array([
        [0, 0],                         # Esquina superior izquierda
        [screen_width, 0],              # Esquina superior derecha
        [screen_width, screen_height],  # Esquina inferior derecha
        [0, screen_height]              # Esquina inferior izquierda
    ], dtype=np.float32)

    # Calcula la homografía
    homography_matrix, _ = cv2.findHomography(frame_points, screen_points)
    return homography_matrix

def map_coordinates(coords, homography_matrix):
    """
    Maps coordinates from the resolution of the captured frame to the full screen resolution.

    Parameters:
    - frame_size: A tuple (width, height) representing the resolution of the captured frame.
    - screen_size: A tuple (width, height) representing the resolution of the screen.
    - coords: A tuple (x, y) representing coordinates in the frame resolution.

    Returns:
    - A tuple (mapped_x, mapped_y) with coordinates mapped to the screen resolution.
    """
    point = np.array([[coords[0], coords[1]]], dtype=np.float32).reshape(-1, 1, 2)
    transformed_point = cv2.perspectiveTransform(point, homography_matrix)
    mapped_x, mapped_y = transformed_point[0][0]
    return int(mapped_x), int(mapped_y)

def scale_coordinates(screen_size, reference_size, reference_mask):
    """
    Scales the weapon mask coordinates dynamically based on screen resolution.

    Parameters:
    - screen_size: Tuple (width, height) representing the current screen resolution.

    Returns:
    - Tuple with scaled coordinates (x_start, y_start, x_end, y_end).
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
    mapped_detections = []

    homography_matrix = compute_homography(cfg.TARGET_SIZE, cfg.SCREEN_SIZE)
    # Calcula las coordenadas mapeadas y la distancia al centro de la pantalla
    for detection in detections:
        (x_min, y_min, x_max, y_max), conf, cls, _, head_position = detection

        # Mapea las coordenadas de la caja desde 640x640 a la pantalla
        mapped_x_min, mapped_y_min = map_coordinates((x_min, y_min), homography_matrix)
        mapped_x_max, mapped_y_max = map_coordinates((x_max, y_max), homography_matrix)

        # Calcula el centro de la caja mapeada
        center_x = (mapped_x_min + mapped_x_max) // 2
        center_y = (mapped_y_min + mapped_y_max) // 2

        # Calcula la distancia al centro de la pantalla
        distance = ((center_x - cfg.SCREEN_CENTER[0]) ** 2 + (center_y - cfg.SCREEN_CENTER[1]) ** 2) ** 0.5
        
        mapped_head_position = None
        if head_position:
            mapped_head_position = map_coordinates(head_position, homography_matrix)

        # Añade la detección mapeada con su distancia
        mapped_detections.append(((mapped_x_min, mapped_y_min, mapped_x_max, mapped_y_max), conf, cls, distance, mapped_head_position))

    mapped_detections.sort(key=lambda d: d[3])  # Ordena por la distancia (índice 3)

    return mapped_detections
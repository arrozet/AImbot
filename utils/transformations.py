def map_coordinates(frame_size, screen_size, coords):
    """
    Maps coordinates from the resolution of the captured frame to the full screen resolution.

    Parameters:
    - frame_size: A tuple (width, height) representing the resolution of the captured frame.
    - screen_size: A tuple (width, height) representing the resolution of the screen.
    - coords: A tuple (x, y) representing coordinates in the frame resolution.

    Returns:
    - A tuple (mapped_x, mapped_y) with coordinates mapped to the screen resolution.
    """
    frame_width, frame_height = frame_size
    screen_width, screen_height = screen_size

    x, y = coords
    mapped_x = int(x * screen_width / frame_width)  # Scale x-coordinate to screen width.
    mapped_y = int(y * screen_height / frame_height)  # Scale y-coordinate to screen height.

    return mapped_x, mapped_y

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
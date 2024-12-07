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

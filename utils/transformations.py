def map_coordinates(frame_size, screen_size, coords):
    """
    Mapea coordenadas del frame capturado a la pantalla completa.
    """
    frame_width, frame_height = frame_size
    screen_width, screen_height = screen_size

    x, y = coords
    mapped_x = int(x * screen_width / frame_width)
    mapped_y = int(y * screen_height / frame_height)

    return mapped_x, mapped_y

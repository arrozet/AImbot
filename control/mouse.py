from pynput.mouse import Controller, Button  # Library for mouse control.

# Initialize a mouse controller instance.
mouse = Controller()

def aim_and_shoot(target):
    """
    Moves the mouse to the center of a detected bounding box and simulates a click.

    Parameters:
    - target: A tuple ((x_min, y_min, x_max, y_max), conf) containing the bounding box coordinates
              and the confidence of the detection.
    """
    # Extract bounding box coordinates.
    (x_min, y_min, x_max, y_max), _ = target

    # Calculate the center of the bounding box.
    target_x = (x_min + x_max) // 2
    target_y = (y_min + y_max) // 2

    # Move the mouse to the center of the bounding box.
    #mouse.position = (target_x, target_y)

    # Simulate a left mouse click (uncomment if needed).
    # mouse.click(Button.left, 1)

from pynput.mouse import Controller, Button  # Library for mouse control.

# Initialize a mouse controller instance.
mouse = Controller()

def aim_and_shoot(target):
    """
    Moves the mouse to the specified coordinates and simulates a click.

    Parameters:
    - target: A tuple (x, y) representing the target coordinates.
    """
    if not isinstance(target, tuple) or len(target) != 2:
        raise ValueError("Target must be a tuple with two elements: (x, y).")

    x, y = target  # Unpack the target coordinates

    # Move the mouse to the specified coordinates.
    mouse.position = (x, y)

    # Simulate a left mouse click.
    #mouse.click(Button.left, 1)

    print(f"Moved to ({x}, {y}) and clicked.")

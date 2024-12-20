import ctypes
import os
import time
import utils.config as cfg

"""
This script defines a function to handle aiming and shooting behavior in a controlled environment.
It uses relative mouse movement and verifies alignment with a specified target before simulating a shot.
"""

def aim_and_shoot(mouse, target):
    """
    Aim towards the target and simulate a shot if the mouse is aligned with the target.

    Parameters:
        mouse (object): Mouse control object with methods to move and click.
        target (tuple): Coordinates (x, y) of the target.

    Raises:
        ValueError: If the target is not a tuple with two elements.
    """
    if not isinstance(target, tuple) or len(target) != 2:
        raise ValueError("Target must be a tuple with two elements: (x, y).")

    target_x, target_y = target

    # Calculate the relative offset from the screen center
    offset_x = target_x - cfg.SCREEN_CENTER[0]
    offset_y = target_y - cfg.SCREEN_CENTER[1]

    print("Target ({},{}) | Screen center {}".format(target_x, target_y, cfg.SCREEN_CENTER))
    print("Offset ({},{})".format(offset_x, offset_y))

    # Move the mouse using the relative offset
    if cfg.AIMING:
        mouse.mouse_move(offset_x, offset_y, True)  # True indicates relative movement

    # Verify if the screen center is within the tolerance of the target
    center_x, center_y = cfg.SCREEN_CENTER

    if cfg.SHOOTING and abs(center_x - target_x) <= cfg.TOLERANCE and abs(center_y - target_y) <= cfg.TOLERANCE:
        # If within tolerance, shoot
        mouse.mouse_left_click()
        print(f"Aimed and shot at target ({target_x}, {target_y}).")
    else:
        print(f"Target not aligned with screen center. Mouse at ({center_x}, {center_y}), Target at ({target_x}, {target_y}).")

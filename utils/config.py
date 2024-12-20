import utils.transformations as transformations
import keyboard

"""
Configuration file for centralizing adjustable parameters used in the project.

This file defines constants, parameters, and utility functions related to the project's configuration.
It includes settings for screen dimensions, model thresholds, input configurations, and event handling.
"""

# Function to handle keyboard events
def handle_keyboard_events(paused):
    """
    Handles keyboard events such as exiting the program or pausing/resuming the aimbot.

    Parameters:
    - paused (bool): Current state of the aimbot (paused or active).

    Returns:
    - paused (bool): Updated state of the aimbot.
    - exit_program (bool): Indicates whether to exit the program.
    """
    exit_program = False

    # Check if Ctrl + T is pressed to exit
    if keyboard.is_pressed('ctrl+t'):
        print("Ctrl + T pressed. Exiting program.")
        exit_program = True

    # Check if Ctrl + Q is pressed to pause/resume
    if keyboard.is_pressed('ctrl+q'):
        paused = not paused  # Toggle aimbot state
        if paused:
            print("Aimbot paused. Press 'Ctrl + Q' to resume.")
        else:
            print("Aimbot resumed.")
        # Wait briefly to prevent multiple detections of the same key press
        while keyboard.is_pressed('ctrl+q'):
            pass

    return paused, exit_program

# Screen settings
SCREEN_SIZE = (1920, 1200)  # Full screen resolution
TARGET_SIZE = (640, 640)    # Model input size
SCREEN_CENTER = (SCREEN_SIZE[0] // 2, SCREEN_SIZE[1] // 2)  # Center of the processed screen image
TARGET_CENTER = (TARGET_SIZE[0] // 2, TARGET_SIZE[1] // 2)  # Center of the model's input image
DISPLAY_SIZE = (SCREEN_SIZE[0] // 1, SCREEN_SIZE[1] // 1)  # Display size for debugging
TARGET_FPS = 60  # Desired frames per second

# Reference resolution for calculating dynamic mask positions
REFERENCE_SIZE = (1920, 1080)  # Reference resolution used as a baseline
REFERENCE_WEAPON_MASK = (1080, 600, 1680, 1070)  # Mask coordinates for 1920x1080 resolution

# Mask settings
WEAPON_MASK = transformations.scale_coordinates(SCREEN_SIZE, REFERENCE_SIZE, REFERENCE_WEAPON_MASK)

# Model settings
MODEL_CONFIDENCE_THRESHOLD = 0.5  # Confidence threshold for object detection
MODEL_CLASSES = [0, 7]  # Classes: 0 = Person, 7 = Head

# General settings
ENABLE_MASK = False  # Whether to use the mask during screen capture
DLL_PATH = "./control/rzctl.dll"  # Path to the DLL used for controlling input
WEIGHT_FILE = "sunxds_0.7.1.pt"  # File name of the weights used by the detection model
WEIGHTS_PATH = "./detection/weights/" + WEIGHT_FILE  # Full path to the weight file
IMAGES_PATH = "./utils/images/"  # Path to the images used in the project
TOLERANCE = 2  # Tolerance value for adjustments
TITLE_TEXT = "AImbot view"  # Title text for any displayed windows
DRAW = False  # Whether to draw visual elements (e.g., bounding boxes)
AIMING = True  # Initial state for aiming
SHOOTING = True  # Initial state for shooting
VERBOSE = True  # Whether to enable verbose logging

# Mouse sensitivity (adjust based on game configuration)
MOUSE_SENSITIVITY = 1  # Example sensitivity value

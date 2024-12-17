import utils.transformations as transformations
import keyboard

"""
Configuration file for centralizing adjustable parameters used in the project.
"""

# Función para manejar eventos de teclado
def handle_keyboard_events(paused):
    """
    Maneja eventos de teclado como salir del programa o pausar/reanudar.

    Parameters:
    - paused (bool): Estado actual del aimbot (pausado o no).

    Returns:
    - paused (bool): Estado actualizado del aimbot.
    - exit_program (bool): Indica si se debe salir del programa.
    """
    exit_program = False

    # Verifica si se presionó Ctrl + T para salir
    if keyboard.is_pressed('ctrl+t'):
        print("Ctrl + T pressed. Exiting program.")
        exit_program = True

    # Verifica si se presionó Ctrl + Q para pausar/reanudar
    if keyboard.is_pressed('ctrl+q'):
        paused = not paused  # Cambia el estado del aimbot
        if paused:
            print("Aimbot paused. Press 'Ctrl + Q' to resume.")
        else:
            print("Aimbot resumed.")
        # Espera un breve momento para evitar múltiples detecciones de la misma tecla
        while keyboard.is_pressed('ctrl+q'):
            pass

    return paused, exit_program


# Screen settings
SCREEN_SIZE = (1920, 1200)  # Full screen resolution
TARGET_SIZE = (640, 640)    # Model input size
SCREEN_CENTER = (SCREEN_SIZE[0] // 2, SCREEN_SIZE[1] // 2)  # Center of processed image
TARGET_CENTER = (TARGET_SIZE[0] // 2, TARGET_SIZE[1] // 2)  # Center of processed image
DISPLAY_SIZE = (SCREEN_SIZE[0] // 1, SCREEN_SIZE[1] // 1)
TARGET_FPS = 60

# Reference resolution for calculating dynamic mask positions
REFERENCE_SIZE = (1920, 1080)
REFERENCE_WEAPON_MASK = (1080, 600, 1680, 1070)  # Mask coordinates for 1920x1080

# Mask settings
WEAPON_MASK = transformations.scale_coordinates(SCREEN_SIZE, REFERENCE_SIZE, REFERENCE_WEAPON_MASK)


# Model settings
MODEL_CONFIDENCE_THRESHOLD = 0.5  # Confidence threshold for object detection
MODEL_CLASSES = [0, 7]  # Classes: 0 = Person, 7 = Head

# General settings
ENABLE_MASK = False  # Whether to use the mask during screen capture
DLL_PATH = "./control/rzctl.dll"
WEIGHT_FILE = "sunxds_0.7.1.pt"
WEIGHTS_PATH = "./detection/weights/" + WEIGHT_FILE
TOLERANCE = 20
TITLE_TEXT = "User View (Left) | Machine View (Right)"
DRAW = True

# Sensibilidad del ratón (ajusta según la configuración del juego)
MOUSE_SENSITIVITY = 1  # Por ejemplo



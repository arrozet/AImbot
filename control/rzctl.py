"""Forked from https://github.com/SunOner/sunone_aimbot/blob/main/logic/rzctl.py#L50
as well as the DLL"""
import ctypes

def enum(**enums):
    """
    Creates an enumeration type for defining constants.

    Args:
        enums: Key-value pairs for enumeration members.

    Returns:
        A dynamically created enumeration type.
    """
    return type('Enum', (), enums)

MOUSE_CLICK = enum(
    LEFT_DOWN=1,
    LEFT_UP=2,
    RIGHT_DOWN=4,
    RIGHT_UP=8,
    SCROLL_CLICK_DOWN=16,
    SCROLL_CLICK_UP=32,
    BACK_DOWN=64,
    BACK_UP=128,
    FORWARD_DOWN=256,
    FORWARD_UP=512,
    SCROLL_DOWN=4287104000,
    SCROLL_UP=7865344
)

class RazerMouse:
    """
    A class for interacting with a Razer mouse using a custom DLL.

    Provides methods for mouse movement, clicks, and keyboard input simulation.
    """
    def __init__(self, dll_path):
        """
        Initializes the RazerMouse object and sets up function bindings to the DLL.

        Args:
            dll_path (str): Path to the DLL used for mouse control.
        """
        self.dll = ctypes.WinDLL(dll_path)

        # Initialize DLL function bindings
        self.init = self.dll.init
        self.init.argtypes = []
        self.init.restype = ctypes.c_bool

        self.mouse_move = self.dll.mouse_move
        self.mouse_move.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_bool]
        self.mouse_move.restype = None

        self.mouse_click = self.dll.mouse_click
        self.mouse_click.argtypes = [ctypes.c_int]
        self.mouse_click.restype = None

        self.keyboard_input = self.dll.keyboard_input
        self.keyboard_input.argtypes = [ctypes.c_short, ctypes.c_int]
        self.keyboard_input.restype = None

    def init(self):
        """
        Finds the symbolic link that contains the name RZCONTROL and opens a handle to the respective device.

        Returns:
            bool: Indicates if a valid device handle was found.
        """
        return self.init()

    def mouse_move(self, x, y, relative_to_center):
        """
        Moves the mouse pointer to a specified position.

        Args:
            x (int): X-coordinate for the mouse pointer.
            y (int): Y-coordinate for the mouse pointer.
            relative_to_center (bool): If True, moves relative to the current position.

        Notes:
            - If relative_to_center is False, x and y represent positions between 1 and 65536, where (1, 1) is the top-left corner of the screen.
            - x and/or y cannot be 0 unless moving from a start point.
            - Behavior may vary with multiple monitors.
        """
        self.mouse_move(x, y, relative_to_center)

    def mouse_click(self, click_mask):
        """
        Simulates a mouse click using the specified click mask.

        Args:
            click_mask (MOUSE_CLICK): Specifies the type of click (e.g., left, right, or scroll).
        """
        self.mouse_click(click_mask)

    def mouse_left_click(self):
        """
        Simulates a left mouse click (press and release).
        """
        self.mouse_click(MOUSE_CLICK.LEFT_DOWN)
        self.mouse_click(MOUSE_CLICK.LEFT_UP)

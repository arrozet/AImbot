"""Forked from https://github.com/SunOner/sunone_aimbot/blob/main/logic/rzctl.py#L50
as well as the DLL"""
import ctypes

def enum(**enums):
    return type('Enum', (), enums)

MOUSE_CLICK = enum(
    LEFT_DOWN = 1,
	LEFT_UP = 2,
	RIGHT_DOWN = 4,
	RIGHT_UP = 8,
	SCROLL_CLICK_DOWN = 16,
	SCROLL_CLICK_UP = 32,
	BACK_DOWN = 64,
	BACK_UP = 128,
	FOWARD_DOWN = 256,
	FOWARD_UP = 512,
	SCROLL_DOWN = 4287104000,
	SCROLL_UP = 7865344
)
    
KEYBOARD_INPUT_TYPE = enum(
    KEYBOARD_DOWN = 0,
    KEYBOARD_UP = 1
)

class RazerMouse():
    def __init__(self, dll_path):
        self.dll = ctypes.WinDLL(dll_path)
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
        """finds the symbolic link that contains name RZCONTROL and opens a handle to the respective device

        Returns:
            bool: valid device handle
        """
        return self.init()
        
    def mouse_move(self, x, y, relative_to_center):
        """if going from point, x and y will be the offset from current mouse position
	       otherwise it will be a number in range of 1 to 65536, where 1, 1 is top left of screen
	       if using multiple monitors the input values remain the same, but outcome different, i just don't recommend bothering with this bs
	       note: x and/or y can not be 0 unless going from start point	

        Args:
            x (int)
            y (int)
            from_start_point (bool)
        """
        self.mouse_move(x, y, relative_to_center)

    def mouse_click(self, click_mask):
        """
        Args:
            click_mask (MOUSE_CLICK):
        """
        self.mouse_click(click_mask)

    def mouse_left_click(self):
        self.mouse_click(MOUSE_CLICK.LEFT_DOWN)
        self.mouse_click(MOUSE_CLICK.LEFT_UP)
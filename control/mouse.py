import ctypes
import os


class RazerMouseError(Exception):
    """Clase personalizada para manejar errores relacionados con RazerMouse."""
    pass


class RazerMouse:
    """
    Clase para controlar el ratón Razer mediante rzctl.dll.
    """

    def __init__(self, dll_path):
        """
        Inicializa el controlador de Razer usando rzctl.dll.

        Args:
            dll_path (str): Ruta al archivo rzctl.dll.
        """
        if not os.path.isfile(dll_path):
            raise RazerMouseError(f"DLL not found at {dll_path}")

        # Carga la DLL
        try:
            self.dll = ctypes.WinDLL(dll_path)
        except OSError as e:
            raise RazerMouseError(f"Failed to load rzctl.dll: {e}")

        # Definir las funciones de la DLL
        self._define_functions()

    def _define_functions(self):
        """Define las funciones exportadas por la DLL."""
        # Función init()
        self.dll.init.argtypes = []
        self.dll.init.restype = ctypes.c_bool

        # Función mouse_move()
        self.dll.mouse_move.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_bool]
        self.dll.mouse_move.restype = None

        # Función mouse_click()
        self.dll.mouse_click.argtypes = [ctypes.c_int]
        self.dll.mouse_click.restype = None

    def initialize(self):
        """Inicializa el controlador del dispositivo Razer."""
        if not self.dll.init():
            raise RazerMouseError("Failed to initialize Razer control. Is the device connected?")
        print("Razer control initialized successfully!")

    @staticmethod
    def get_screen_resolution():
        """Obtiene la resolución actual de la pantalla principal."""
        user32 = ctypes.windll.user32
        screen_width = user32.GetSystemMetrics(0)
        screen_height = user32.GetSystemMetrics(1)
        return screen_width, screen_height

    @staticmethod
    def normalize_coordinates(x, y):
        """Convierte coordenadas de píxeles a coordenadas normalizadas."""
        screen_width, screen_height = RazerMouse.get_screen_resolution()
        norm_x = int((x / screen_width) * 65535)
        norm_y = int((y / screen_height) * 65535)
        return norm_x, norm_y

    def move_mouse(self, x, y, from_start_point=False):
        """
        Mueve el ratón a una posición específica o relativa.

        Args:
            x (int): Coordenada X.
            y (int): Coordenada Y.
            from_start_point (bool): Si True, el movimiento es relativo; si False, es absoluto.
        """
        if not from_start_point:  # Normalizar solo si el movimiento es absoluto
            x, y = self.normalize_coordinates(x, y)

        self.dll.mouse_move(x, y, from_start_point)
        print(f"Mouse moved to ({x}, {y}) with relative: {from_start_point}")

    def click_mouse(self, click_type):
        """
        Simula un clic del ratón.

        Args:
            click_type (int): Tipo de clic (1 = Down, 2 = Up, etc.).
        """
        if click_type not in [1, 2]:
            raise ValueError("Invalid click type. Use 1 for Down and 2 for Up.")
        self.dll.mouse_click(click_type)
        print(f"Mouse click of type {click_type} executed.")

    def aim_and_shoot(self, target):
        """
        Mueve el ratón a las coordenadas del objetivo y simula un clic.

        Args:
            target (tuple): Coordenadas (x, y) del objetivo.
        """
        if not isinstance(target, tuple) or len(target) != 2:
            raise ValueError("Target must be a tuple with two elements: (x, y).")

        x, y = target
        self.move_mouse(x, y, from_start_point=False)
        """
        self.click_mouse(1)  # Left Click Down
        self.click_mouse(2)  # Left Click Up
        """
        print(f"Aimed and shot at target ({x}, {y}).")

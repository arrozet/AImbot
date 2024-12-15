import ctypes
import os
import time
import utils.config as cfg


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
    def get_screen_center():
        """Obtiene el centro de la pantalla."""
        return cfg.SCREEN_SIZE[0] // 2, cfg.SCREEN_SIZE[1] // 2

    def move_mouse(self, dx, dy, delay=0):
        """
        Mueve el ratón en un desplazamiento relativo.

        Args:
            dx (int): Desplazamiento en X.
            dy (int): Desplazamiento en Y.
            delay (float): Pausa entre movimientos, en segundos.
        """
        self.dll.mouse_move(dx, dy, True)
        if delay > 0:
            time.sleep(delay)

    def move_to_target(self, target_x, target_y, max_step=50, min_step=5, delay=0):
        """
        Mueve el ratón al objetivo en pasos dinámicos.

        Args:
            target_x (int): Coordenada X del objetivo.
            target_y (int): Coordenada Y del objetivo.
            max_step (int): Tamaño máximo del paso.
            min_step (int): Tamaño mínimo del paso.
            delay (float): Pausa entre movimientos, en segundos.
        """
        center_x, center_y = self.get_screen_center()
        dx = target_x - center_x
        dy = target_y - center_y

        while abs(dx) > min_step or abs(dy) > min_step:
            # Ajusta el tamaño del paso dinámicamente
            step_x = max(min(max_step, abs(dx)), min_step) * (1 if dx > 0 else -1)
            step_y = max(min(max_step, abs(dy)), min_step) * (1 if dy > 0 else -1)

            # Mueve en pasos
            self.move_mouse(step_x, step_y, delay)

            # Reduce la distancia restante
            dx -= step_x
            dy -= step_y
        
        # Simula un clic al llegar al objetivo
        self.left_click_mouse()
        print(f"Mouse moved to target ({target_x}, {target_y}).")

    
    
    def click_mouse(self, click_type):
        """
        Simula un clic del ratón.

        Args:
            click_type (int): Tipo de clic (1 = Down, 2 = Up, etc.).
        """
        if click_type not in [1, 2]:
            raise ValueError("Invalid click type. Use 1 for Down and 2 for Up.")
        self.dll.mouse_click(click_type)

    def left_click_mouse(self):
        """
        Simula un clic del ratón.

        Args:
            click_type (int): Tipo de clic (1 = Down, 2 = Up, etc.).
        """
        self.click_mouse(1)  # Left Click Down
        self.click_mouse(2)  # Left Click Up

    def aim_and_shoot(self, target):
        """
        Apunta hacia el objetivo y simula un disparo si el ratón está alineado con el objetivo.

        Args:
            target (tuple): Coordenadas (x, y) del objetivo.
        """
        if not isinstance(target, tuple) or len(target) != 2:
            raise ValueError("Target must be a tuple with two elements: (x, y).")

        target_x, target_y = target

        # Mueve el ratón hacia el objetivo
        self.move_to_target(target_x, target_y)

        # Verifica si el centro de la pantalla está sobre el objetivo
        center_x, center_y = self.get_screen_center()
        
        if abs(center_x - target_x) <= cfg.TOLERANCE and abs(center_y - target_y) <= cfg.TOLERANCE:
            # Si el centro está dentro de la tolerancia, dispara
            self.left_click_mouse()
            print(f"Aimed and shot at target ({target_x}, {target_y}).")
        else:
            print(f"Target not aligned with screen center. Mouse at ({center_x}, {center_y}), Target at ({target_x}, {target_y}).")

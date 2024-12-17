import ctypes
import os
import time
import utils.config as cfg

def aim_and_shoot(mouse, target):
    """
    Apunta hacia el objetivo y simula un disparo si el ratón está alineado con el objetivo.
    Args:
        target (tuple): Coordenadas (x, y) del objetivo.
    """
    if not isinstance(target, tuple) or len(target) != 2:
        raise ValueError("Target must be a tuple with two elements: (x, y).")

    target_x, target_y = target

    # Mueve el ratón hacia el objetivo
    print("Target ({},{}) | Screen center {}".format(target_x,target_y, cfg.SCREEN_CENTER))
    print("Offset ({},{})".format(cfg.SCREEN_CENTER[0]-target_x, cfg.SCREEN_CENTER[1]-target_y))
    #mouse.mouse_move(target_x, target_y, True)

    """    # Verifica si el centro de la pantalla está sobre el objetivo
    center_x, center_y = self.get_screen_center()

    if abs(center_x - target_x) <= cfg.TOLERANCE and abs(center_y - target_y) <= cfg.TOLERANCE:
        # Si el centro está dentro de la tolerancia, dispara
        self.left_click_mouse()
        print(f"Aimed and shot at target ({target_x}, {target_y}).")
    else:
        print(f"Target not aligned with screen center. Mouse at ({center_x}, {center_y}), Target at ({target_x}, {target_y}).")
    """

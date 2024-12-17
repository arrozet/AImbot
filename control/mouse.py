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

    # Calcula el desplazamiento relativo desde el centro de la pantalla
    offset_x = target_x - cfg.SCREEN_CENTER[0]
    offset_y = target_y - cfg.SCREEN_CENTER[1]

    print("Target ({},{}) | Screen center {}".format(target_x, target_y, cfg.SCREEN_CENTER))
    print("Offset ({},{})".format(offset_x, offset_y))

    # Mueve el ratón usando el desplazamiento relativo
    mouse.mouse_move(offset_x, offset_y, True)  # True indica movimiento relativo

    # Verifica si el centro de la pantalla está sobre el objetivo
    center_x, center_y = cfg.SCREEN_CENTER

    if abs(center_x - target_x) <= cfg.TOLERANCE and abs(center_y - target_y) <= cfg.TOLERANCE:
        # Si el centro está dentro de la tolerancia, dispara
        mouse.mouse_left_click()
        print(f"Aimed and shot at target ({target_x}, {target_y}).")
    else:
        print(f"Target not aligned with screen center. Mouse at ({center_x}, {center_y}), Target at ({target_x}, {target_y}).")
    

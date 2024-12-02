from pynput.mouse import Controller, Button

mouse = Controller()

def aim_and_shoot(target):
    """
    Mueve el ratón al centro del bounding box detectado y simula un disparo.
    
    Parámetros:
    - target: Una tupla ((x_min, y_min, x_max, y_max), conf) con las coordenadas del bounding box y la confianza.
    """
    # Extrae las coordenadas del bounding box
    (x_min, y_min, x_max, y_max), _ = target

    # Calcula el centro del bounding box
    target_x = (x_min + x_max) // 2
    target_y = (y_min + y_max) // 2

    # Mueve el ratón al centro del bounding box
    mouse.position = (target_x, target_y)

    # Simula el clic izquierdo (descomenta si lo necesitas)
    # mouse.click(Button.left, 1)

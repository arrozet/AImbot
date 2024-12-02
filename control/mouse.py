from pynput.mouse import Controller, Button

mouse = Controller()

def aim_and_shoot(target):
    # Calcula el centro del bounding box del objetivo
    (x_min, y_min, x_max, y_max), _ = target
    target_x = (x_min + x_max) // 2
    target_y = (y_min + y_max) // 2

    # Mueve el ratón al objetivo
    mouse.position = (target_x, target_y)

    # Dispara (click izquierdo)
    mouse.click(Button.left, 1)

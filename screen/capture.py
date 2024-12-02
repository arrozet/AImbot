import mss
import numpy as np
import cv2

def capture_screen(region=None):
    with mss.mss() as sct:
        # Define la región de captura (puedes ajustarla al área del juego)
        monitor = region if region else sct.monitors[1]
        screenshot = sct.grab(monitor)

        # Convierte la imagen a un formato compatible con OpenCV
        frame = np.array(screenshot)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        return frame

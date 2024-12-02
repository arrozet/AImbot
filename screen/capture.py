import mss
import numpy as np
import cv2

def capture_screen(region=None):
    with mss.mss() as sct:
        # Captura el monitor completo
        monitor = region if region else sct.monitors[2]  # Cambia el índice si usas otro monitor
        screenshot = sct.grab(monitor)

        # Convierte la imagen a formato OpenCV
        frame = np.array(screenshot)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        return frame

from ultralytics import YOLO
import cv2
import numpy as np
import mss

def main():
    # Carga el modelo YOLOv8n
    model = YOLO('yolov8n.pt')  # Asegúrate de que 'yolov8n.pt' esté disponible

    # Configuración de captura de pantalla
    with mss.mss() as sct:
        # Listar todos los monitores disponibles
        monitors = sct.monitors
        print("Monitores detectados:")
        for i, monitor in enumerate(monitors):
            print(f"Monitor {i}: {monitor}")
        
        # Selecciona el monitor correspondiente al de abajo (ajusta el índice según la lista)
        monitor = monitors[2]  # Normalmente, el índice 2 es el segundo monitor; verifica esto.

        print("Iniciando detección en tiempo real. Presiona 'q' para salir.")
        
        while True:
            # Captura de pantalla
            screenshot = np.array(sct.grab(monitor))
            frame = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)  # Convierte a formato BGR

            # Realiza la detección
            results = model.predict(source=frame, conf=0.5, verbose=False)  # Confianza mínima 50%

            # Dibuja las detecciones en el frame
            for box in results[0].boxes.xyxy:
                x_min, y_min, x_max, y_max = map(int, box)
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            
            # Muestra el frame con las detecciones
            cv2.imshow("Detección en tiempo real", frame)

            # Salir con 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

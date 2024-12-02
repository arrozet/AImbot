from screen.capture import capture_screen
from detection.detector import detect_targets
from control.mouse import aim_and_shoot
import cv2

def main():
    print("Iniciando detección en tiempo real. Presiona 'q' para salir.")

    while True:
        # Captura el frame de la pantalla completa
        frame = capture_screen()  # Captura la pantalla en resolución completa (1920x1080)

        # Detecta los objetivos directamente en la resolución completa
        detections = detect_targets(frame)

        for detection in detections:
            # Extrae el bounding box y la confianza
            (x_min, y_min, x_max, y_max), conf, cls = detection

            # Apunta y dispara directamente
            aim_and_shoot(((x_min, y_min, x_max, y_max), conf))

            # Dibuja el bounding box en el frame capturado
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            # Añade la clase y la confianza como texto
            label = f"{cls}: {conf:.2f}"
            cv2.putText(frame, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Muestra el frame con las detecciones
        cv2.imshow("Detección en tiempo real", frame)

        # Salir del programa con 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

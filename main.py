from screen.capture import capture_screen
from detection.detector import detect_targets
from control.mouse import aim_and_shoot
from utils.transformations import map_coordinates
import cv2
import time

def main():
    print("Starting real-time detection with a specific mask. Press 'q' to quit.")

    # Configuración de la pantalla y del modelo
    screen_size = (1920, 1080)  # Resolución completa de la pantalla
    region = None  # Captura toda la pantalla
    target_size = (640, 640)  # Tamaño de entrada del modelo

    # Coordenadas de la máscara (ajusta según sea necesario)
    # x_start, y_start, x_end, y_end
    mask_coords = (1080, 600, 1680, 1070)  # Ejemplo de máscara en la zona del arma y HUD

    frame_count = 0
    start_time = time.time()

    while True:
        # Captura el frame original y el frame procesado con la máscara aplicada
        frame_original, frame_processed = capture_screen(
            region=region, use_mask=True, mask_coords=mask_coords, target_size=target_size
        )
        if frame_original is None or frame_processed is None:
            continue

        # Detecta objetos en el frame procesado
        detections = detect_targets(frame_processed)

        # Dibuja detecciones en el frame original
        for detection in detections:
            (x_min, y_min, x_max, y_max), conf, cls = detection

            # Mapea las coordenadas desde el frame redimensionado (target_size) al original
            mapped_x_min, mapped_y_min = map_coordinates(target_size, frame_original.shape[1::-1], (x_min, y_min))
            mapped_x_max, mapped_y_max = map_coordinates(target_size, frame_original.shape[1::-1], (x_max, y_max))

            # Dibujar bounding boxes en el frame original
            cv2.rectangle(frame_original, (mapped_x_min, mapped_y_min), (mapped_x_max, mapped_y_max), (0, 255, 0), 2)
            label = f"{cls}: {conf:.2f}"
            cv2.putText(
                frame_original, label, (mapped_x_min, mapped_y_min - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
            )

        # Muestra el frame original completo
        cv2.imshow("Real-Time Detection", frame_original)

        # Calcula FPS cada segundo
        frame_count += 1
        elapsed_time = time.time() - start_time
        if elapsed_time >= 1.0:
            fps = frame_count / elapsed_time
            print(f"FPS: {fps:.2f}")
            frame_count = 0
            start_time = time.time()

        # Salida al presionar 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()


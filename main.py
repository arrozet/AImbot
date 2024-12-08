from screen.capture import capture_screen
from detection.detector import detect_targets, detect_head
from control.mouse import aim_and_shoot
from utils.transformations import map_coordinates
import cv2

def print_performance_summary(total_inference_time, frame_count):
    """
    Calcula y muestra un resumen de rendimiento con FPS promedio y tiempo de inferencia promedio.

    Parameters:
    - total_inference_time: Tiempo total de inferencia acumulado.
    - frame_count: Número total de frames procesados.
    """
    average_inference_time = total_inference_time / frame_count
    average_fps = 1 / average_inference_time

    # Formato estético para los resultados
    print("\n========== Performance Summary ==========")
    print(f"Average Inference Time per Frame: {average_inference_time * 1000:.2f} ms")
    print(f"Average FPS: {average_fps:.2f}")
    print("========================================")

def main():
    print("Starting real-time detection with a specific mask. Press 'q' to quit.")

    # Configuración de la pantalla y del modelo
    screen_size = (1920, 1080)  # Resolución completa de la pantalla
    target_size = (640, 640)  # Tamaño de entrada del modelo
    screen_center = (target_size[0] // 2, target_size[1] // 2)  # Centro de la imagen procesada

    # Coordenadas de la máscara (ajusta según sea necesario)
    # x_start, y_start, x_end, y_end
    weapon_mask = (1080, 600, 1680, 1070)  # Weapon mask

    frame_count = 0
    total_inference_time = 0  # Acumula el tiempo de inferencia de cada frame

    while True:
         # Captura el frame original y el frame procesado
        frame_original, frame_processed = capture_screen(region=None, use_mask=True, mask_coords=weapon_mask, target_size=target_size)
        if frame_original is None or frame_processed is None:
            continue

        # Detecta objetos ordenados por distancia y obtiene el tiempo de inferencia
        detections, inference_time = detect_targets(frame_processed, screen_center=screen_center)

        # Acumula el tiempo de inferencia
        total_inference_time += inference_time
        frame_count += 1

         # Procesa y dibuja las detecciones
        for detection in detections:
            (x_min, y_min, x_max, y_max), conf, cls, _, head_position = detection

            # Mapea las coordenadas desde 640x640 al tamaño original
            mapped_x_min, mapped_y_min = map_coordinates(target_size, screen_size, (x_min, y_min))
            mapped_x_max, mapped_y_max = map_coordinates(target_size, screen_size, (x_max, y_max))

            # Dibuja las bounding boxes en la imagen original
            cv2.rectangle(frame_original, (mapped_x_min, mapped_y_min), (mapped_x_max, mapped_y_max), (0, 255, 0), 2)
            label = f"{cls}: {conf:.2f}"
            cv2.putText(
                frame_original, label, (mapped_x_min, mapped_y_min - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
            )

            head_x, head_y = -1, -1

            if head_position:
                # Si se detectó la cabeza, dibuja el punto
                mapped_head_x, mapped_head_y = map_coordinates(target_size, screen_size, head_position)
                cv2.circle(frame_original, (mapped_head_x, mapped_head_y), 5, (0, 255, 0), -1)
                head_x = mapped_head_x
                head_y = mapped_head_y
            else:
                # Usa SIFT como respaldo
                sift_position = detect_head(frame_processed, (mapped_x_min, mapped_y_min, mapped_x_max, mapped_y_max))
                if sift_position:
                    mapped_sift_x, mapped_sift_y = map_coordinates(target_size, screen_size, sift_position)
                    cv2.circle(frame_original, (mapped_sift_x, mapped_sift_y), 5, (0, 0, 255), -1)
                    head_x = mapped_sift_x
                    head_y = mapped_sift_y

            aim_and_shoot((head_x, head_y))
            
            

        # Muestra el frame original completo
        cv2.imshow("Real-Time Detection", frame_original)

        # Salida al presionar 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Calcula los FPS promedio al final del programa
    print_performance_summary(total_inference_time, frame_count)
    
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()


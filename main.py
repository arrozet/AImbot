from screen.capture import process_frame
from detection.detector import detect_targets, detect_head
from control.mouse import RazerMouse
from utils.transformations import map_coordinates
import cv2
import torch
import utils.config as cfg
import bettercam

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
    device = None
    if torch.cuda.is_available():
        device = "CUDA"
    else:
        device = "CPU"

    print(f"Model running on {device}")
    print(f"Average Inference Time per Frame: {average_inference_time * 1000:.2f} ms")
    print(f"Average FPS: {average_fps:.2f}")
    print("========================================")

def main():
    print("Starting real-time detection with a specific mask. Press 'q' to quit.")
    
    # Inicializa la cámara con BetterCam
    camera = bettercam.create()
    camera.start(target_fps=cfg.TARGET_FPS)

    # Configura la ventana como redimensionable
    cv2.namedWindow("Real-Time Detection", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Real-Time Detection", *cfg.DISPLAY_SIZE)  # The * is to unpack value

    frame_count = 0
    total_inference_time = 0  # Acumula el tiempo de inferencia de cada frame

     # Inicializa el controlador de ratón Razer
    razer_mouse = RazerMouse(cfg.DLL_PATH)
    razer_mouse.initialize()

    try:
        while True:
            # Captura el frame original y el frame procesado
            frame = camera.get_latest_frame()
            frame_original, frame_processed = process_frame(frame, region=None, use_mask=cfg.ENABLE_MASK, mask_coords=cfg.WEAPON_MASK, target_size=cfg.TARGET_SIZE)
            if frame_original is None or frame_processed is None:
                continue

            # Detecta objetos ordenados por distancia y obtiene el tiempo de inferencia
            detections, inference_time = detect_targets(frame_processed, screen_center=cfg.SCREEN_CENTER)

            # Acumula el tiempo de inferencia
            total_inference_time += inference_time
            frame_count += 1

            # Procesa y dibuja las detecciones
            for detection in detections:
                (x_min, y_min, x_max, y_max), conf, cls, _, head_position = detection

                # Mapea las coordenadas desde 640x640 al tamaño original
                mapped_x_min, mapped_y_min = map_coordinates(cfg.TARGET_SIZE, cfg.SCREEN_SIZE, (x_min, y_min))
                mapped_x_max, mapped_y_max = map_coordinates(cfg.TARGET_SIZE, cfg.SCREEN_SIZE, (x_max, y_max))

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
                    mapped_head_x, mapped_head_y = map_coordinates(cfg.TARGET_SIZE, cfg.SCREEN_SIZE, head_position)
                    cv2.circle(frame_original, (mapped_head_x, mapped_head_y), 5, (0, 255, 0), -1)
                    head_x = mapped_head_x
                    head_y = mapped_head_y
                else:
                    # Usa SIFT como respaldo
                    sift_position = detect_head(frame_original, (mapped_x_min, mapped_y_min, mapped_x_max, mapped_y_max))
                    if sift_position:
                        sift_x, sift_y = sift_position
                        #map_coordinates(target_size, screen_size, sift_position)
                        print("Head is now at ({0},{1}) coordinates".format(sift_x,sift_y))
                        cv2.circle(frame_original, (sift_x, sift_y), 5, (0, 0, 255), -1)
                        head_x = sift_x
                        head_y = sift_y

                #print("Shooting at ({0},{1})".format(head_x,head_y))
                razer_mouse.aim_and_shoot((head_x, head_y))
                
            # Muestra el frame original completo
            cv2.imshow("Real-Time Detection", frame_original)

            # Salida al presionar 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        # Detiene la captura y libera recursos
        camera.stop()
        camera.release()
        cv2.destroyAllWindows()

        # Resumen de rendimiento
        print_performance_summary(total_inference_time, frame_count)


if __name__ == "__main__":
    main()


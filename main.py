from screen.capture import process_frame
from detection.detector import detect_targets, detect_head
from utils.transformations import map_detections
from control import mouse
from control import rzctl
import cv2
import torch
import utils.config as cfg
import bettercam
import keyboard
import numpy as np

def pad_image_to_match_height(img, target_height):
    """
    Rellena la imagen con bordes negros para igualar la altura objetivo.
    
    Args:
        img (np.ndarray): Imagen a rellenar.
        target_height (int): Altura deseada.
    
    Returns:
        np.ndarray: Imagen con padding aplicado.
    """
    current_height, current_width = img.shape[:2]
    padding_top = (target_height - current_height) // 2
    padding_bottom = target_height - current_height - padding_top
    return cv2.copyMakeBorder(img, padding_top, padding_bottom, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])

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
    print("Starting real-time detection with a specific mask. Press 'ctrl+q' to pause and resume. Press 'ctrl+p' to quit.")
    
    # Inicializa la cámara con BetterCam
    camera = bettercam.create(device_idx=0, output_idx=0)
    camera.start(target_fps=cfg.TARGET_FPS)


    if(cfg.DRAW):
        # Configura la ventana como redimensionable
        cv2.namedWindow(cfg.TITLE_TEXT, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(cfg.TITLE_TEXT, *cfg.DISPLAY_SIZE)  # The * is to unpack value


    frame_count = 0
    total_inference_time = 0  # Acumula el tiempo de inferencia de cada frame

     # Inicializa el controlador de ratón Razer
    razer_mouse = rzctl.RazerMouse(cfg.DLL_PATH)

    paused = False  # Variable para controlar el estado del aimbot
    
    try:
        paused = False
        while True:
            detected_by_sift = False
            paused, exit_program = cfg.handle_keyboard_events(paused)
            if exit_program:
                break
            if paused:
                continue
            
            # Captura el frame original y el frame procesado
            frame = camera.get_latest_frame()
                
            frame_eq, frame_processed = process_frame(frame, region=None, use_mask=cfg.ENABLE_MASK, mask_coords=cfg.WEAPON_MASK, target_size=cfg.TARGET_SIZE)
            if frame_eq is None or frame_processed is None:
                continue

            # Detecta objetos ordenados por distancia y obtiene el tiempo de inferencia
            detections, inference_time = detect_targets(frame_processed, screen_center=cfg.TARGET_CENTER)
            
            # Acumula el tiempo de inferencia
            total_inference_time += inference_time
            frame_count += 1

            # Lista para almacenar detecciones mapeadas y ordenadas
            ordered_mapped_detections = map_detections(detections)            

            head_positions = []
            # Procesa y dibuja las detecciones
            for detection in ordered_mapped_detections:
                (x_min, y_min, x_max, y_max), conf, cls, _, head_position = detection
                head_x, head_y = -1, -1
                if head_position:
                    (head_x, head_y) = head_position
                    head_positions.append((head_x,head_y))
                else:
                    # Usa SIFT como respaldo
                    sift_position = detect_head(frame_eq, (x_min, y_min, x_max, y_max))
                    if sift_position:
                        (head_x, head_y) = sift_position
                        head_positions.append((head_x,head_y))
                        detected_by_sift = True
                
                # Dibujo todo lo relativo a la detección
                if(cfg.DRAW):
                    color = (0, 255, 0)
                    # Si es la que está más cerca, ponla en roja
                    if detection == ordered_mapped_detections[0]:
                        color = (255, 0, 0)
                    
                    # Dibuja las bounding boxes en la imagen original
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)

                    # Pon la confianza de detección
                    label = f"{cls}: {conf:.2f}"
                    cv2.putText(
                        frame, label, (x_min, y_min - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
                    )

                    # Marca la cabeza
                    if(head_x != -1 and head_y != -1):
                        if detected_by_sift:
                            color = (0,0,255)
                        cv2.circle(frame, (head_x, head_y), 5, color, -1)

            # Disparo al que está más cerca si hay deteccione
            if head_positions != []:
                mouse.aim_and_shoot(razer_mouse, head_positions[0])

            # Muestro el frame
            if(cfg.DRAW):
                # Añade padding a la imagen de la maquina
                #frame_machine = pad_image_to_match_height(frame_machine, frame.shape[0])

                # Combinar las dos imágenes horizontalmente
                #combined_frame = np.hstack((frame, frame_machine))

                # Mostrar el frame combinado
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                cv2.imshow(cfg.TITLE_TEXT, frame)
                cv2.waitKey(1)

    finally:
        # Detiene la captura y libera recursos
        camera.stop()
        camera.release()
        cv2.destroyAllWindows()

        # Resumen de rendimiento
        print_performance_summary(total_inference_time, frame_count)


if __name__ == "__main__":
    main()


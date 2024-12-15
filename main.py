from screen.capture import process_frame
from detection.detector import detect_targets, detect_head
from control.mouse import RazerMouse
from utils.transformations import map_coordinates
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
    camera = bettercam.create()
    camera.start(target_fps=cfg.TARGET_FPS)

    if(cfg.DRAW):
        # Configura la ventana como redimensionable
        cv2.namedWindow(cfg.TITLE_TEXT, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(cfg.TITLE_TEXT, *cfg.DISPLAY_SIZE)  # The * is to unpack value


    frame_count = 0
    total_inference_time = 0  # Acumula el tiempo de inferencia de cada frame

     # Inicializa el controlador de ratón Razer
    razer_mouse = RazerMouse(cfg.DLL_PATH)
    razer_mouse.initialize()

    paused = False  # Variable para controlar el estado del aimbot
    
    try:
        while True:
            # Verifica si se presionó Ctrl + T para salir
            if keyboard.is_pressed('ctrl+t'):
                print("Ctrl + T pressed. Exiting program.")
                break

            # Verifica si se presionó Ctrl + Q para pausar/reanudar
            if keyboard.is_pressed('ctrl+q'):
                paused = not paused  # Cambia el estado del aimbot
                if paused:
                    print("Aimbot paused. Press 'Ctrl + Q' to resume.")
                else:
                    print("Aimbot resumed.")
                # Espera un breve momento para evitar múltiples detecciones de la misma tecla
                while keyboard.is_pressed('ctrl+q'):
                    pass

            # Si el aimbot está en pausa, no procesa los frames
            if paused:
                continue
            
            # Captura el frame original y el frame procesado
            frame = camera.get_latest_frame()
                
            frame_machine, frame_processed = process_frame(frame, region=None, use_mask=cfg.ENABLE_MASK, mask_coords=cfg.WEAPON_MASK, target_size=cfg.TARGET_SIZE)
            if frame_machine is None or frame_processed is None:
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

                if head_position:
                    # Si se detectó la cabeza, dibuja el punto
                    mapped_head_x, mapped_head_y = map_coordinates(cfg.TARGET_SIZE, cfg.SCREEN_SIZE, head_position)
                    
                    head_x = mapped_head_x
                    head_y = mapped_head_y
                else:
                    # Usa SIFT como respaldo
                    sift_position = detect_head(frame, (mapped_x_min, mapped_y_min, mapped_x_max, mapped_y_max))
                    if sift_position:
                        sift_x, sift_y = sift_position
                        head_x = sift_x
                        head_y = sift_y

                #print("Shooting at ({0},{1})".format(head_x,head_y))
                #razer_mouse.aim_and_shoot((head_x, head_y))

                # Dibujo todo lo relativo a la detección
                if(cfg.DRAW):
                    # Dibuja las bounding boxes en la imagen original
                    cv2.rectangle(frame, (mapped_x_min, mapped_y_min), (mapped_x_max, mapped_y_max), (0, 255, 0), 2)
                    label = f"{cls}: {conf:.2f}"
                    cv2.putText(
                        frame, label, (mapped_x_min, mapped_y_min - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
                    )

                    # Marca la cabeza
                    cv2.circle(frame, (head_x, head_y), 5, (0, 255, 0), -1)

            # Muestro el frame
            if(cfg.DRAW):
                # Añade padding a la imagen de la maquina
                frame_machine = pad_image_to_match_height(frame_machine, frame.shape[0])

                # Combinar las dos imágenes horizontalmente
                combined_frame = np.hstack((frame, frame_machine))

                # Mostrar el frame combinado
                cv2.imshow(cfg.TITLE_TEXT, combined_frame)
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


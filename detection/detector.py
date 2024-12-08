from ultralytics import YOLO
import cv2
import numpy as np

# Cargar el modelo YOLO
model = YOLO('sunxds_0.5.6.pt')  # Modelo ligero para inferencia rápida
model.to('cpu')  # Ejecutar en CPU

def detect_targets(frame, screen_center=(320, 320)):
    """
    Detecta personas en un frame utilizando YOLOv8 y las ordena por cercanía al centro de la imagen procesada.

    Parameters:
    - frame: Imagen en formato NumPy o tensor procesado.
    - screen_center: Coordenadas (x, y) del punto de referencia en la imagen redimensionada.

    Returns:
    - detections: Lista de detecciones ordenadas por distancia en formato:
                  ((x_min, y_min, x_max, y_max), confidence, class_label, distance).
    """
    results = model.predict(source=frame, conf=0.3, classes=[0,7])  # Detecta personas (0) y cabezas (7)
    stats = results[0].speed  # Extraer tiempos de procesamiento (ms)
    inference_time = stats['inference'] / 1000.0  # Convertir a segundos

    detections = []
    ref_x, ref_y = screen_center  # Centro de referencia en la imagen 640x640

    # Separar detecciones de personas y cabezas
    person_boxes = []
    head_boxes = []

    for box in results[0].boxes.data:
        x_min, y_min, x_max, y_max, conf, cls = box.tolist()
        if int(cls) == 0:  # Clase persona
            person_boxes.append(((int(x_min), int(y_min), int(x_max), int(y_max)), conf))
        elif int(cls) == 7:  # Clase cabeza
            head_boxes.append(((int(x_min), int(y_min), int(x_max), int(y_max)), conf))
    
    # Procesar cada persona y buscar cabezas asociadas
    for (x_min, y_min, x_max, y_max), conf in person_boxes:
        # Calcular el centro del bounding box de la persona
        center_x = (x_min + x_max) / 2
        center_y = (y_min + y_max) / 2

        # Calcular la distancia al centro de la pantalla redimensionada
        distance = ((center_x - ref_x) ** 2 + (center_y - ref_y) ** 2) ** 0.5

        # Buscar una cabeza dentro del bounding box de la persona
        head_position = None
        for (hx_min, hy_min, hx_max, hy_max), h_conf in head_boxes:
            if x_min <= hx_min <= x_max and y_min <= hy_min <= y_max:
                head_x = int((hx_min + hx_max) / 2)
                head_y = int((hy_min + hy_max) / 2)
                head_position = (head_x, head_y)
                break

        # Guardar detección completa
        detections.append(((x_min, y_min, x_max, y_max), conf, 'person', distance, head_position))
        
    # Ordenar las detecciones por distancia
    detections_sorted = sorted(detections, key=lambda d: d[3])  # Ordenar por distancia (índice 3)

    return detections_sorted, inference_time


def detect_head(frame, bbox):
    """
    Detecta la cabeza dentro de un bounding box usando SIFT.

    Parameters:
    - frame: Imagen en formato NumPy preprocesada.
    - bbox: Bounding box con formato (x_min, y_min, x_max, y_max).

    Returns:
    - sift_position: Coordenadas promedio de los puntos clave detectados por SIFT o None si no se detectan.
    """
    x_min, y_min, x_max, y_max = map(int, bbox)

    # Ajustar el bounding box para enfocarse en el cuarto superior
    head_region = frame[y_min:y_min + (y_max - y_min) // 4, x_min:x_max]

    # Validar que la región no esté vacía ni fuera de los límites
    if head_region is None or head_region.size == 0:
        print("Error: head_region está vacía o fuera de los límites.")
        return None

    # Verificar si las dimensiones de head_region son válidas
    if len(head_region.shape) < 2 or head_region.shape[0] == 0 or head_region.shape[1] == 0:
        print("Error: head_region tiene dimensiones no válidas.")
        return None

    try:
        # Convertir a escala de grises para SIFT
        gray_region = cv2.cvtColor(head_region, cv2.COLOR_BGR2GRAY)
    except Exception as e:
        print(f"Error al convertir a escala de grises en detect_head: {e}")
        return None

    # Detectar keypoints usando SIFT
    sift = cv2.SIFT_create()
    keypoints = sift.detect(gray_region, None)

    if not keypoints:
        return None

    # Calcular la posición promedio de los keypoints
    avg_x = int(np.mean([kp.pt[0] for kp in keypoints]))
    avg_y = int(np.mean([kp.pt[1] for kp in keypoints]))

    # Transformar la posición promedio al espacio original del frame
    sift_x = x_min + avg_x
    sift_y = y_min + avg_y

    return (sift_x, sift_y)

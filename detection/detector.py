from ultralytics import YOLO

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
    results = model.predict(source=frame, conf=0.3, classes=[0])  # Detecta solo personas
    stats = results[0].speed  # Extraer tiempos de procesamiento (ms)
    inference_time = stats['inference'] / 1000.0  # Convertir a segundos

    detections = []
    ref_x, ref_y = screen_center  # Centro de referencia en la imagen 640x640

    for box in results[0].boxes.data:
        x_min, y_min, x_max, y_max, conf, cls = box.tolist()

        # Calcular el centro del bounding box
        center_x = (x_min + x_max) / 2
        center_y = (y_min + y_max) / 2

        # Calcular la distancia al centro de la pantalla redimensionada
        distance = ((center_x - ref_x) ** 2 + (center_y - ref_y) ** 2) ** 0.5

        # Guardar detección con la distancia calculada
        detections.append(((int(x_min), int(y_min), int(x_max), int(y_max)), conf, model.names[int(cls)], distance))

    # Ordenar las detecciones por distancia
    detections_sorted = sorted(detections, key=lambda d: d[3])  # Ordenar por distancia (índice 3)

    return detections_sorted, inference_time

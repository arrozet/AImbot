from ultralytics import YOLO

# Cargar el modelo YOLO
model = YOLO('yolov8n.pt')  # Modelo ligero para inferencia rápida
model.to('cpu')  # Ejecutar en CPU

def detect_targets(frame):
    """
    Detecta personas en un frame utilizando YOLOv8.

    Parameters:
    - frame: Imagen en formato NumPy.

    Returns:
    - detections: Lista de detecciones con formato ((x_min, y_min, x_max, y_max), confidence, class_label).
    """
    # Realizar detección solo para la clase 'person' (clase 0)
    results = model.predict(source=frame, conf=0.3, classes=[0])  # Filtra solo personas
    detections = []

    for box in results[0].boxes.data:
        # Extraer coordenadas, confianza y clase
        x_min, y_min, x_max, y_max, conf, cls = box.tolist()
        if conf > 0.3:  # Aplicar umbral adicional
            detections.append(((int(x_min), int(y_min), int(x_max), int(y_max)), conf, model.names[int(cls)]))

    return detections

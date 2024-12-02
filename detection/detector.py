from ultralytics import YOLO

model = YOLO('yolov8n.pt')  # Cambia por el modelo que estés usando

def detect_targets(frame):
    # Realiza detección directamente en la resolución completa
    results = model.predict(source=frame, conf=0.3)
    detections = []
    for box in results[0].boxes.data:
        x_min, y_min, x_max, y_max, conf, cls = box.tolist()
        if int(cls) == 0 and conf > 0.3:  # Filtrar solo la clase 'person'
            detections.append(((int(x_min), int(y_min), int(x_max), int(y_max)), conf, model.names[int(cls)]))
    return detections

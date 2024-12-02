import torch

# Carga el modelo YOLOv5 (puedes cambiarlo por YOLOv8 si prefieres)
model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True)

def detect_targets(frame):
    # Realiza la detección
    results = model(frame)

    # Filtra detecciones (ajusta las clases según el juego)
    detections = []
    for *box, conf, cls in results.xyxy[0].numpy():
        if conf > 0.5:  # Umbral de confianza
            x_min, y_min, x_max, y_max = map(int, box)
            detections.append(((x_min, y_min, x_max, y_max), conf))

    return detections

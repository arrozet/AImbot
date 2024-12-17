import bettercam
import numpy as np
import cv2
import torch
import utils.config as cfg

def process_frame(frame, region=None, use_mask=False, mask_coords=None, target_size=cfg.TARGET_SIZE):
    """
    Captura un frame de la pantalla utilizando BetterCam y aplica preprocesamiento.

    Parameters:
    - region: Región específica para capturar (x, y, width, height). Si None, captura toda la pantalla.
    - use_mask: Si True, aplica una máscara para eliminar áreas no relevantes.
    - mask_coords: Coordenadas absolutas de la máscara (x_start, y_start, x_end, y_end).
    - target_size: Tamaño al que se debe redimensionar la imagen (width, height).

    Returns:
    - frame_original: Imagen original capturada (NumPy array).
    - frame_processed: Imagen procesada lista para el modelo (tensor de PyTorch).
    """
    try:
        # Convertir a NumPy
        frame_original = np.array(frame, dtype=np.uint8)

        # Aplica máscara si se especifica
        if use_mask and mask_coords is not None:
            x_start, y_start, x_end, y_end = mask_coords
            frame_original[y_start:y_end, x_start:x_end] = 0  # Bloquea la región especificada

        # Ecualiza el histograma del canal de luminosidad
        frame_yuv = cv2.cvtColor(frame_original, cv2.COLOR_BGR2YUV)
        frame_yuv[:, :, 0] = cv2.equalizeHist(frame_yuv[:, :, 0])  # Ecualiza solo el canal Y
        frame_equalized = cv2.cvtColor(frame_yuv, cv2.COLOR_YUV2BGR)
        

        # Redimensionar para el modelo
        frame_resized = cv2.resize(frame_equalized, target_size)

        # Normalizar y convertir a tensor
        frame_processed = torch.from_numpy(frame_resized).float().div(255).permute(2, 0, 1)
        frame_processed = frame_processed.unsqueeze(0)  # Añadir batch dimension

        return frame_equalized,frame_processed
    except Exception as e:
        print(f"Error en capture_screen: {e}")
        return None, None





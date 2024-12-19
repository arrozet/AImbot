import bettercam
import numpy as np
import cv2
import torch
import utils.config as cfg

"""
This script captures and processes frames from the screen using BetterCam.
It applies optional masking, histogram equalization, resizing, and normalization to prepare the frame for model inference.
"""

def process_frame(frame, region=None, use_mask=False, mask_coords=None, target_size=cfg.TARGET_SIZE):
    """
    Captures a frame from the screen using BetterCam and applies preprocessing.

    Parameters:
    - frame (array): The input frame to process.
    - region (tuple, optional): Specific region to capture (x, y, width, height). If None, captures the entire screen.
    - use_mask (bool, optional): If True, applies a mask to block irrelevant areas.
    - mask_coords (tuple, optional): Absolute coordinates of the mask (x_start, y_start, x_end, y_end).
    - target_size (tuple, optional): Size to resize the image to (width, height).

    Returns:
    - frame_original (array): The original captured image (NumPy array).
    - frame_processed (torch.Tensor): The processed image ready for the model (PyTorch tensor).
    """
    try:
        # Convert to NumPy array
        frame_original = np.array(frame, dtype=np.uint8)

        # Apply mask if specified
        if use_mask and mask_coords is not None:
            x_start, y_start, x_end, y_end = mask_coords
            frame_original[y_start:y_end, x_start:x_end] = 0  # Block the specified region

        # Equalize the histogram of the luminance channel
        frame_yuv = cv2.cvtColor(frame_original, cv2.COLOR_BGR2YUV)
        frame_yuv[:, :, 0] = cv2.equalizeHist(frame_yuv[:, :, 0])  # Equalize only the Y channel
        frame_equalized = cv2.cvtColor(frame_yuv, cv2.COLOR_YUV2BGR)

        # Resize for the model
        frame_resized = cv2.resize(frame_equalized, target_size)

        # Normalize and convert to tensor
        frame_processed = torch.from_numpy(frame_resized).float().div(255).permute(2, 0, 1)
        frame_processed = frame_processed.unsqueeze(0)  # Add batch dimension

        return frame_equalized, frame_processed
    except Exception as e:
        print(f"Error in process_frame: {e}")
        return None, None

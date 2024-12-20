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

"""
This script implements real-time detection and interaction using BetterCam, YOLO, and a Razer mouse controller.
It processes frames, detects targets, and optionally performs actions such as aiming and shooting.
"""

def pad_image_to_match_height(img, target_height):
    """
    Pads the image with black borders to match the target height.

    Args:
        img (np.ndarray): Image to pad.
        target_height (int): Desired height.

    Returns:
        np.ndarray: Image with padding applied.
    """
    current_height, current_width = img.shape[:2]
    padding_top = (target_height - current_height) // 2
    padding_bottom = target_height - current_height - padding_top
    return cv2.copyMakeBorder(img, padding_top, padding_bottom, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])

def print_performance_summary(total_inference_time, frame_count):
    """
    Calculates and displays a performance summary with average FPS and inference time.

    Parameters:
    - total_inference_time: Accumulated inference time.
    - frame_count: Total number of processed frames.
    """
    average_inference_time = total_inference_time / frame_count
    average_fps = 1 / average_inference_time

    print("\n========== Performance Summary ==========")
    device = "CUDA" if torch.cuda.is_available() else "CPU"
    print(f"Model running on {device}")
    print(f"Average Inference Time per Frame: {average_inference_time * 1000:.2f} ms")
    print(f"Average FPS: {average_fps:.2f}")
    print("========================================")

def main():
    print("Starting real-time detection with a specific mask. Press 'ctrl+q' to pause and resume. Press 'ctrl+p' to quit.")

    # Initialize camera with BetterCam
    camera = bettercam.create(device_idx=0, output_idx=0)
    camera.start(target_fps=cfg.TARGET_FPS)

    if cfg.DRAW:
        # Ensures visualization is enabled to display bounding boxes, labels, and head markers on the frame.
        cv2.namedWindow(cfg.TITLE_TEXT, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(cfg.TITLE_TEXT, *cfg.DISPLAY_SIZE)  # Unpack dimensions

    frame_count = 0
    total_inference_time = 0

    # Initialize Razer mouse controller
    razer_mouse = rzctl.RazerMouse(cfg.DLL_PATH)

    paused = False

    try:
        while True:
            detected_by_sift = False
            paused, exit_program = cfg.handle_keyboard_events(paused)
            if exit_program:
                break
            if paused:
                continue

            # Capture original and processed frames
            frame = camera.get_latest_frame()
            frame_eq, frame_processed = process_frame(frame, region=None, use_mask=cfg.ENABLE_MASK, mask_coords=cfg.WEAPON_MASK, target_size=cfg.TARGET_SIZE)
            if frame_eq is None or frame_processed is None:
                continue

            # Detect targets ordered by distance and get inference time
            detections, inference_time = detect_targets(frame_processed, screen_center=cfg.TARGET_CENTER)

            # Update inference time and frame count for performance metrics
            total_inference_time += inference_time
            frame_count += 1

            # Map detections to screen coordinates and order them by proximity
            ordered_mapped_detections = map_detections(detections)

            head_positions = []  # Stores positions of detected heads
            for detection in ordered_mapped_detections:
                (x_min, y_min, x_max, y_max), conf, cls, _, head_position = detection
                head_x, head_y = -1, -1
                if head_position:
                    # Use model-detected head position
                    head_x, head_y = head_position
                    head_positions.append((head_x, head_y))
                else:
                    # Use SIFT to detect head position if not provided by the model
                    sift_position = detect_head(frame_eq, (x_min, y_min, x_max, y_max))
                    if sift_position:
                        head_x, head_y = sift_position
                        head_positions.append((head_x, head_y))
                        detected_by_sift = True

                if cfg.DRAW:
                    # Choose color based on detection priority
                    color = (0, 255, 0)
                    if detection == ordered_mapped_detections[0]:
                        color = (255, 0, 0)  # Highlight closest detection

                    # Draw bounding box around detected object
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)

                    # Add detection label with class and confidence
                    label = f"{cls}: {conf:.2f}"
                    cv2.putText(frame, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                    # Mark detected head position
                    if head_x != -1 and head_y != -1:
                        color = (0, 0, 255) if detected_by_sift else (0, 255, 0)
                        cv2.circle(frame, (head_x, head_y), 5, color, -1)

            # Aim and shoot at the closest detected head if conditions are met
            if head_positions:
                mouse.aim_and_shoot(razer_mouse, head_positions[0])

            if cfg.DRAW:
                # Convert frame to RGB and display it in a window
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                cv2.imshow(cfg.TITLE_TEXT, frame)
                cv2.waitKey(1)

    finally:
        # Stop camera and release resources
        camera.stop()
        camera.release()
        cv2.destroyAllWindows()

        # Print performance summary after exiting
        print_performance_summary(total_inference_time, frame_count)

if __name__ == "__main__":
    main()

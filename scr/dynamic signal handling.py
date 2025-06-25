import sys
import os
import cv2
import numpy as np
import time
import threading
from ultralytics import YOLO

sys.path.append(os.path.join(os.getcwd(), "sort"))
from sort import Sort  # SORT tracker

# === Settings ===
vehicle_model_path = "yolov8m.pt"
video_paths = [
    "E:/internships project/tamizhan skills/SendAnywhere_122409/vid1.mp4",
    "E:/internships project/tamizhan skills/SendAnywhere_122409/vid2.mp4",
    "E:/internships project/tamizhan skills/SendAnywhere_122409/vid3.mp4",
    "E:/internships project/tamizhan skills/SendAnywhere_122409/vid1.mp4"
]
DISPLAY_WIDTH, DISPLAY_HEIGHT = 640, 360
VEHICLE_CLASSES = [2, 3, 5, 7]
PPM = 8
FRAME_RATE = 30

vehicle_model = YOLO(vehicle_model_path)
tracker = Sort(max_age=5, min_hits=3, iou_threshold=0.3)
caps = [cv2.VideoCapture(path) for path in video_paths]
last_frames = [None] * 4  # To store last frame for red/yellow
traffic_signals = [1, 0, 0, 0]  # Lane 1 green
lock = threading.Lock()
previous_positions = {}

def create_signal_image(lane_idx, frame, signal_state):
    color_map = {0: (0, 0, 255), 1: (0, 255, 0), 2: (0, 255, 255)}
    label_map = {0: "RED", 1: "GREEN", 2: "YELLOW"}
    if frame is None:
        frame = np.zeros((DISPLAY_HEIGHT, DISPLAY_WIDTH, 3), dtype=np.uint8)
    resized = cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
    label = f"Lane {lane_idx + 1} - {label_map[signal_state]}"
    cv2.putText(resized, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color_map[signal_state], 3)
    return resized

def traffic_control():
    global traffic_signals
    while True:
        for lane in range(4):
            with lock:
                traffic_signals = [0] * 4
                traffic_signals[lane] = 1
            time.sleep(7)
            with lock:
                traffic_signals[lane] = 2
            time.sleep(2)
            with lock:
                traffic_signals[lane] = 0

def estimate_speed(prev, curr, ppm, fps):
    dx = curr[0] - prev[0]
    dy = curr[1] - prev[1]
    pixel_dist = np.sqrt(dx ** 2 + dy ** 2)
    meters = pixel_dist / ppm
    speed_mps = meters * fps
    return speed_mps * 3.6

def process_lanes():
    global last_frames
    while True:
        display_frames = []

        for i, cap in enumerate(caps):
            with lock:
                signal = traffic_signals[i]

            if signal == 1:  # GREEN
                ret, frame = cap.read()
                if not ret:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = cap.read()
                if not ret:
                    frame = np.zeros((DISPLAY_HEIGHT, DISPLAY_WIDTH, 3), dtype=np.uint8)

                last_frames[i] = frame.copy()

                results = vehicle_model(frame, verbose=False)
                detections = []

                for result in results:
                    for box, conf, cls in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):
                        if int(cls) in VEHICLE_CLASSES and conf > 0.4:
                            x1, y1, x2, y2 = box
                            detections.append([x1.item(), y1.item(), x2.item(), y2.item(), conf.item()])

                detections = np.array(detections) if detections else np.empty((0, 5))
                tracked_objects = tracker.update(detections)
                speeds = {}

                for obj in tracked_objects:
                    x1, y1, x2, y2, track_id = obj
                    cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                    centroid = (cx, cy)

                    if track_id in previous_positions:
                        speeds[track_id] = estimate_speed(previous_positions[track_id], centroid, PPM, FRAME_RATE)
                    previous_positions[track_id] = centroid

                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                    label = f"ID {int(track_id)} | {speeds.get(track_id, 0):.1f} km/h"
                    cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                display_frames.append(create_signal_image(i, frame, signal))

            else:
                frame = last_frames[i]
                display_frames.append(create_signal_image(i, frame, signal))

        top = np.hstack((display_frames[0], display_frames[1]))
        bottom = np.hstack((display_frames[2], display_frames[3]))
        grid = np.vstack((top, bottom))

        cv2.imshow("Traffic Monitoring System", grid)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    for cap in caps:
        cap.release()
    cv2.destroyAllWindows()

# === Start ===
controller_thread = threading.Thread(target=traffic_control, daemon=True)
controller_thread.start()
process_lanes()

import cv2
import numpy as np
from ultralytics import YOLO

# Load the helmet detection model (Update the path to your .pt file)
helmet_model = YOLO("E:/internships project/tamizhan skills/SendAnywhere_122409/hemletYoloV8_100epochs.pt")  # Just the filename if it's in the same folder

# Start webcam or use a video file (0 = default webcam) gve 1 or 2 if your using secondary camera

cap = cv2.VideoCapture(0)

# Define Region of Interest (ROI) bounds
roi_y1, roi_y2 = 50, 300  # Adjust these based on expected head area

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to read from webcam or video")
        break

    # Resize for consistency
    resized = cv2.resize(frame, (640, 480))

    # Extract the Region of Interest
    roi = resized[roi_y1:roi_y2, :]

    # Run helmet detection only in ROI
    results = helmet_model.predict(roi, conf=0.4)

    boxes = []
    for result in results:
        if result.boxes is not None:
            for det in result.boxes.data.cpu().numpy():
                x1, y1, x2, y2 = map(int, det[:4])
                conf = float(det[4])
                cls_id = int(det[5])
                label = helmet_model.model.names.get(cls_id, "Unknown").lower()

                # Map box coords back to full frame
                y1 += roi_y1
                y2 += roi_y1
                boxes.append([x1, y1, x2, y2, conf, label])

    # Apply Non-Maximum Suppression (NMS)
    nms_threshold = 0.4
    if boxes:
        nms_indices = cv2.dnn.NMSBoxes(
            bboxes=[box[:4] for box in boxes],
            scores=[box[4] for box in boxes],
            score_threshold=0.4,
            nms_threshold=nms_threshold
        )

        # If valid NMS results found
        if nms_indices is not None and len(nms_indices) > 0:
            # Flatten if it’s a list of lists
            if isinstance(nms_indices, (list, tuple)):
                nms_indices = np.array(nms_indices).flatten()

            for i in nms_indices:
                x1, y1, x2, y2, conf, label = boxes[i]
                color = (0, 255, 0) if label == "helmet" else (0, 0, 255)
                tag = "Helmet" if label == "helmet" else "No Helmet"
                cv2.rectangle(resized, (x1, y1), (x2, y2), color, 2)
                cv2.putText(resized, f"{tag} ({conf:.2f})", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Show the frame
    cv2.imshow("Helmet Detection (Real-time)", resized)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()

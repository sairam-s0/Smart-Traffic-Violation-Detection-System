import os
os.environ['PADDLEOCR_HOME'] = './.paddleocr'

import cv2
import torch
import random
import qrcode
import numpy as np
import requests
from datetime import datetime
from paddleocr import PaddleOCR
from ultralytics import YOLO

# Load models
vehicle_model = YOLO('yolov8n')  # YOLOv8 Vehicle detection model
helmet_model = YOLO('E:/internships project/tamizhan skills/SendAnywhere_122409/helmetdetectionmodel.pt')  # YOLOv8 Helmet detection model
plate_model = YOLO('E:/internships project/runs/detect/lcmodel2/weights/lcmodel2.pt')  # YOLOv8 Number plate detection model
seatbelt_model = YOLO("E:/internships project/tamizhan skills/SendAnywhere_122409/runs/runs/detect/lcmodel/weights/seatbelttrainedbymistake.pt")  # YOLOv8 Seatbelt detection model
ocr = PaddleOCR(use_angle_cls=True, lang='en')

# Constants
VEHICLE_CLASSES = [2, 3]  # Motorbike and Car class IDs (2 = Car, 3 = Motorcycle)
HELMET_CONF = 0.4
PLATE_CONF = 0.4
SEATBELT_CONF = 0.4  # Set your seatbelt detection confidence threshold

# Telegram Bot Config
TELEGRAM_BOT_TOKEN = "your bot token "
TELEGRAM_CHAT_ID = "your id s u can give in a list f have many"

# Random Place List
places = [
    ("Chennai", "Teynampet Traffic Police Station, Anna Salai, Chennai – 600018"),
    ("Madurai", "Tallakulam Traffic Police Station, Madurai – 625002"),
    ("Coimbatore", "RS Puram Traffic Police Station, Coimbatore – 641002"),
    ("Trichy", "Trichy South Traffic Police Station, Near Main Bazaar, Trichy – 620001"),
    ("Salem", "Hasthampatti Traffic Police Station, Salem – 636007"),
]

# QR code generator
def generate_qr(text):
    qr = qrcode.QRCode(version=1, box_size=2, border=1)
    qr.add_data(text)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")
    qr_image = np.array(img.convert('RGB'))
    return qr_image

# Telegram alert function
def send_violation_alert(image, plate_text="Unknown"):
    place, station = random.choice(places)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    fine_payment_link = f"https://traffic.tn.gov.in/payfine/{random.randint(10000,99999)}"

    qr_img = generate_qr(fine_payment_link)
    qr_img = cv2.resize(qr_img, (100, 100))

    h, w, _ = image.shape
    x_offset = w - qr_img.shape[1] - 10
    y_offset = h - qr_img.shape[0] - 10
    image[y_offset:y_offset+qr_img.shape[0], x_offset:x_offset+qr_img.shape[1]] = qr_img

    caption = f"""🚨 *Traffic Violation Detected* 🚨

*Violation:* No Helmet or No Seatbelt
*Time:* {now}
*Place:* {place}, Tamil Nadu
*Fine Amount:* ₹1000

*License Plate:* {plate_text}

👉 _To appeal: Call 1800-100-1234 or visit [www.traffic.tn.gov.in/appeal](https://traffic.tn.gov.in/appeal)_  

💰 *Pay Fine At:*
{station}
"""

    _, img_encoded = cv2.imencode('.jpg', image)
    files = {'photo': ('violation.jpg', img_encoded.tobytes())}
    data = {'chat_id': TELEGRAM_CHAT_ID, 'caption': caption, 'parse_mode': 'Markdown'}

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
    response = requests.post(url, files=files, data=data)

    if not response.ok:
        print("Failed to send Telegram alert:", response.text)

# Main Detection
cap = cv2.VideoCapture(0)  # Use '0' for the default camera feed
if not cap.isOpened():
    print("Error: Unable to access the camera.")
    exit()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to read frame from camera.")
        break

    frame = cv2.resize(frame, (640, 480))
    small = frame.copy()

    vehicle_results = vehicle_model.predict(small, conf=0.4, verbose=False)[0]

    for box in vehicle_results.boxes:
        cls_id = int(box.cls[0].cpu().numpy())
        bx1, by1, bx2, by2 = map(int, box.xyxy[0].cpu().numpy())

        if cls_id in VEHICLE_CLASSES:
            # Detect car or motorcycle
            vehicle_crop = small[by1:by2, bx1:bx2]

            if cls_id == 2:  # Car detected
                # Seatbelt detection for cars
                seatbelt_results = seatbelt_model.predict(vehicle_crop, conf=SEATBELT_CONF, verbose=False)[0]

                seatbelt_detected = False
                for sbox in seatbelt_results.boxes:
                    seatbelt_detected = True  # Found seatbelt

                if not seatbelt_detected:
                    # No seatbelt detected → Violation!
                    cv2.rectangle(small, (bx1, by1), (bx2, by2), (0, 0, 255), 2)
                    cv2.putText(small, "No Seatbelt", (bx1, by1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                    plate_text = "Unknown"

                    # Number plate detection (same as before)
                    plate_results = plate_model.predict(vehicle_crop, conf=PLATE_CONF, verbose=False)[0]

                    for pbox in plate_results.boxes:
                        px1, py1, px2, py2 = map(int, pbox.xyxy[0].cpu().numpy())
                        gx1, gy1 = bx1 + px1, by1 + py1
                        gx2, gy2 = bx1 + px2, by1 + py2

                        cv2.rectangle(small, (gx1, gy1), (gx2, gy2), (0, 255, 255), 2)

                        plate_crop = vehicle_crop[py1:py2, px1:px2]
                        if plate_crop.size > 0:
                            try:
                                plate_crop_rgb = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2RGB)
                                plate_result = ocr.ocr(plate_crop_rgb, cls=True)
                                if plate_result and isinstance(plate_result[0], list) and len(plate_result[0]) > 0:
                                    plate_text = plate_result[0][0][1][0].strip()
                                    cv2.putText(small, plate_text, (gx1, gy1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                            except Exception as e:
                                print("OCR Error:", e)

                    # Send to Telegram
                    send_violation_alert(small.copy(), plate_text)

            elif cls_id == 3:  # Motorcycle detected
                # Helmet detection for motorcycles (same logic as before)
                helmet_results = helmet_model.predict(vehicle_crop, conf=HELMET_CONF, verbose=False)[0]

                helmet_detected = False
                for hbox in helmet_results.boxes:
                    helmet_detected = True  # Found helmet inside bike

                if not helmet_detected:
                    # No helmet detected → Violation!
                    cv2.rectangle(small, (bx1, by1), (bx2, by2), (0, 0, 255), 2)
                    cv2.putText(small, "No Helmet", (bx1, by1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                    plate_text = "Unknown"

                    # Number plate detection (same as before)
                    plate_results = plate_model.predict(vehicle_crop, conf=PLATE_CONF, verbose=False)[0]

                    for pbox in plate_results.boxes:
                        px1, py1, px2, py2 = map(int, pbox.xyxy[0].cpu().numpy())
                        gx1, gy1 = bx1 + px1, by1 + py1
                        gx2, gy2 = bx1 + px2, by1 + py2

                        cv2.rectangle(small, (gx1, gy1), (gx2, gy2), (0, 255, 255), 2)

                        plate_crop = vehicle_crop[py1:py2, px1:px2]
                        if plate_crop.size > 0:
                            try:
                                plate_crop_rgb = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2RGB)
                                plate_result = ocr.ocr(plate_crop_rgb, cls=True)
                                if plate_result and isinstance(plate_result[0], list) and len(plate_result[0]) > 0:
                                    plate_text = plate_result[0][0][1][0].strip()
                                    cv2.putText(small, plate_text, (gx1, gy1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                            except Exception as e:
                                print("OCR Error:", e)

                    # Send to Telegram
                    send_violation_alert(small.copy(), plate_text)

    cv2.imshow('Traffic Monitoring', small)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

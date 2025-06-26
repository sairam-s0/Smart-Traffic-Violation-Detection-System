# 🚦 Smart Traffic Violation Detection System

This Python-based system uses real-time object detection and OCR to identify traffic rule violations such as:
- **No Helmet (Motorcycles)**
- **No Seatbelt (Cars)**
- **License Plate Recognition**
- **Violation Alerts via Telegram (with QR code for fine payment)**

---

## 🔧 Features

- YOLOv8-based detection of:
  - Vehicles (cars, bikes)
  - Helmets
  - Seatbelts
  - Number plates
- OCR with PaddleOCR for plate text recognition
- Violation notification sent via Telegram with:
  - Annotated image
  - Location and time
  - Fine QR code

---

## 📁 Project Structure
helmetdetection-using-yolo8m-opencv/
├──datafortesting-and-trainning/
├──testing/
├──train/
├── models/                 # Pretrained YOLOv8 models for helmet, seatbelt, vehicle, and license plate detection
├── scr/
├── main.py                 # Main script for video stream processin
├──dynamic signal handling.py
├──helmetvc.py
├── requirements.txt        # Required Python packages
└── README.md               # Project documentation



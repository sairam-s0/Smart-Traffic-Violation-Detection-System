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
├── datafortesting-and-trainning/       # Datasets for training and testing
├── testing/                            # Testing scripts and sample test data
├── train/                              # Training scripts and configurations
├── models/                             # Pretrained YOLOv8 models (helmet, seatbelt, vehicle, plate)
├── scr/                                # Supporting scripts (utilities, helpers, etc.)
├── main.py                             # Main script for real-time video stream processing
├── dynamic signal handling.py          # Module for adaptive traffic signal control
├── helmetvc.py                         # Helmet violation checker logic
├── requirements.txt                    # Python dependencies
└── README.md                           # Project documentation



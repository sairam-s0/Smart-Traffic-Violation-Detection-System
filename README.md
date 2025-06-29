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

```text
helmetdetection-using-yolo8m-opencv/
├── datafortesting-and-trainning/
├── testing/
├── train/
├── models/
├── scr/
├── main.py
├── dynamic signal handling.py
├── helmetvc.py
├── requirements.txt
└── README.md
```
---
## IMPORTANT 
- To create a telegram bot follow the instruction in the wiki page of this repo



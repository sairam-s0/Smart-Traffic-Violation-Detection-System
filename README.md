# 🚦 Smart Traffic Violation Detection System

An AI-powered traffic surveillance system that detects multiple types of traffic violations using computer vision and deep learning models. Each violation type is handled by a separate, independently optimized model to allow flexibility, better benchmarking, and modular updates.

---

## 🧠 Core Features

- 🎯 **Helmet Detection** – Detects riders without helmets using YOLOv8.
- 😷 **Face Mask Detection** – Identifies whether individuals are wearing masks (ResNet).
- 🧍 **Triple Riding Detection** – Detects more than 2 people on a bike (YOLO/ResNet).
- 🚫 **Red Light Violation Detection** – Detects vehicle movement beyond stop-line (OpenCV logic).
- 🔍 **Modular Architecture** – Each model runs independently; combine or deploy separately.

---

## 🗂️ Project Structure

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
- Check out the [The wiki Page](https://github.com/sairam-s0/Smart-Traffic-Violation-Detection-System/wiki/telegram-bot-instructions#creating-telegram-bot)


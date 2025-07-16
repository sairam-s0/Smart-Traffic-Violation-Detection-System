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
---

## 🛠 Installation

1. Clone the repo

   ```bash
   git clone https://github.com/sairam-s0/Smart-Traffic-Violation-Detection-System.git
   cd Smart-Traffic-Violation-Detection-System
   ```

2. Install dependencies

   ```bash
   pip install -r requirements.txt
   ```

3. Download model weights
   *(Place your YOLO and other model weights in the appropriate folders)*

---

## 🚀 How to Run

Each module runs independently:

```bash
# Helmet detection
cd Helmet-Detection
python helmet_detect.py --source your_video.mp4

# Mask detection
cd ../seatbelt-Detection
python mask_detect.py --source your_video.mp4

# Red light violation detection
cd ../Red-Light-Violation
python redlight_detect.py --source your_video.mp4
```
---
## MODEL EVALUATION

## 🧩 Why Modular?

Instead of a single monolithic pipeline, this system treats each violation type as a separate module for:

* Easier maintenance and debugging
* Model-specific training & optimization
* Scalable deployment (microservices, edge devices)

---
## 🧩 Why Modular?

Instead of a single monolithic pipeline, this system treats each violation type as a separate module for:

* Easier maintenance and debugging
* Model-specific training & optimization
* Scalable deployment (microservices, edge devices)

---
## 📜 License

This project is licensed under the **MIT License**. See [LICENSE](./LICENSE) for details.

---

## 🙋‍♂️ Author

**Sairam**
GitHub: [@sairam-s0](https://github.com/sairam-s0)

---

## 🤝 Contributions

Pull requests, suggestions, and bug reports are welcome.
Please open an issue or submit a PR with improvements or additional models.




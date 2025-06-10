# YOLOv4 Object Detection with OpenCV

This project demonstrates real-time object detection using YOLOv4 with OpenCV in Python. It loads pretrained weights and config from the Darknet YOLOv4 model and detects objects in live webcam feed.

## 🚀 Features

- Real-time object detection from webcam
- Uses pretrained `yolov4.weights` and `yolov4.cfg`
- Detects objects using COCO dataset (`coco.names`)
- Built with OpenCV’s DNN module (no darknet build needed)

## 🧾 Requirements

- Python 3.x
- OpenCV (cv2)
- NumPy

Install dependencies:
```bash
pip install opencv-python numpy
```

## 🗂️ Project Structure

```
YOLO_PROJECT2/
├── yolov4.cfg
├── yolov4.weights
├── coco.names
├── object_detection.py
└── venv/
```

## 🎯 How to Run

1. Make sure you’re in the virtual environment:
```bash
venv\Scripts\activate  # On Windows
```

2. Run the detection script:
```bash
python object_detection.py
```

It will open your webcam and start detecting objects with bounding boxes.

## 🧠 Model Info

- Model: YOLOv4
- Framework: OpenCV DNN
- Dataset: COCO (80 object classes)

## 📌 Note

If OpenCV gives a **"transpose weights not implemented"** error, try using official YOLOv3 instead, or convert to ONNX if needed.

---



## 📁 Source

Based on: [AlexeyAB's YOLOv4](https://github.com/AlexeyAB/darknet) and OpenCV’s DNN module.


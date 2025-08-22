# human-gender-detector
Real-time human detection, tracking, and gender classification with YOLOv8, EfficientNetV2-L, and ByteTrack.
# Real-Time Gender Tracker  

🚀 **Real-time human detection, tracking, and gender classification** using **YOLOv8**, **EfficientNetV2-L**, and **ByteTrack**.  

This project combines state-of-the-art object detection and tracking with a custom gender classification model to create a full computer vision pipeline.  

---

## 🔍 Overview  
- **YOLOv8** → Detects humans in video frames.  
- **ByteTrack** → Assigns unique IDs and tracks humans across frames.  
- **EfficientNetV2-L (pretrained on 180k+ images)** → Classifies gender (Male/Female).  
- Combined → A real-time system that detects, tracks, and classifies humans in videos.  

---

## 🎯 What I Learned  
- Fundamentals of **Object Detection** using YOLOv8.  
- Basics of **tracking algorithms** with ByteTrack.  
- How to **integrate multiple models** (detector + tracker + classifier) into one pipeline.  
- Differences between **Ultralytics YOLO** and raw **PyTorch workflows**.  
- How to **optimize batch sizes** to avoid GPU crashes.  

---

## ⚡ Challenges Faced  
- ❌ Training crashed due to a **too large batch size**.  
   - ✅ Solved by reducing batch size to **4**.  
- ❌ Initially, working with Ultralytics YOLO felt different compared to my usual PyTorch training.  
   - ✅ Learned how to use **Ultralytics training & inference system**.  
- ✅ Apart from this, I faced fewer issues since I had already built and trained a **gender classification model** earlier.  

---

## 📊 Dataset  
- **Gender Classifier (EfficientNetV2-L)** trained on **180k+ custom images**.  
- **YOLOv8 Human Detector** trained on **90k+ images** for human bounding boxes.  

---

## 🛠️ Tech Stack  
- [YOLOv8](https://github.com/ultralytics/ultralytics)  
- [EfficientNetV2-L](https://arxiv.org/abs/2104.00298)  
- [ByteTrack](https://github.com/ifzhang/ByteTrack)  
- [PyTorch](https://pytorch.org/)  
- [OpenCV](https://opencv.org/)  

---

## 🚀 Results  
📌 The model can:  
- Detect humans in real-time.  
- Track individuals across frames with unique IDs.  
- Classify gender for each tracked person.  

🎥 Example demo video available soon.  

---

## 📂 Future Work  
- Deploy model on **Hugging Face Spaces**.  
- Release complete training + inference **code on GitHub**.  
- Extend pipeline for **age classification** or **multi-attribute recognition**.  

---

## 🏷️ Tags  
`#ComputerVision` `#DeepLearning` `#YOLOv8` `#ByteTrack` `#EfficientNet` `#PyTorch` `#AIProjects` `#ModelDeployment`  

---

✨ *This project shows how multiple computer vision techniques can be combined into a powerful real-time application.*  

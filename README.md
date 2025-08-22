# 👤 Real-Time Gender Tracker  

🎯 **Live demo (Hugging Face):**  
👉 [**Click here to test the app**](https://zain1133604-gender-classification-tracker.hf.space/docs)  

---

## 📌 Overview  

This project is a **real-time human detection, tracking, and gender classification system** built using:  
- 🧠 **YOLOv8** for human detection  
- 🎯 **ByteTrack** for multi-object tracking  
- 👨‍🦰 **EfficientNetV2-L** (trained on 180k+ images) for gender classification  
- 🔄 Integrated into one pipeline to **detect, track, and classify humans in real-time videos**  

We are reusing the **pre-trained gender classification model** from our [Gender Classification Repository](https://github.com/zain1133604/efficientnetv2-gender-classification) (trained earlier on a custom dataset), and combining it with YOLOv8 and ByteTrack.  

---

## 🎥 Demo Preview  
🧪 Test the full pipeline in action here:  
👉 [**Hugging Face Demo**](https://zain1133604-gender-classification-tracker.hf.space/docs)  

---

## 🧠 What I Learned  
During this project, I learned:  
- ✅ The basics of **object detection** using YOLOv8.  
- ✅ How to use **ByteTrack** to assign IDs and track people across frames.  
- ✅ The difference between **Ultralytics YOLO training system** and raw **PyTorch training**.  
- ✅ How to **combine multiple models** (detector + tracker + classifier) in one pipeline.  
- ✅ How to **optimize batch sizes** to prevent training crashes.  

---

## ⚠️ Challenges I Faced  
- ❌ Training crashed initially due to a **large batch size**.  
   - ✅ Fixed by reducing batch size to **4**.  
- ❌ Ultralytics was **different from my usual PyTorch workflow**.  
   - ✅ Took some time to learn how it works, but now I’m comfortable.  
- ✅ No major issues with gender classification, since I had **already trained that model** in a previous project.  

---

## 📊 Dataset Details  
- 🧍 **YOLOv8 Human Detector** trained on **90k+ images** (custom dataset).  
- 🧑‍🤝‍🧑 **Gender Classifier (EfficientNetV2-L)** trained earlier on **180k+ images** (custom dataset).  
- Both datasets were **manually curated and labeled** for better accuracy.  

---

## 🛠️ Tech Stack  
- [YOLOv8](https://github.com/ultralytics/ultralytics) – Object detection  
- [ByteTrack](https://github.com/ifzhang/ByteTrack) – Object tracking  
- [EfficientNetV2-L](https://arxiv.org/abs/2104.00298) – Gender classification  
- [PyTorch](https://pytorch.org/) – Training framework  
- [OpenCV](https://opencv.org/) – Video processing  

---

## 🧪 Features  
✔️ Detects humans in video frames  
✔️ Tracks each human with a unique ID  
✔️ Classifies gender for every tracked person  
✔️ Works in real-time  
✔️ Ready to deploy on web (via Hugging Face)  

---

## 🏷️ Tags  
`#ComputerVision` `#YOLOv8` `#ByteTrack` `#EfficientNet`  
`#PyTorch` `#DeepLearning` `#AIProjects` `#GenderClassification`  
`#ModelDeployment` `#HuggingFace` `#OpenCV` `#ObjectDetection`  

---

✨ *This project shows how to integrate detection, tracking, and classification into one real-time AI pipeline.*  
🔥 *Made with passion and deep learning.*  

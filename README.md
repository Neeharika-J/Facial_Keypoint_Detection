# 🎭 Facial Keypoint Detection

A real-time **Facial Keypoint Detection** application built using **PyTorch**, **ResNet-18**, **ONNX Runtime**, and **OpenCV**. The model is trained on the **Kaggle Facial Keypoints Detection** dataset to predict **15 facial landmarks** from a live webcam feed.

---

## ✨ Features

* 🎥 Real-time facial keypoint detection
* 🧠 Transfer learning with **ResNet-18**
* 🔄 Data augmentation using **Albumentations**
* 🎯 Hyperparameter optimization with **Optuna**
* 📊 Experiment tracking with **MLflow** and **DagsHub**
* ⚡ Fast inference using **ONNX Runtime**
* 🌐 Interactive web application built with **Streamlit**
* 👤 Automatic face detection using **OpenCV Haar Cascade**
* 📍 Real-time visualization of predicted facial landmarks

---

## 📂 Dataset

The model was trained on the **Kaggle Facial Keypoints Detection** dataset.

🔗 https://www.kaggle.com/competitions/facial-keypoints-detection

---

## 🔗 Other links

**Read the complete blog here:** *(Add your Medium article link.)*
**See the streamlit version here:** *(https://facialkeypointdetection-3ktroyrkokkmjyirbllqgx.streamlit.app/)*


## 📁 Project Structure

```text
.
├── 📂 models/
├── 🐍 main.py              # Local OpenCV application
├── 🌐 streamlit.py         # Streamlit web application
├── 🤖 model.onnx           # ONNX model used for inference
├── 📄 README.md
├── 📦 requirements.txt
└── 🚫 .gitignore
```

---

## 📝 Notes

### 📌 Running the Project

* Clone the repository together with the **model.onnx** file.
* The application performs inference using **ONNX Runtime** for faster predictions.

### 💻 Two Available Versions

#### 🖥️ Local OpenCV Version (`main.py`)

* ✅ Smooth real-time performance
* ✅ Recommended for the best user experience

#### 🌐 Streamlit Version (`streamlit.py`)

* Interactive browser-based interface
* Easy to deploy and demonstrate
* Slightly slower than the local implementation

### ⚠️ Streamlit Performance

📝 Note: The performance of the Streamlit application depends on the execution environment. During development, the local Streamlit version exhibited higher latency due to WebRTC overhead. However, the deployed Streamlit Cloud application performed noticeably smoother in our testing. The Streamlit application introduces additional latency because every frame passes through the following pipeline:

```text
📷 Browser Camera
        ↓
🌐 WebRTC
        ↓
🐍 Python
        ↓
🤖 ONNX Runtime
        ↓
🎯 Draw Facial Keypoints
        ↓
🌐 WebRTC
        ↓
🖥️ Browser Display
```

This extra processing overhead makes the Streamlit version slower than the native OpenCV implementation.

### 🚀 Performance Optimizations Attempted

Several optimizations were implemented to reduce latency:

* ✅ Detect faces every 5th frame instead of every frame
* ✅ Reduced webcam resolution
* ✅ Adjusted camera frame rate
* ✅ Enabled asynchronous video processing (`async_processing=True`)

Although these improvements reduced lag, the native OpenCV application continues to provide the smoothest real-time experience.

---

## 🛠️ Technologies Used

* 🐍 Python
* 🔥 PyTorch
* 👁️ OpenCV
* ⚡ ONNX Runtime
* 🌐 Streamlit
* 🖼️ Albumentations
* 🎯 Optuna
* 📊 MLflow
* 🐶 DagsHub

---

## 📖 Project Blog

Want to understand how this project was built from start to finish?

The Medium article covers:

* 📂 Dataset exploration
* 🧹 Data preprocessing
* 🖼️ Data augmentation
* 🧠 Model architecture
* 🎯 Hyperparameter tuning with Optuna
* 📊 Experiment tracking with MLflow & DagsHub
* ⚡ ONNX model export
* 🎥 Real-time deployment with OpenCV & Streamlit
---

⭐ If you found this project helpful, consider giving the repository a **Star** on GitHub!

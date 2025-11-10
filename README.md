# Real-Time Deep Learning-Based Face Detection and Recognition with Integrated Liveness Detection for Attendance System

## 📘 Overview
This project is a **real-time facial recognition attendance system** developed as part of a Final Year Project (FYP) in 2025.  
It integrates **face detection**, **deep learning-based face recognition**, and **liveness detection** (blink, smile, and head movement) to prevent spoofing attacks such as photo or video impersonation.  

The system provides automated and secure attendance marking using a **Flask-based web application** and **CNN-based face embedding model** optimized for low-latency real-time performance.

---

## 🚀 Key Features
- **Real-Time Face Detection & Recognition:** Uses dlib HOG/CNN detectors and deep feature embeddings for accurate recognition.  
- **68-Point Facial Landmark Liveness Detection:** Detects eye blinks, mouth movement, and head pose using dlib landmarks.  
- **End-to-End Pipeline:** From video capture → preprocessing → face detection → recognition → liveness check → attendance logging.  
- **Flask Web Interface:** Displays live video feed, recognized staff, and attendance logging.  
- **Low-Latency Deployment:** Optimized for real-time performance on CPU/GPU.

---

## 🧠 Technical Stack
| Component | Technology |
|------------|-------------|
| **Programming Language** | Python |
| **Frameworks** | TensorFlow, Flask |
| **Face Detection** | MTCNN |
| **Liveness Detection** | Blink (EAR), Smile (MAR), Head Pose (SolvePnP) |
| **Recognition Model** | Custom CNN with Residual Blocks |
| **Database** | SQLite |
| **Libraries** | OpenCV, Dlib, NumPy, Pandas, Scikit-learn |

---

## 🏗️ System Architecture
1. **Face Detection** (Dlib HOG/CNN)
2. Facial Landmark Detection (68-point detector) 
3. **Liveness Detection** → EAR, MAR, and head movement analyzed to confirm live human presence.  
4. **Feature Extraction** → CNN model converts detected faces into 1024D embeddings.  
5. **Face Recognition** → Cosine similarity used for identity verification.  
6. **Attendance Logging** → Flask app updates attendance records in SQLite database.  

---

## 📂 Project Structure
```
📁 RealTimeFaceRecognition
├── get_faces.py                # Face registration and dataset creation
├── feature_extraction_to_csv.py# Embedding extraction and storage
├── model_retrained.py          # CNN model training and retraining
├── app.py                      # Flask web application
├── static/                     # Frontend assets (CSS, JS, images)
├── templates/                  # HTML templates
├── database/attendance.db      # SQLite database
├── model/face_model.h5         # Trained CNN model
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

---

## ⚙️ Installation & Setup
### 1️⃣ Clone Repository
```bash
git clone https://github.com/your-username/RealTimeFaceRecognition.git
cd RealTimeFaceRecognition
```

### 2️⃣ Create Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate   # Windows
# or
source venv/bin/activate  # macOS/Linux
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 4️⃣ Run the Application
```bash
python app.py
```

Access the system on your browser:  
➡️ http://127.0.0.1:5000/

---

## 🧩 How It Works
1. **Register Face** — new users capture their faces using the webcam.  
2. **Extract Features** — CNN generates embeddings and stores them in CSV/SQLite.  
3. **Train/Update Model** — system retrains when new faces are registered.  
4. **Scan Attendance** — real-time recognition + liveness detection ensures security.  
5. **View Records** — attendance logs displayed through Flask dashboard.

---

## 📈 Model Details
- **Input Size:** 64×64 RGB  
- **Architecture:** Custom CNN with Residual Blocks  
- **Embedding Size:** 1024D vector  
- **Activation:** ReLU  
- **Optimizer:** Adam  
- **Loss Function:** Categorical Crossentropy  
- **Training Dataset:** LFW + custom dataset  
- **Performance:** ~97% accuracy on test set  

---

## 🔒 Liveness Detection Modules
| Module | Technique | Description |
|---------|------------|-------------|
| Blink Detection | Eye Aspect Ratio (EAR) | Detects blinking via Dlib landmarks |
| Smile Detection | Mouth Aspect Ratio (MAR) | Detects smile expression |
| Head Movement | SolvePnP (3D head pose) | Detects natural head motion |


---

## 📚 References
- MTCNN: Zhang et al., “Joint Face Detection and Alignment using Multi-task Cascaded CNNs.”  
- Dlib: King, D.E., “Dlib-ml: A Machine Learning Toolkit.”  
- OpenCV Documentation: https://docs.opencv.org/  
- TensorFlow Documentation: https://www.tensorflow.org/

---

## 🏁 Future Enhancements
- Replace rule-based liveness detection with **YOLO-based motion/liveness model**.  
- Deploy on **mobile/web cloud** for real-time remote access.  
- Integrate **student QR + face hybrid authentication** for higher reliability.  

---


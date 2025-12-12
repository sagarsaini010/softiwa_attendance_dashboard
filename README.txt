Softiwa Technologies — Face Recognition Attendance Dashboard

Instructions:
1. Create and activate Python 3.10 virtual environment:
   py -3.10 -m venv tfenv
   .\tfenv\Scripts\activate

2. Install dependencies:
   pip install -r requirements.txt

3. Run the Streamlit dashboard:
   streamlit run app.py

Workflow:
- Menu -> Add Student: enter Name and Roll, press Capture & Register (camera will save images).
- After registering students, encodings are updated automatically.
- Menu -> Take Attendance: start camera and press Enter to stop; attendance CSV saved to Attendance/ folder.
- Menu -> View Attendance: view saved CSV records.

CSV format: Roll,Name,Time

--------------------------------------------------------------------------------------------------------------------

# Smart Face Recognition Attendance System
A simple and efficient **Face Recognition-based Attendance System** built using:

- Streamlit (UI Dashboard)
- face_recognition (face detection & encoding)
- dlib (underlying face recognition library)
- OpenCV + PIL for image processing
- CSV-based attendance storage

This system allows you to:
- Register new students using the webcam
- Generate face encodings
- Capture attendance photos and automatically mark recognized faces
- View attendance records for all dates

---

## 🚀 Features

### ✔ Student Registration  
- Capture multiple photos per student  
- Automatically validates face before saving  
- Stores images in `dataset/`  

### ✔ Face Encoding  
- Extracts 128-dim face embeddings  
- Saves them to `encodings.pickle`  
- Used for matching during attendance  

### ✔ Attendance Capture  
- Capture live image via webcam  
- Detect & recognize multiple faces  
- Marks attendance once per student per day  
- Saves in `Attendance/YYYY-MM-DD.csv`  

### ✔ View Attendance Records  
- Load and display any past record  
- CSV-based easy portability  

---





import os
import io
import pickle
from PIL import Image
import numpy as np
import cv2
import face_recognition
import pandas as pd
from datetime import datetime

ENC_PATH = "encodings.pickle"
ATT_DIR = "Attendance"
os.makedirs(ATT_DIR, exist_ok=True)

def load_image_bytes(image_bytes):
    """Convert Streamlit camera bytes to RGB numpy array."""
    try:
        image = Image.open(io.BytesIO(image_bytes))
        if image.mode not in ("RGB", "L"):
            image = image.convert("RGB")

        arr = np.array(image)

        if arr is None or arr.ndim not in (2, 3):
            img_array = np.frombuffer(image_bytes, np.uint8)
            arr = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)

        # ⭐ CRITICAL FIX
        arr = np.ascontiguousarray(arr)

        return arr

    except:
        return None


def process_attendance_image(image_bytes, match_threshold=0.6):
    result = {"recognized": [], "unknown": [], "file": None}

    if not os.path.exists(ENC_PATH):
        print("Encodings not found.")
        return result

    with open(ENC_PATH, "rb") as f:
        data = pickle.load(f)

    known_encodings = data["encodings"]
    known_names     = data["names"]
    known_rolls     = data["rolls"]

    arr = load_image_bytes(image_bytes)
    if arr is None:
        print("Image load failed.")
        return result

    # Detect faces
    face_locations = face_recognition.face_locations(arr)
    face_encodings = face_recognition.face_encodings(arr, face_locations)

    today_file = os.path.join(ATT_DIR, f"{datetime.now().date()}.csv")
    old_df = pd.read_csv(today_file) if os.path.exists(today_file) else pd.DataFrame(columns=["Roll","Name","Time"])
    already = set(old_df["Roll"].astype(str).tolist())

    new_records = []

    for enc in face_encodings:
        distances = np.linalg.norm(np.array(known_encodings) - enc, axis=1)
        idx = distances.argmin()
        best_dist = distances[idx]

        if best_dist <= match_threshold:
            roll = str(known_rolls[idx])
            name = known_names[idx]
            time_str = datetime.now().strftime("%H:%M:%S")

            if roll not in already:
                new_records.append([roll, name, time_str])
        else:
            result["unknown"].append("Unknown")

    if new_records:
        df_new = pd.DataFrame(new_records, columns=["Roll","Name","Time"])
        df_full = pd.concat([old_df, df_new], ignore_index=True)
        df_full.drop_duplicates(subset="Roll", keep="first", inplace=True)
        df_full.to_csv(today_file, index=False)
        result["recognized"] = new_records
        result["file"] = today_file

    return result
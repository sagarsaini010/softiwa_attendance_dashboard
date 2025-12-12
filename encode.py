import os
import io
import pickle
from PIL import Image
import numpy as np
import cv2
import face_recognition
from datetime import datetime

DATASET_DIR = "dataset"
ENC_PATH = "encodings.pickle"
os.makedirs(DATASET_DIR, exist_ok=True)

def load_image_bytes(image_bytes):
    """Safely convert Streamlit camera bytes into RGB numpy array."""
    try:
        image = Image.open(io.BytesIO(image_bytes))

        # Convert any mode (RGBA, CMYK, etc.) → RGB
        if image.mode not in ("RGB", "L"):
            image = image.convert("RGB")

        arr = np.array(image)

        # If PIL decode fails, use OpenCV fallback
        if arr is None or arr.ndim not in (2, 3):
            img_array = np.frombuffer(image_bytes, np.uint8)
            arr = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)

        # ⭐ CRITICAL FIX → Make it C-contiguous
        arr = np.ascontiguousarray(arr)

        return arr

    except Exception as e:
        print("Image load error:", e)
        return None


def save_captured_image(name: str, roll: str, image_bytes: bytes):
    """Save only if face is detected."""
    arr = load_image_bytes(image_bytes)
    if arr is None:
        return None

    # Face detection (safe)
    faces = face_recognition.face_locations(arr)
    if not faces:
        print("No face detected.")
        return None

    # Save image
    safe_name = name.replace(" ", "_")
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    filename = f"{safe_name}__{roll}__{timestamp}.jpg"
    path = os.path.join(DATASET_DIR, filename)

    Image.fromarray(arr).save(path)
    return path


def update_encodings():
    """
    Generate ONE average encoding per student
    """
    images = [
        f for f in os.listdir(DATASET_DIR)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    student_encodings = {}  # roll -> list of encodings
    student_names = {}      # roll -> name

    for img in images:
        try:
            path = os.path.join(DATASET_DIR, img)
            image = face_recognition.load_image_file(path)
            encs = face_recognition.face_encodings(image)

            if not encs:
                continue

            enc = encs[0]

            # filename format: Name__Roll__timestamp.jpg
            parts = img.split("__")
            name = parts[0].replace("_", " ")
            roll = parts[1]

            if roll not in student_encodings:
                student_encodings[roll] = []

            student_encodings[roll].append(enc)
            student_names[roll] = name

        except Exception as e:
            print("Encoding error:", img, e)

    # --- CREATE AVERAGE ENCODINGS ---
    final_encodings = []
    final_names = []
    final_rolls = []

    for roll, enc_list in student_encodings.items():
        if len(enc_list) < 3:
            print(f"⚠ Not enough images for roll {roll}")
            continue

        avg_encoding = np.mean(enc_list, axis=0)

        final_encodings.append(avg_encoding)
        final_rolls.append(roll)
        final_names.append(student_names[roll])

    with open(ENC_PATH, "wb") as f:
        pickle.dump({
            "encodings": final_encodings,
            "names": final_names,
            "rolls": final_rolls,
        }, f)

    print(f"Saved {len(final_encodings)} average encodings")

    """Generate encodings from dataset folder."""
    images = [
        f for f in os.listdir(DATASET_DIR)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    known_encodings = []
    known_names = []
    known_rolls = []

    for img in images:
        path = os.path.join(DATASET_DIR, img)
        try:
            image = face_recognition.load_image_file(path)
            encs = face_recognition.face_encodings(image)

            if not encs:
                print("Encoding failed for:", img)
                continue

            enc = encs[0]

            parts = img.split("__")
            name = parts[0].replace("_", " ")
            roll = parts[1] if len(parts) > 1 else "NA"

            known_encodings.append(enc)
            known_names.append(name)
            known_rolls.append(roll)

        except Exception as e:
            print("Error encoding:", img, e)

    with open(ENC_PATH, "wb") as f:
        pickle.dump({
            "encodings": known_encodings,
            "names": known_names,
            "rolls": known_rolls,
        }, f)

    print(f"Saved {len(known_encodings)} encodings to encodings.pickle")
import os
import pandas as pd
import streamlit as st
from encode import save_captured_image, update_encodings
from attendance import process_attendance_image, ATT_DIR

st.set_page_config(page_title="Smart Attendance", layout="wide")
st.title("Smart Attendance System")

menu = ["Add Student", "Take Attendance", "View Records"]
choice = st.sidebar.selectbox("Menu", menu)

DATASET = "dataset"
os.makedirs(DATASET, exist_ok=True)
os.makedirs(ATT_DIR, exist_ok=True)

# ----------------------- ADD STUDENT -----------------------
if choice == "Add Student":
    st.header("Register New Student")

    name = st.text_input("Name")
    roll = st.text_input("Roll No")

    img = st.camera_input("Capture Image")

    if st.button("Save Image"):
        if img and name and roll:
            saved = save_captured_image(name, roll, img.getvalue())
            if saved:
                st.success(f"Image saved: {saved}")
            else:
                st.error("Face not detected / invalid image format.")
        else:
            st.error("Fill details and capture image.")

    if st.button("Update Encodings"):
        update_encodings()
        st.success("Encodings updated successfully!")

# ----------------------- TAKE ATTENDANCE -----------------------
elif choice == "Take Attendance":
    st.header("Take Attendance")
    img = st.camera_input("Capture Attendance Photo")

    if st.button("Process Attendance"):
        if img:
            result = process_attendance_image(img.getvalue())
            st.write(result)
        else:
            st.error("Please capture a photo.")

# ----------------------- VIEW RECORDS -----------------------
elif choice == "View Records":
    st.header("Attendance Records")

    files = os.listdir(ATT_DIR)

    if not files:
        st.info("No attendance files found.")
    else:
        file = st.selectbox("Select file", files)
        df = pd.read_csv(os.path.join(ATT_DIR, file))
        st.dataframe(df)

import cv2
import face_recognition
import os
import pandas as pd
from datetime import datetime
import pyttsx3
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import zipfile
import time

# === TTS Setup ===
engine = pyttsx3.init()
engine.setProperty('rate', 150)

def speak(text):
    print(f"[TTS] {text}")
    engine.say(text)
    engine.runAndWait()

# === Excel Setup ===
EXCEL_FILE = "attendance.xlsx"

def init_excel():
    if not os.path.exists(EXCEL_FILE):
        df = pd.DataFrame(columns=["Name", "Date", "Time"])
        df.to_excel(EXCEL_FILE, index=False, engine='openpyxl')
        print(f"[INFO] Created new Excel file: {EXCEL_FILE}")

init_excel()  # 🟢 Ensure the file is created before using

def mark_attendance_excel(name):
    now = datetime.now()
    date = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")

    try:
        df = pd.read_excel(EXCEL_FILE, engine='openpyxl')
    except zipfile.BadZipFile:
        print("[ERROR] attendance.xlsx is not a valid Excel file.")
        messagebox.showerror("File Error", "attendance.xlsx is corrupted or not a valid Excel file.")
        return
    except Exception as e:
        print(f"[ERROR] Could not read Excel file: {e}")
        return

    if not ((df["Name"] == name) & (df["Date"] == date)).any():
        new_entry = pd.DataFrame([[name, date, time_str]], columns=["Name", "Date", "Time"])
        df = pd.concat([df, new_entry], ignore_index=True)
        df.to_excel(EXCEL_FILE, index=False, engine='openpyxl')
        print(f"[INFO] Marked attendance for {name} at {time_str}")
        speak(f"Attendance marked for {name}")
    else:
        print(f"[INFO] {name} already marked today.")

# === Load known faces ===
known_faces_dir = 'known_faces'
known_encodings = []
known_names = []

for file in os.listdir(known_faces_dir):
    if file.lower().endswith(('.jpg', '.png')):
        path = os.path.join(known_faces_dir, file)
        image = face_recognition.load_image_file(path)
        encodings = face_recognition.face_encodings(image)
        if encodings:
            known_encodings.append(encodings[0])
            known_names.append(os.path.splitext(file)[0])
        else:
            print(f"[WARNING] No face found in {file}")

# === Create GUI Window ===
root = tk.Tk()
root.title("Face Attendance System")
root.geometry("600x500")

# === Canvas to display the webcam image ===
canvas = tk.Canvas(root, width=600, height=400)
canvas.pack()

# === Label to show recognized name ===
name_label = tk.Label(root, text="No student recognized", font=('Arial', 14))
name_label.pack(pady=10)

# === Button to open the attendance file ===
def open_attendance():
    try:
        df = pd.read_excel(EXCEL_FILE, engine='openpyxl')
        messagebox.showinfo("Attendance", df.to_string(index=False))
    except Exception as e:
        messagebox.showerror("Error", f"Could not open the attendance file:\n{str(e)}")

attendance_button = tk.Button(root, text="View Attendance", command=open_attendance, font=('Arial', 12))
attendance_button.pack(pady=20)

# === Capture and process the webcam feed ===
cap = cv2.VideoCapture(0)

# === Detection Delay Logic ===
detection_memory = {}  # {name: last_detected_timestamp}
DETECTION_DELAY = 2  # seconds

def update_frame():
    ret, frame = cap.read()
    recognized_name = "No face recognized"

    if ret:
        rgb_frame = frame[:, :, ::-1]
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        now = time.time()

        for face_encoding, face_location in zip(face_encodings, face_locations):
            matches = face_recognition.compare_faces(known_encodings, face_encoding)
            face_distances = face_recognition.face_distance(known_encodings, face_encoding)

            if matches and any(matches):
                best_match_index = face_distances.argmin()
                if matches[best_match_index]:
                    name = known_names[best_match_index]

                    # Draw face box
                    top, right, bottom, left = [v * 4 for v in face_location]
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                    cv2.putText(frame, name, (left + 6, bottom - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                    last_detected = detection_memory.get(name, 0)
                    if now - last_detected > DETECTION_DELAY:
                        mark_attendance_excel(name)
                        detection_memory[name] = now  # Update time
                        recognized_name = name
                    else:
                        recognized_name = f"{name} (waiting...)"

        # Convert frame for Tkinter
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_image = Image.fromarray(frame_rgb)
        frame_photo = ImageTk.PhotoImage(frame_image)

        canvas.create_image(0, 0, anchor=tk.NW, image=frame_photo)
        canvas.image = frame_photo

    name_label.config(text=recognized_name)
    root.after(10, update_frame)

# Start the webcam feed loop
update_frame()

# Start GUI event loop
root.mainloop()

# Cleanup
cap.release()
cv2.destroyAllWindows()

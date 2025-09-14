import tkinter as tk
import customtkinter as ck

import pandas as pd
import numpy as np
import pickle
import mediapipe as mp
import cv2
from PIL import Image, ImageTk

from landmarks import landmarks

ck.set_appearance_mode("dark")
ck.set_default_color_theme("blue")

window = ck.CTk()
window.title("Pose Classification")
window.state("zoomed")

SCREEN_WIDTH = window.winfo_screenwidth()
SCREEN_HEIGHT = window.winfo_screenheight()

CARD_WIDTH = 250
CARD_HEIGHT = 140
FONT_TITLE = ("Arial", 24, "bold")
FONT_VALUE = ("Arial", 32, "bold")

def create_info_card(master, title, color="#333333", number_color="lime"):
    frame = ck.CTkFrame(master, width=CARD_WIDTH, height=CARD_HEIGHT, corner_radius=25, fg_color="#1e1e1e")
    frame.pack_propagate(False)

    label = ck.CTkLabel(frame, text=title, font=FONT_TITLE, text_color="white")
    label.pack(pady=(15,5))

    value = ck.CTkLabel(frame, text="0", font=FONT_VALUE, text_color=number_color,
                        fg_color=color, width=140, height=60, corner_radius=15)
    value.pack(pady=(5,15))
    return frame, value

top_frame = ck.CTkFrame(window, fg_color="#111111", height=CARD_HEIGHT+40)
top_frame.pack(fill="x", pady=20)

cards_frame = ck.CTkFrame(top_frame, fg_color="#111111")
cards_frame.pack(expand=True)

class_frame, classBox = create_info_card(cards_frame, "STAGE", number_color="yellow")
counter_frame, counterBox = create_info_card(cards_frame, "REPS", number_color="lime")
prob_frame, probBox = create_info_card(cards_frame, "PROB", number_color="orange")
status_frame, statusBox = create_info_card(cards_frame, "STATUS", number_color="lime")
statusBox.configure(text="PAUSED", text_color="red") 

class_frame.pack(side="left", padx=40)
counter_frame.pack(side="left", padx=40)
prob_frame.pack(side="left", padx=40)
status_frame.pack(side="left", padx=40)

counter = 0
def reset_counter():
    global counter
    counter = 0
    counterBox.configure(text="0")

reset_btn = ck.CTkButton(
    window,
    text="RESET",
    command=reset_counter,
    height=60,
    width=240,
    font=("Arial", 22, "bold"),
    text_color="white",
    fg_color="#1F2936",
    hover_color="#264A78",
    corner_radius=25
)
reset_btn.pack(side="bottom", pady=30)

video_frame = ck.CTkFrame(window, width=800, height=600, fg_color="black", corner_radius=20)
video_frame.pack(pady=20)
lmain = tk.Label(video_frame, bg="black")
lmain.place(relx=0.5, rely=0.5, anchor="center")

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)

with open('deadlift.pkl', 'rb') as f:
    model = pickle.load(f)

cap = cv2.VideoCapture(0)
current_stage = ''
counter = 0
bodylang_prob = np.array([0, 0, 0])
bodylang_class = ''
prev_stage = None
counting_active = False
gesture_cooldown = 0

def is_open_palm_relaxed(hand_landmarks):
    tips = [mp_hands.HandLandmark.THUMB_TIP,
            mp_hands.HandLandmark.INDEX_FINGER_TIP,
            mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
            mp_hands.HandLandmark.RING_FINGER_TIP,
            mp_hands.HandLandmark.PINKY_TIP]
    pip_joints = [mp_hands.HandLandmark.THUMB_IP,
                  mp_hands.HandLandmark.INDEX_FINGER_PIP,
                  mp_hands.HandLandmark.MIDDLE_FINGER_PIP,
                  mp_hands.HandLandmark.RING_FINGER_PIP,
                  mp_hands.HandLandmark.PINKY_PIP]

    extended = sum(
        1 for tip, pip in zip(tips, pip_joints)
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[pip].y
    )
    if extended < 5:
        return False

    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
    index_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
    pinky_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP]

    p1 = np.array([wrist.x, wrist.y, wrist.z])
    p2 = np.array([index_mcp.x, index_mcp.y, index_mcp.z])
    p3 = np.array([pinky_mcp.x, pinky_mcp.y, pinky_mcp.z])

    v1 = p2 - p1
    v2 = p3 - p1
    normal = np.cross(v1, v2)

    if normal[2] > -0.05:  
        return False

    spread = abs(index_mcp.x - pinky_mcp.x)
    if spread < 0.08: 
        return False

    return True

def detect():
    global current_stage, counter, bodylang_class, bodylang_prob, prev_stage
    global counting_active, gesture_cooldown

    ret, frame_cam = cap.read()
    if not ret:
        lmain.after(10, detect)
        return

    frame_cam = cv2.flip(frame_cam, 1)
    image = cv2.cvtColor(frame_cam, cv2.COLOR_BGR2RGB)
    
    result = pose.process(image)
    if result.pose_landmarks:
        mp_drawing.draw_landmarks(
            image, result.pose_landmarks, mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=3),
            mp_drawing.DrawingSpec(color=(0,200,255), thickness=2, circle_radius=2)
        )
        try:
            row = np.array([[res.x, res.y, res.z, res.visibility] for res in result.pose_landmarks.landmark]).flatten()
            x = pd.DataFrame([row], columns=landmarks)
            bodylang_prob = model.predict_proba(x)[0]
            bodylang_class = model.predict(x)[0]

            if counting_active:
                if bodylang_class == 'down' and bodylang_prob.max() > 0.7:
                    current_stage = 'down'
                elif bodylang_class == 'up' and bodylang_prob.max() > 0.7:
                    if prev_stage == 'down':
                        counter += 1
                    current_stage = 'up'
                prev_stage = current_stage
        except Exception:
            pass

    hand_result = hands.process(image)
    if hand_result.multi_hand_landmarks and gesture_cooldown == 0:
        for hand_landmarks in hand_result.multi_hand_landmarks:
            if is_open_palm_relaxed(hand_landmarks):
                counting_active = not counting_active
                statusBox.configure(text="ACTIVE" if counting_active else "PAUSED",
                                    text_color="lime" if counting_active else "red")
                gesture_cooldown = 20 
    if gesture_cooldown > 0:
        gesture_cooldown -= 1

    frame_h, frame_w = 600, 800
    img_pil = Image.fromarray(image)
    img_pil.thumbnail((frame_w, frame_h))
    imgtk = ImageTk.PhotoImage(img_pil)
    lmain.imgtk = imgtk
    lmain.configure(image=imgtk)
    lmain.after(10, detect)

    counterBox.configure(text=str(counter))
    probBox.configure(text=f"{round(bodylang_prob.max()*100,2)}%")
    classBox.configure(text=current_stage.upper())

detect()
window.mainloop()

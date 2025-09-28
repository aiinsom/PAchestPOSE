import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from pose_checker import analyze_pose
from PIL import ImageFont, ImageDraw, Image

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

st.set_page_config(page_title="Chest PA Real-time Pose Checker")
st.title("📹 ตรวจจับท่า Chest PA ")

run = st.checkbox('✅ เริ่มเปิดกล้อง')

FRAME_WINDOW = st.image([])

def draw_thai_text(frame, feedback, font_path="fonts/THSarabun.ttf"):
    image_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image_pil)
    font = ImageFont.truetype(font_path, 32)
    y = 30
    for msg in feedback:
        draw.text((10, y), msg, font=font, fill=(0, 255, 0))
        y += 40
    return cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

if run:
    cap = cv2.VideoCapture(0)
    with mp_pose.Pose(min_detection_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.error("ไม่สามารถเปิดกล้องได้")
                break

            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)

            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                feedback = analyze_pose(results.pose_landmarks.landmark)
                frame = draw_thai_text(frame, feedback)

            FRAME_WINDOW.image(frame, channels="BGR")
    cap.release()

import streamlit as st
from ultralytics import YOLO
import cv2
import tempfile
import numpy as np

st.title("Nut Counting on Conveyor Belt")

model = YOLO("best.pt")

# YOLO class index mapping (from your dataset YAML)
class_names = ["bolt", "nut"]
target_class_index = 1  # nut

uploaded_video = st.file_uploader("Upload conveyor belt video", type=["mp4", "avi", "mov"])

LINE_POSITION = 400  # adjust if needed

if uploaded_video is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_video.read())

    cap = cv2.VideoCapture(tfile.name)
    stframe = st.empty()

    counted_positions = set()
    total_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=0.45)
        annotated = results[0].plot()

        detections = results[0].boxes.xyxy.cpu().numpy()
        classes = results[0].boxes.cls.cpu().numpy()

        # Draw the counting line
        cv2.line(annotated, (0, LINE_POSITION), (annotated.shape[1], LINE_POSITION), (255, 0, 0), 2)

        for i, box in enumerate(detections):
            cls_id = int(classes[i])

            # Count only nuts
            if cls_id != target_class_index:
                continue

            x1, y1, x2, y2 = box
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            # Check when nut crosses the line
            if LINE_POSITION - 6 <= cy <= LINE_POSITION + 6:
                if cx not in counted_positions:
                    counted_positions.add(cx)
                    total_count += 1

            cv2.putText(annotated, f"Count: {total_count}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        stframe.image(annotated, channels="BGR")

    cap.release()
    st.success(f"Final Nut Count: {total_count}")

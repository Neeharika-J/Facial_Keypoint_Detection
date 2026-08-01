
# ==========================================================
# IMPORT REQUIRED LIBRARIES
# ==========================================================
import os
import cv2
import av
import mlflow
import torch
import numpy as np
import onnxruntime as ort
import streamlit as st
from dotenv import load_dotenv
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

st.set_page_config(
    page_title="Facial Keypoint Detection",
    layout="wide",
)

#cache_resource is used to cache the resources like model, session,
#face_cascade so that they are not loaded multiple times which can slow down the application.
@st.cache_resource 
def load_resources():
    load_dotenv()

    
    # ONNX Inference session
    session = ort.InferenceSession(
        "model.onnx",
        providers=["CPUExecutionProvider"],
    )
    input_name = session.get_inputs()[0].name
    print('ONNX Inference session initialized successfully.')


    # Face detector
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades +
        "haarcascade_frontalface_default.xml"
    )
    if face_cascade.empty():
        raise RuntimeError("Could not load Haar Cascade.")
    print("Face detector initialized successfully.")
    

    return session, input_name, face_cascade


# ==========================================================
# LOAD RESOURCES
# ==========================================================
session, input_name, face_cascade = load_resources()

# ==========================================================
# VIDEO PROCESSOR
# ==========================================================
class VideoProcessor(VideoProcessorBase):

    def __init__(self):
        self.frame_count = 0
        self.last_faces = ()

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


        self.frame_count += 1
        # Detect faces only every 5th frame
        if self.frame_count % 10 == 0 or len(self.last_faces) == 0:
            self.last_faces = face_cascade.detectMultiScale(
                gray_frame,
                scaleFactor=1.1,
                minNeighbors=5
            )

        # Reuse previous detections
        faces = self.last_faces
        for (x, y, w, h) in faces:
            face = img[y:y+h, x:x+w]
            if face.size == 0:
                continue
            face = cv2.resize(face, (96, 96))
            face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            input_tensor = face_rgb.astype(np.float32) / 255.0
            input_tensor = np.transpose(input_tensor, (2, 0, 1))
            input_tensor = np.expand_dims(input_tensor, axis=0)
            output = session.run(
                None,
                {input_name: input_tensor},
            )[0]
            points = output.reshape(-1, 2)
            scale_x = w / 96
            scale_y = h / 96
            for px, py in points:
                px = int(px * scale_x + x)
                py = int(py * scale_y + y)
                cv2.circle(
                    img,
                    (px, py),
                    2,
                    (0, 255, 0),
                    -1,
                )

            cv2.rectangle(
                img,
                (x, y),
                (x + w, y + h),
                (255, 0, 0),
                2,
            )
        return av.VideoFrame.from_ndarray(
            img,
            format="bgr24",
        )


# ==========================================================
# UI
# ==========================================================
st.title("🎥 Real-Time Facial Keypoint Detection")
st.write(
    "Allow camera access to run real-time facial keypoint detection."
)
left, center, right = st.columns([2, 6, 2])
with center:
    webrtc_streamer(
        key="camera",
        video_processor_factory=VideoProcessor,
        media_stream_constraints={
            "video": {
                        "width": {"ideal": 640},
                        "height": {"ideal": 480},
                        "frameRate": {"ideal": 15},
                    },
            "audio": False,
        },
        async_processing=True,
    )

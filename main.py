# Import necessary libraries
import cv2
import numpy as np
import torch
import numpy as np
import onnxruntime as ort
import mlflow
import os
import onnx
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file


# Load model from MLflow
# ENTER YOUR MLflow TRACKING URI HERE OR DOWNLOAD MODEL FROM model.onnx FILE
mlflow.set_tracking_uri(os.environ['MLFLOW_TRACKING_URI'])
model_uri = os.environ["MODEL_URI"]  # models:/[model_name]/[model_version]
model = mlflow.pytorch.load_model(model_uri)
device = torch.device("cpu")
model = model.to(device)
model.eval()
print("Model loaded successfully from MLflow.")



# Export only once
if not os.path.exists("model.onnx"):
    dummy = torch.randn(1, 3, 96, 96) #dummy input for the model
    torch.onnx.export(
        model,
        dummy,
        "model.onnx",
        opset_version=17,
        dynamo=False,
    )
face = torch.randn(1, 3, 96, 96, dtype=torch.float32, device=device)
face = face.to(device)
torch.onnx.export(
    model,
    face,
    "model.onnx",
    opset_version=17,
    dynamo=False,
)
print("Model exported to ONNX format successfully.")



# ONNX Inference session
session = ort.InferenceSession(
    "model.onnx",
    providers=["CPUExecutionProvider"]
)
input_name = session.get_inputs()[0].name
print('ONNX Inference session initialized successfully.')



# Face detector
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
xml_path = os.path.join(BASE_DIR, "haarcascade_frontalface_default.xml")
print(xml_path)
face_cascade = cv2.CascadeClassifier(xml_path)
if face_cascade.empty():
    raise FileNotFoundError(
        f"Could not load cascade model. Make sure 'haarcascade_frontalface_default.xml' exists at: {xml_path}"
    )
print("Face detector initialized successfully.")



# Initialize local desktop webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()
print("Webcam started. Press 'q' on the video window to exit.")


# Detect faces and predict facial landmarks
try:
    while True:
        # Read frame directly from desktop camera
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break
        # ---------------- PREPROCESSING & INFERENCE ----------------
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray_frame,
            scaleFactor=1.1,
            minNeighbors=5
        )
        for (x, y, w, h) in faces:
            # Crop the face
            face = frame[y:y+h, x:x+w]
            # Resize to model expected size
            face = cv2.resize(face, (96, 96))
            # Preprocess for ONNX runtime
            face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            input_tensor = face_rgb.astype(np.float32) / 255.0
            input_tensor = np.transpose(input_tensor, (2, 0, 1))
            input_tensor = np.expand_dims(input_tensor, axis=0)
            # Inference
            output = session.run(None, {input_name: input_tensor})[0]
            points = output.reshape(-1, 2)
            # Scale landmarks back to bounding box dimensions
            scale_x = w / 96
            scale_y = h / 96
            # Draw the landmarks on the original frame
            for px, py in points:
                px = int(px * scale_x + x)
                py = int(py * scale_y + y)
                cv2.circle(frame, (px, py), 2, (0, 255, 0), -1)
            # Draw the face bounding box
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        # -----------------------------------------------------------

        # Render frame in a native desktop window
        cv2.imshow("Facial Landmark Detection", frame)
        # Press 'q' on your keyboard to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("\nInterrupted by user.")
finally:
    # Release hardware and close window
    cap.release()
    cv2.destroyAllWindows()
    print("Webcam released and window closed successfully.")



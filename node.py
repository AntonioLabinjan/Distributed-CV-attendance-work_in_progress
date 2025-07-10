import cv2
import numpy as np
import torch
import requests
import threading
import time
from queue import Queue
from transformers import CLIPProcessor, CLIPModel
from scipy.spatial.distance import cosine  
import mediapipe as mp
import os
import logging

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# ----- LOGGING SETUP -----
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler("node.log"),
        logging.StreamHandler()
    ]
)

# Configuration
SERVER_URL = "http://localhost:6010/classify"
NODE_ID = 0
DIFF_THRESHOLD = 0.2  # threshold for sending embedding (lower = more sensitive)

# Model and device
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
model.eval()

# Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# MediaPipe Face Mesh for segmentation
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,  # tracking preko frameova
    max_num_faces=1,
    refine_landmarks=True,    # ili False za dodatnu brzinu
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


# Queue and state
embedding_queue = Queue()
last_message = f"node {NODE_ID}: detected Unknown [--]"
last_message_lock = threading.Lock()
last_embedding = None  # for comparison with previous embedding


def segment_face(image_rgb):
    """Brža segmentacija lica koristeći MediaPipe Face Mesh s resizeom i trackingom."""

    # Resize za bolju brzinu (manja rezolucija)
    small_rgb = cv2.resize(image_rgb, (320, 240))

    # Obrada na manjoj slici
    results = face_mesh.process(small_rgb)
    if not results.multi_face_landmarks:
        logging.debug("No face mesh detected.")
        return None

    # Skaliranje nazad na originalnu veličinu
    h_orig, w_orig = image_rgb.shape[:2]
    h_small, w_small = small_rgb.shape[:2]
    scale_x = w_orig / w_small
    scale_y = h_orig / h_small

    # Kreiranje maske
    mask = np.zeros((h_orig, w_orig), dtype=np.uint8)
    landmarks = results.multi_face_landmarks[0].landmark
    points = [(int(lm.x * w_small * scale_x), int(lm.y * h_small * scale_y)) for lm in landmarks]

    # Konveksni poligon i segmentacija
    hull = cv2.convexHull(np.array(points))
    cv2.fillConvexPoly(mask, hull, 255)
    segmented_face = cv2.bitwise_and(image_rgb, image_rgb, mask=mask)

    return segmented_face



def classify_worker():
    global last_message
    while True:
        embedding = embedding_queue.get()
        if embedding is None:
            logging.info("Classify worker thread stopping.")
            break  # exit thread

        try:
            response = requests.post(SERVER_URL, json={"embedding": embedding.tolist(), "node_id": NODE_ID}, timeout=2)
            if response.status_code == 200:
                data = response.json()
                with last_message_lock:
                    last_message = data.get("message", last_message)
                logging.info(f"Server response: {last_message}")
            else:
                logging.warning(f"Error response from server: status code {response.status_code}")
        except Exception as e:
            logging.error(f"Error sending to server: {e}")


# Start worker thread
threading.Thread(target=classify_worker, daemon=True).start()

# Camera setup
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    ret, frame = cap.read()
    if not ret:
        logging.warning("Failed to grab frame from camera.")
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        face = frame[y:y + h, x:x + w]
        if face.size == 0:
            continue

        # Convert to RGB
        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

        # Segment the face to remove background inside face ROI
        segmented_face = segment_face(face_rgb)
        if segmented_face is None:
            continue  # no face mesh detected

        # Extract embedding from segmented face
        inputs = processor(images=segmented_face, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.get_image_features(**inputs)
        embedding = outputs.cpu().numpy().flatten()
        embedding /= np.linalg.norm(embedding)

        # Send only if embedding is different enough from last sent embedding
        if last_embedding is None or cosine(embedding, last_embedding) > DIFF_THRESHOLD:
            if embedding_queue.qsize() < 3:  # prevent overload
                embedding_queue.put(embedding)
                last_embedding = embedding
                logging.info(f"Embedding sent to server. Queue size: {embedding_queue.qsize()}")

        # Draw rectangle and message
        with last_message_lock:
            display_message = last_message
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, display_message, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow("Distributed CV Node", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        logging.info("Quitting program.")
        break

cap.release()
cv2.destroyAllWindows()
embedding_queue.put(None)  # stop worker thread
logging.info("Program terminated cleanly.")

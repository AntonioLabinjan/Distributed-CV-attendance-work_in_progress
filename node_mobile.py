import cv2
import numpy as np
import torch
import requests
import threading
import time
from queue import Queue
from transformers import CLIPProcessor, CLIPModel
import mediapipe as mp
import logging

# --- Logging setup ---
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler("node.log"),
        logging.StreamHandler()
    ]
)

# Konfiguracija
SERVER_URL = "http://localhost:6010/classify"
NODE_ID = 1
SEND_INTERVAL = 1.5  # sekundi između slanja

# Model i uređaj
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
model.eval()

# Haar Cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# MediaPipe Face Mesh for segmentation
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)

# Queue i stanje
embedding_queue = Queue()
last_message = f"node {NODE_ID}: detected Unknown [--]"
last_message_lock = threading.Lock()


def segment_face(image_rgb):
    """Segment face area using MediaPipe Face Mesh and return masked image."""
    results = face_mesh.process(image_rgb)
    if not results.multi_face_landmarks:
        logging.debug("No face mesh detected.")
        return None

    mask = np.zeros(image_rgb.shape[:2], dtype=np.uint8)
    h, w = mask.shape

    landmarks = results.multi_face_landmarks[0].landmark
    points = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]

    hull = cv2.convexHull(np.array(points))
    cv2.fillConvexPoly(mask, hull, 255)

    segmented_face = cv2.bitwise_and(image_rgb, image_rgb, mask=mask)
    return segmented_face


def classify_worker():
    global last_message
    last_sent_time = 0

    while True:
        embedding = embedding_queue.get()
        if embedding is None:
            logging.info("Classify worker thread stopping.")
            break  # izlazak iz threada

        now = time.time()
        if now - last_sent_time < SEND_INTERVAL:
            logging.debug("Skipping send: send interval not reached yet.")
            continue

        try:
            response = requests.post(SERVER_URL, json={"embedding": embedding.tolist(), "node_id": NODE_ID}, timeout=2)
            if response.status_code == 200:
                data = response.json()
                with last_message_lock:
                    last_message = data.get("message", last_message)
                logging.info(f"Server response: {last_message}")
            else:
                logging.warning(f"Greska u odgovoru sa servera: status code {response.status_code}")
        except Exception as e:
            logging.error(f"Greska pri slanju na server: {e}")

        last_sent_time = time.time()


# Pokreni worker thread
threading.Thread(target=classify_worker, daemon=True).start()

# Kamera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    ret, frame = cap.read()
    if not ret:
        logging.warning("Ne mogu dohvatiti frame s kamere.")
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        face = frame[y:y + h, x:x + w]
        if face.size == 0:
            continue

        # Convert to RGB for MediaPipe and CLIP
        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

        # Segment the face to remove background inside face ROI
        segmented_face = segment_face(face_rgb)
        if segmented_face is None:
            continue  # no face mesh detected, skip

        # Extract embedding from segmented face
        inputs = processor(images=segmented_face, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.get_image_features(**inputs)
        embedding = outputs.cpu().numpy().flatten()
        embedding /= np.linalg.norm(embedding)

        # Pošalji embedding u queue za obradu
        if embedding_queue.qsize() < 3:  # spriječi pretrpavanje
            embedding_queue.put(embedding)
            logging.info(f"Embedding poslan u queue. Velicina queuea: {embedding_queue.qsize()}")

        # Nacrtaj lice i ispiši poruku
        with last_message_lock:
            display_message = last_message
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, display_message, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow("Distributed CV Node", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        logging.info("Prekid programa (pritisnuta tipka 'q').")
        break

cap.release()
cv2.destroyAllWindows()
embedding_queue.put(None)  # zaustavi worker thread
logging.info("Program zavrsio normalno.")

# moja usb kamera; bitno: takat u port desno
import cv2
import numpy as np
import torch
import logging
import redis
import json
from transformers import CLIPProcessor, CLIPModel
from scipy.spatial.distance import cosine
import mediapipe as mp
import os
import threading
from datetime import datetime, timedelta

# === Env setup ===
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# === Logging setup ===
logging.basicConfig(
    level=logging.DEBUG,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler("node.log"),
        logging.StreamHandler()
    ]
)

# === Redis setup ===
try:
    redis_client = redis.Redis(host='localhost', port=6379, db=0)
    redis_client.ping()
    logging.info("Redis povezan uspjesno.")
except redis.ConnectionError as e:
    logging.error(f"Ne mogu se spojiti na Redis: {e}")
    exit(1)

# === Config ===
NODE_ID = 1
THRESHOLD_DISTANCE = 0.2
THRESHOLD_TIME = timedelta(seconds=30)

# === State for classification ===
last_embedding_per_node = {}
last_timestamp_per_node = {}

# === CLIP model ===
device = "cuda" if torch.cuda.is_available() else "cpu"
logging.info(f"Koristi se device: {device}")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
model.eval()

# === Face detection ===
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# === Face mesh ===
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# === UI state ===
last_message = f"node {NODE_ID}: detection successful"
last_message_lock = threading.Lock()

# === Segmentacija lica ===
def segment_face(image_rgb):
    small_rgb = cv2.resize(image_rgb, (320, 240))
    results = face_mesh.process(small_rgb)
    if not results.multi_face_landmarks:
        logging.debug("Nema face landmarks detected.")
        return None
    h_orig, w_orig = image_rgb.shape[:2]
    h_small, w_small = small_rgb.shape[:2]
    scale_x = w_orig / w_small
    scale_y = h_orig / h_small
    mask = np.zeros((h_orig, w_orig), dtype=np.uint8)
    landmarks = results.multi_face_landmarks[0].landmark
    points = [(int(lm.x * w_small * scale_x), int(lm.y * h_small * scale_y)) for lm in landmarks]
    hull = cv2.convexHull(np.array(points))
    cv2.fillConvexPoly(mask, hull, 255)
    segmented_face = cv2.bitwise_and(image_rgb, image_rgb, mask=mask)
    return segmented_face

# === Should Classify funkcija ===
def should_classify(node_id, new_embedding):
    current_time = datetime.now()

    if node_id not in last_embedding_per_node:
        last_embedding_per_node[node_id] = new_embedding
        last_timestamp_per_node[node_id] = current_time
        return True

    prev_embedding = last_embedding_per_node[node_id]
    dist = cosine(new_embedding, prev_embedding)

    if dist > THRESHOLD_DISTANCE:
        last_embedding_per_node[node_id] = new_embedding
        last_timestamp_per_node[node_id] = current_time
        return True

    prev_time = last_timestamp_per_node[node_id]
    if current_time - prev_time > THRESHOLD_TIME:
        last_timestamp_per_node[node_id] = current_time
        return True

    return False

TOO_DARK_THRESHOLD = 20  # average brightness (0-255)
TOO_DARK_CONSEC_FRAMES = 10
too_dark_counter = 0

def is_too_dark(gray_frame):
    global too_dark_counter
    avg_brightness = np.mean(gray_frame)
    if avg_brightness < TOO_DARK_THRESHOLD:
        too_dark_counter += 1
    else:
        too_dark_counter = 0
    return too_dark_counter >= TOO_DARK_CONSEC_FRAMES


# === Kamera setup ===
cap = cv2.VideoCapture(1)


# === Health check kamere ===
health_check_ret, health_check_frame = cap.read()
if not health_check_ret or health_check_frame is None or health_check_frame.size == 0:
    logging.critical("HEALTH CHECK: Kamera nije uspješno dohvatila inicijalni frame. Node se gasi.")
    cap.release()
    exit(1)
else:
    avg_brightness = np.mean(cv2.cvtColor(health_check_frame, cv2.COLOR_BGR2GRAY))
    if avg_brightness < TOO_DARK_THRESHOLD:
        logging.warning(f"HEALTH CHECK: Inicijalni frame je pretaman (avg_brightness={avg_brightness:.2f}).")
    else:
        logging.info("HEALTH CHECK: Kamera uspješno prošla inicijalni test.")

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    logging.error("Kamera se nije mogla otvoriti.")
    exit(1)

logging.info("Node pokrenut. Spreman za obradu...")

while True:
    ret, frame = cap.read()
    if not ret:
        logging.warning("Nisam uspio dohvatiti frame.")
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if is_too_dark(gray):
        logging.warning("It's too dark — kamera ne vidi gotovo ništa.")
        cv2.putText(frame, "Too damn dark!", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    if len(faces) == 0:
        logging.debug("Nema lica u frameu.")

    for (x, y, w, h) in faces:
        face = frame[y:y + h, x:x + w]
        if face.size == 0:
            logging.debug("Izdvojeno lice ima size 0.")
            continue

        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        segmented_face = segment_face(face_rgb)
        if segmented_face is None:
            logging.debug("Segmentacija lica nije uspjela.")
            continue

        inputs = processor(images=segmented_face, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.get_image_features(**inputs)
        embedding = outputs.cpu().numpy().flatten()
        embedding /= np.linalg.norm(embedding)

        logging.debug(f"Embedding shape: {embedding.shape}, first 5: {embedding[:5]}")
        logging.debug(f"Embedding norm: {np.linalg.norm(embedding)}")

        if should_classify(NODE_ID, embedding):
            data = {
                "embedding": embedding.tolist(),
                "node_id": NODE_ID,
                "retries": 0
            }
            try:
                json_data = json.dumps(data)
                redis_client.lpush("embedding_queue", json_data)
                queue_size = redis_client.llen("embedding_queue")
                logging.info(f"Embedding poslan u Redis queue. Queue size: {queue_size}")
                logging.debug(f"JSON podatak: {json_data[:200]}...")
            except Exception as e:
                logging.error(f"Greska pri slanju u Redis: {e}")

        # === UI prikaz ===
        with last_message_lock:
            display_message = last_message
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, display_message, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow("Distributed CV Node", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        logging.info("Prekid programa.")
        break

cap.release()
cv2.destroyAllWindows()
logging.info("Node clean shutdown.")

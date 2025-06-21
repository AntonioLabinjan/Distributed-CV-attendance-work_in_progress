# node.py
import cv2
import numpy as np
import torch
import requests
import threading
import time
from queue import Queue
from transformers import CLIPProcessor, CLIPModel
from scipy.spatial.distance import cosine  



# Konfiguracija
SERVER_URL = "http://localhost:6010/classify"
NODE_ID = 0
DIFF_THRESHOLD = 0.2  # prag za slanje (ča manji = osjetljivije)

# Model i uređaj
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
model.eval()

# Haar Cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Queue i stanje
embedding_queue = Queue()
last_message = f"node {NODE_ID}: detected Unknown [--]"
last_message_lock = threading.Lock()
last_embedding = None  # za usporedbu s prethodnim

def classify_worker():
    global last_message
    while True:
        embedding = embedding_queue.get()
        if embedding is None:
            break  # izlazak iz threada

        try:
            response = requests.post(SERVER_URL, json={"embedding": embedding.tolist(), "node_id": NODE_ID}, timeout=2)
            if response.status_code == 200:
                data = response.json()
                with last_message_lock:
                    last_message = data.get("message", last_message)
            else:
                print("Greška u odgovoru sa servera.")
        except Exception as e:
            print("Greška pri slanju na server:", e)

# Pokreni worker thread
threading.Thread(target=classify_worker, daemon=True).start()

# Kamera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        face = frame[y:y + h, x:x + w]
        if face.size == 0:
            continue

        # Ekstrakcija embeddinga
        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        inputs = processor(images=face_rgb, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.get_image_features(**inputs)
        embedding = outputs.cpu().numpy().flatten()
        embedding /= np.linalg.norm(embedding)

        # Pametno slanje samo ako je različito od prošlog
        if last_embedding is None or cosine(embedding, last_embedding) > DIFF_THRESHOLD:
            if embedding_queue.qsize() < 3:  # spriječi pretrpavanje
                embedding_queue.put(embedding)
                last_embedding = embedding  # ažuriraj

        # Nacrtaj lice i ispiši poruku
        with last_message_lock:
            display_message = last_message
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, display_message, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow("Distributed CV Node", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
embedding_queue.put(None)  # zaustavi worker thread

# server.py
# to run this: docker run -p 6010:6010 face-rec-central_server_6010
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

from flask import Flask, request, jsonify, Response, redirect, url_for
import torch
import numpy as np
import faiss
from transformers import CLIPProcessor, CLIPModel
import cv2
from datetime import datetime

app = Flask(__name__)

# Init model
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)

# Embeddings database
known_face_encodings = []
known_face_names = []
faiss_index = None

# function to add faces and extract embeddings
def add_known_face(image_path, name):
    image = cv2.imread(image_path)
    if image is None:
        print(f"[UPOZORENJE] Ne mogu učitati: {image_path}")
        return
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    inputs = processor(images=image_rgb, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.get_image_features(**inputs)
    embedding = outputs.cpu().numpy().flatten()
    normalized = embedding / np.linalg.norm(embedding)
    known_face_encodings.append(normalized)
    known_face_names.append(name)

# function to load dataset embeddings
def load_dataset():
    count = 0
    for person in os.listdir(dataset_path := "dataset"):
        person_path = os.path.join(dataset_path, person)
        if not os.path.isdir(person_path):
            continue

        for subdir in os.listdir(person_path):
            subdir_path = os.path.join(person_path, subdir)
            if not os.path.isdir(subdir_path):
                continue

            for img_file in os.listdir(subdir_path):
                if not img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue
                img_path = os.path.join(subdir_path, img_file)
                if not os.path.isfile(img_path):
                    continue
                add_known_face(img_path, person)
                count += 1
    print(f"[INFO] Ucitano {count} poznatih lica.")

# build FAISS index using face embeddings
def build_index():
    global faiss_index
    encodings_np = np.array(known_face_encodings).astype('float32')
    faiss_index = faiss.IndexFlatL2(encodings_np.shape[1])
    faiss_index.add(encodings_np)
    print("[INFO] FAISS indeks izgraden.")

# function to classify faces using threshold and voting
def classify_face(face_embedding, k=5, threshold=0.6):
    D, I = faiss_index.search(np.array([face_embedding]).astype('float32'), k)
    votes = {}
    for idx, dist in zip(I[0], D[0]):
        if dist > threshold:
            continue
        name = known_face_names[idx]
        votes[name] = votes.get(name, 0) + 1
    if votes:
        winner = max(votes, key=votes.get)
        return winner, votes[winner]
    return "Unknown", 0


# list to log detections
detection_log = []

# classification route
@app.route("/classify", methods=["POST"])
def classify_api():
    data = request.get_json()
    embedding = np.array(data["embedding"])
    embedding /= np.linalg.norm(embedding)
    name, votes = classify_face(embedding)
    
    node_id = data.get("node_id", 0)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    log_entry = {
        "timestamp": timestamp,
        "name": name,
        "votes": votes,
        "node_id": node_id
    }
    detection_log.append(log_entry)

    message = f"node {node_id}: detected {name} [{timestamp}, votes {votes}]"
    return jsonify({"message": message})

# show logs in JSON format
@app.route("/log", methods=["GET"])
def view_log():
    return jsonify(detection_log)

# show logs in html
@app.route("/log/html")
def view_log_html():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Detekcija lica - Log</title>
        <meta charset="utf-8">
        <style>
            body { font-family: sans-serif; padding: 20px; }
            table { border-collapse: collapse; width: 100%; }
            th, td { border: 1px solid #ccc; padding: 8px; text-align: left; }
            th { background-color: #f2f2f2; }
        </style>
    </head>
    <body>
        <h1>Log Detekcija</h1>
        <table id="logTable">
            <thead>
                <tr>
                    <th>Timestamp</th>
                    <th>Ime</th>
                    <th>Glasovi</th>
                    <th>Node ID</th>
                </tr>
            </thead>
            <tbody></tbody>
        </table>

        <script>
            async function fetchLog() {
                const res = await fetch("/log");
                const data = await res.json();
                const tbody = document.querySelector("#logTable tbody");
                tbody.innerHTML = "";

                data.slice().forEach(entry => {
                    const row = document.createElement("tr");
                    row.innerHTML = `
                        <td>${entry.timestamp}</td>
                        <td>${entry.name}</td>
                        <td>${entry.votes}</td>
                        <td>${entry.node_id}</td>
                    `;
                    tbody.appendChild(row);
                });
            }

            setInterval(fetchLog, 1000); // osvježi svakih 1 sekundu
            fetchLog(); // odmah prvi put
        </script>
    </body>
    </html>
    """


@app.route("/")
def home():
    return redirect(url_for("view_log_html"))


if __name__ == "__main__":
    load_dataset()
    build_index()
    app.run(host="0.0.0.0", port=6010)


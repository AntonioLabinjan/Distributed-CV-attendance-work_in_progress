# server.py
# to pull this: docker pull antoniolabinjan/face-rec-central_server:latest
# to run this: docker run -p 6010:6010 face-rec-central_server_6010
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

from flask import Flask, request, jsonify, redirect, url_for
import torch
import numpy as np
import faiss
from transformers import CLIPProcessor, CLIPModel
import cv2
from datetime import datetime
from queue import Queue
import threading

app = Flask(__name__)

# Inicijalizacija modela
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
model.eval()

# Baza embeddinga
known_face_encodings = []
known_face_names = []
faiss_index = None

# Log i queue
detection_log = []
embedding_queue = Queue()

# Threshold testiranje
thresholds_to_test = thresholds = [
    0.20, 0.25, 0.30, 0.35, 0.40, 0.42, 0.45, 0.47, 0.50, 0.52, 0.55, 0.57, 0.60, 0.65, 0.70
]
threshold_stats = {th: [] for th in thresholds_to_test}

# === Dataset i FAISS ===
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

def load_dataset():
    count = 0
    dataset_path = "dataset"
    for person in os.listdir(dataset_path):
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
                if os.path.isfile(img_path):
                    add_known_face(img_path, person)
                    count += 1
    print(f"[INFO] Učitano {count} poznatih lica.")

def build_index():
    global faiss_index
    encodings_np = np.array(known_face_encodings).astype('float32')
    faiss_index = faiss.IndexFlatL2(encodings_np.shape[1])
    faiss_index.add(encodings_np)
    print("[INFO] FAISS indeks izgrađen.")

# === Klasifikacija ===
def classify_face(face_embedding, k=7, threshold=0.45):
    D, I = faiss_index.search(np.array([face_embedding]).astype('float32'), k)
    votes = {}
    for idx, dist in zip(I[0], D[0]):
        if dist > threshold:
            continue
        name = known_face_names[idx]
        weight = 1.0 / (dist + 1e-6)
        votes[name] = votes.get(name, 0) + weight
    if votes:
        winner = max(votes, key=votes.get)
        return winner, round(votes[winner], 2)
    return "Unknown", 0

# === Worker koji obrađuje queue ===
def classify_worker():
    while True:
        item = embedding_queue.get()
        if item is None:
            break  # shutdown
        embedding, node_id = item
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        best_result = ("Unknown", 0, 0.0)  # name, votes, threshold

        for th in thresholds_to_test:
            name, score = classify_face(embedding, k=7, threshold=th)
            threshold_stats[th].append((name, score))
            if name != "Unknown" and score > best_result[1]:
                best_result = (name, score, th)

        log_entry = {
            "timestamp": timestamp,
            "name": best_result[0],
            "votes": best_result[1],
            "node_id": node_id,
            "used_threshold": best_result[2]
        }
        detection_log.append(log_entry)

worker_thread = threading.Thread(target=classify_worker, daemon=True)
worker_thread.start()

# === Flask routes ===
@app.route("/classify", methods=["POST"])
def classify_api():
    data = request.get_json()
    embedding = np.array(data["embedding"])
    embedding /= np.linalg.norm(embedding)
    node_id = data.get("node_id", 0)

    embedding_queue.put((embedding, node_id))
    return jsonify({"message": f"node {node_id}: embedding received"})

@app.route("/log", methods=["GET"])
def view_log():
    return jsonify(detection_log)

@app.route("/log/html")
def view_log_html():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Detekcija lica - Log</title>
        <meta charset=\"utf-8\">
        <style>
            body { font-family: sans-serif; padding: 20px; }
            table { border-collapse: collapse; width: 100%; }
            th, td { border: 1px solid #ccc; padding: 8px; text-align: left; }
            th { background-color: #f2f2f2; }
        </style>
    </head>
    <body>
        <h1>Log Detekcija</h1>
        <table id=\"logTable\">
            <thead>
                <tr>
                    <th>Timestamp</th>
                    <th>Ime</th>
                    <th>Score</th>
                    <th>Node ID</th>
                    <th>Threshold</th>
                </tr>
            </thead>
            <tbody></tbody>
        </table>

        <script>
            function getScoreColor(score) {
                if (score >= 6) return '#b3e5fc';
                if (score >= 4) return '#c8e6c9';
                if (score >= 2) return '#fff9c4';
                return '#ffcdd2';
            }

            async function fetchLog() {
                const res = await fetch("/log");
                const data = await res.json();
                const tbody = document.querySelector("#logTable tbody");
                tbody.innerHTML = "";
                data.forEach(entry => {
                    const row = document.createElement("tr");
                    const scoreColor = getScoreColor(entry.votes);
                    row.innerHTML = `
                        <td>${entry.timestamp}</td>
                        <td>${entry.name}</td>
                        <td style=\"background-color: ${scoreColor}; font-weight: bold;\">${entry.votes.toFixed(2)}</td>
                        <td>${entry.node_id}</td>
                        <td>${entry.used_threshold.toFixed(2)}</td>
                    `;
                    tbody.appendChild(row);
                });
            }

            setInterval(fetchLog, 1000);
            fetchLog();
        </script>
    </body>
    </html>
    """

@app.route("/")
def home():
    return redirect(url_for("view_log_html"))

from flask import render_template_string

@app.route("/threshold_stats")
def threshold_stats_view():
    summary = {
        str(th): len([res for res in results if res[0] != "Unknown"])
        for th, results in threshold_stats.items()
    }

    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Threshold Analiza</title>
        <meta charset="utf-8">
        <style>
            body { font-family: sans-serif; padding: 20px; }
            table { border-collapse: collapse; width: 50%; margin: auto; }
            th, td { border: 1px solid #ccc; padding: 10px; text-align: center; }
            th { background-color: #f2f2f2; }
            h1 { text-align: center; }
        </style>
    </head>
    <body>
        <h1>Rezultati po Thresholdu</h1>
        <table>
            <thead>
                <tr>
                    <th>Threshold</th>
                    <th>Broj pogodaka</th>
                </tr>
            </thead>
            <tbody>
                {% for threshold, count in summary.items() %}
                <tr>
                    <td>{{ threshold }}</td>
                    <td>{{ count }}</td>
                </tr>
                {% endfor %}
            </tbody>
        </table>
    </body>
    </html>
    """
    return render_template_string(html_template, summary=summary)

if __name__ == "__main__":
    known_face_encodings.clear()
    known_face_names.clear()

    print("[INIT] Učitavanje poznatih lica...")
    load_dataset()

    print("[INIT] Gradnja FAISS indeksa...")
    build_index()

    print("[INIT] Server pokrenut na portu 6010.")
    app.run(host="0.0.0.0", port=6010)

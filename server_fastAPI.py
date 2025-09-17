
# fastapi imports
from fastapi import FastAPI, HTTPException, status, Request, Header
from fastapi.responses import JSONResponse, HTMLResponse, RedirectResponse, PlainTextResponse
from pydantic import BaseModel
import uvicorn
from jinja2 import Template

# db imports
import redis

# faceRec system imports
import faiss
from transformers import CLIPProcessor, CLIPModel
import cv2
import numpy as np
import torch
from scipy.spatial.distance import cosine


# other python imports
import logging
from datetime import datetime, timedelta
import json
import threading
from collections import Counter
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"



# Logging konfiguracija
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

app = FastAPI()

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

import os
import redis

redis_host = os.getenv("REDIS_HOST", "localhost")
redis_port = int(os.getenv("REDIS_PORT", 6380)) # kad radimo s local serveron, port je 6379, ali kad pokušavamo gađat na server koji se pokrene kroz docker compose, port je 6382

redis_client = redis.Redis(host=redis_host, port=redis_port, db=0)



def send_global_log(name: str, node_id: str, event: str):
    try:
        redis_client.xadd("global_logs", {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "name": name,
            "node_id": node_id,
            "event": event
        })
    except Exception as e:
        print(f"[GlobalLog] Failed to send log: {e}")


# Threshold testiranje
thresholds_to_test = [
    0.20, 0.25, 0.30, 0.35, 0.40, 0.42, 0.45, 0.47, 0.50, 0.52, 0.55, 0.57, 0.60, 0.65, 0.70
]
threshold_stats = {th: [] for th in thresholds_to_test}


def load_all_tokens(folder="credentials"):
    tokens = {}
    for filename in os.listdir(folder):
        if filename.startswith("node_") and filename.endswith("_token.json"):
            with open(os.path.join(folder, filename)) as f:
                data = json.load(f)
                tokens[str(data["node_id"])] = {
                    "token": data["token"],
                    "timezone": data.get("timezone", "UTC")
                }
    return tokens


VALID_TOKENS = load_all_tokens()

class TokenRequest(BaseModel):
    token: str

class TokenResponse(BaseModel):
    valid: bool
    node_id: str | None = None
    error: str | None = None



@app.get("/redis-test")
def redis_test():
    try:
        # Postavi neki ključ i vrijednost u Redis
        redis_client.set('test_key', 'Redis works!')

        # Uzmi vrijednost nazad
        value = redis_client.get('test_key')
        if value:
            value = value.decode('utf-8')
        else:
            value = 'No value returned'

        return JSONResponse(content={"status": "success", "value": value})
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})

# === Dataset i FAISS ===
def add_known_face(image_path, name):
    image = cv2.imread(image_path)
    if image is None:
        logging.warning(f"Unable to load: {image_path}")
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
    known_faces = set()
    dataset_path = "dataset"

    for person in os.listdir(dataset_path):
        person_path = os.path.join(dataset_path, person)
        if not os.path.isdir(person_path):
            continue

        train_path = os.path.join(person_path, "train_segm")
        if not os.path.isdir(train_path):
            continue

        for img_file in os.listdir(train_path):
            if not img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue

            img_path = os.path.join(train_path, img_file)
            if os.path.isfile(img_path):
                add_known_face(img_path, person)
                known_faces.add(person)
                count += 1

    logging.info(f"Loaded {count} known faces.")
    logging.info(f"Known face class-names: {', '.join(sorted(known_faces))}")

# === FAISS indeks ===
def build_index():
    global faiss_index
    encodings_np = np.array(known_face_encodings).astype('float32')
    faiss_index = faiss.IndexFlatL2(encodings_np.shape[1])
    faiss_index.add(encodings_np)
    logging.info("FAISS index built successfully.")


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

unknown_attempts = []
intruder_alerts = []

def check_for_intruder_alert(timestamp_str, node_id, window_seconds=30, threshold_attempts=3):
    global unknown_attempts, intruder_alerts

    try:
        ts = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
    except:
        return

    # Dodaj trenutni pokušaj
    unknown_attempts.append({"timestamp": ts, "node_id": node_id})

    # Makni stare pokušaje
    cutoff = datetime.now() - timedelta(seconds=window_seconds)
    unknown_attempts[:] = [a for a in unknown_attempts if a["timestamp"] >= cutoff]

    if len(unknown_attempts) >= threshold_attempts:
        alert_entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "count": len(unknown_attempts),
            "nodes": list(set(a["node_id"] for a in unknown_attempts))
        }
        intruder_alerts.append(alert_entry)
        logging.warning(f"[ALERT] Unknown intruder! {alert_entry}")
        unknown_attempts.clear()  # reset za novi prozor

# === Worker koji obrađuje queue ===

MAX_RETRIES = 3

def classify_worker():
    while True:
        try:
            result = redis_client.brpop("embedding_queue", timeout=2)
            if result is None:
                continue  # nema embeddinga trenutno

            _, message = result
            data = json.loads(message)
            embedding = np.array(data["embedding"])
            node_id = data["node_id"]
            retries = data.get("retries", 0)

            # === NOVO: should_classify cutoff ===
            if not should_classify(node_id, embedding):
                logging.info(f" Skipping classification for node {node_id} (too similar/recent).")
                continue

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

            if best_result[0] == "Unknown":
                check_for_intruder_alert(timestamp, node_id)

        except (json.JSONDecodeError, KeyError) as e:
            try:
                redis_client.lpush("embedding_dead", message)
                logging.warning(f" Invalid JSON or KeyError: {e} | Message: {message}")
            except Exception as inner_e:
                logging.error(f"Error while storing invalid message in dead-letter: {inner_e}")

        except Exception as e:
            try:
                retries = data.get("retries", 0) if 'data' in locals() else 0
            except Exception:
                retries = 0

            if retries < MAX_RETRIES:
                try:
                    if 'data' in locals():
                        data["retries"] = retries + 1
                        redis_client.lpush("embedding_queue", json.dumps(data))
                        logging.warning(f" Retry #{retries + 1} | Node {data.get('node_id')} | Error: {e}")
                    else:
                        logging.warning(f" Retry failed - no data variable | Error: {e}")
                except Exception as e2:
                    logging.error(f" Retry failed. Error: {e2}")
            else:
                try:
                    if 'data' in locals():
                        redis_client.lpush("embedding_dead", json.dumps(data))
                        logging.error(f" Too many retries, sending into dead-letter | Message: {data} | Error: {e}")
                    else:
                        logging.error(f" Dead-letter fallback failed - no data variable | Error: {e}")
                except Exception as e2:
                    logging.error(f" Dead-letter fallback failed. Error: {e2}")





# === Pydantic model za validaciju ulaznih podataka ===
class EmbeddingRequest(BaseModel):
    embedding: list[float]
    node_id: int = 0



# === Globalni trackeri po nodeu ===
last_embedding_per_node = {}      # npr. {0: np.array([...])}
last_timestamp_per_node = {}      # npr. {0: datetime object}

# === Parametri koji se lako fino štimaju ===
THRESHOLD_DISTANCE = 0.2          # Ako je embedding značajno različit → klasificiraj
THRESHOLD_TIME = timedelta(seconds=30)  # Ako je prošlo više vremena → klasificiraj opet

def should_classify(node_id, new_embedding):
    current_time = datetime.now()

    # === 1. Prvi put za ovaj node? ===
    if node_id not in last_embedding_per_node:
        last_embedding_per_node[node_id] = new_embedding
        last_timestamp_per_node[node_id] = current_time
        return True

    # === 2. Udaljenost između novog i zadnjeg embeddinga ===
    prev_embedding = last_embedding_per_node[node_id]
    dist = cosine(new_embedding, prev_embedding)

    if dist > THRESHOLD_DISTANCE:
        last_embedding_per_node[node_id] = new_embedding
        last_timestamp_per_node[node_id] = current_time
        return True

    # === 3. Ako je sličan embedding, ali prošlo je puno vremena ===
    prev_time = last_timestamp_per_node[node_id]
    if current_time - prev_time > THRESHOLD_TIME:
        last_timestamp_per_node[node_id] = current_time
        return True

    # === 4. Inače: preskoči klasifikaciju ===
    return False

@app.get("/log")
def view_log():
    return JSONResponse(content=detection_log)

@app.get("/log/html", response_class=HTMLResponse)
def view_log_html():
    return """
    <!DOCTYPE html>
<html>
<head>
    <title>Face detection - Log</title>
    <meta charset="utf-8">
    <style>
        /* Dark‑mode paleta */
        :root {
            --bg-main: #121212;
            --bg-card: #1e1e1e;
            --bg-header: #1f1f1f;
            --text-main: #e0e0e0;
            --border: #333;
        }

        body {
            font-family: sans-serif;
            padding: 20px;
            background: var(--bg-main);
            color: var(--text-main);
        }

        h1 { margin-bottom: 1rem; }

        table {
            border-collapse: collapse;
            width: 100%;
            background: var(--bg-card);
        }

        th, td {
            border: 1px solid var(--border);
            padding: 8px;
            text-align: left;
        }

        th {
            background: var(--bg-header);
            color: #fff;
        }

        /* Malo zaobljenja za moderniji look (opcionalno) */
        table, th:first-child, td:first-child { border-left-width: 2px; }
        table, th:last-child,  td:last-child  { border-right-width: 2px; }
        th:first-child, td:first-child { border-left-width: 2px; }
        th:last-child,  td:last-child  { border-right-width: 2px; }
    </style>
</head>
<body>
    <h1>Detection log</h1>
    <table id="logTable">
        <thead>
            <tr>
                <th>Timestamp</th>
                <th>Name</th>
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
                    <td style="background-color: ${scoreColor}; font-weight: bold;">${entry.votes.toFixed(2)}</td>
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



@app.get("/")
def home():
    return RedirectResponse(url="/log/html")


@app.get("/queue_contents")
def get_queue_contents():
    try:
        # Uzmi max 20 elemenata iz queuea (list length može biti velik)
        items = redis_client.lrange("embedding_queue", 0, 20)
        # Decode bytes u string i parse JSON
        decoded = []
        for item in items:
            try:
                decoded.append(item.decode('utf-8'))
            except Exception as e:
                decoded.append(f"<decode error: {e}>")

        queue_length = redis_client.llen("embedding_queue")
        return JSONResponse(content={"queue_length": queue_length, "items": decoded})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/threshold_stats", response_class=HTMLResponse)
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
        :root {
            --bg-main: #121212;
            --bg-card: #1e1e1e;
            --bg-header: #1f1f1f;
            --text-main: #e0e0e0;
            --border: #333;
        }

        body {
            font-family: sans-serif;
            padding: 20px;
            background: var(--bg-main);
            color: var(--text-main);
        }

        h1 {
            text-align: center;
            margin-bottom: 2rem;
        }

        table {
            border-collapse: collapse;
            width: 50%;
            margin: auto;
            background: var(--bg-card);
            color: var(--text-main);
        }

        th, td {
            border: 1px solid var(--border);
            padding: 10px;
            text-align: center;
        }

        th {
            background-color: var(--bg-header);
            color: #fff;
        }
    </style>
</head>
<body>
    <h1>Results by threshold</h1>
    <table>
        <thead>
            <tr>
                <th>Threshold</th>
                <th>Num of correct classifications</th>
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

    template = Template(html_template)
    rendered_html = template.render(summary=summary)
    return HTMLResponse(content=rendered_html)

'''
@app.get("/ping")
def ping():
    return {"message": "FASTAPI Server is up and running"}
'''
@app.get("/ping")
def ping():
    try:
        # Primjer provjere ako želiš pingati neki servis (dummy logika)
        # npr. if not redis_connection.ping(): raise Exception("Redis not responding")

        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={"message": "FASTAPI Server is up and running", "status": "healthy"}
        )
    except ConnectionError:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"message": "Dependent service unavailable", "status": "degraded"}
        )
    except Exception as e:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"message": f"Unexpected error: {str(e)}", "status": "error"}
        )


detection_log = []  # pretpostavljam da ovo već puniš drugdje

@app.get("/active_nodes/html", response_class=HTMLResponse)
def active_nodes_html():
    active_threshold_seconds = 60
    now = datetime.now()

    recent_nodes = {}
    node_counts = Counter()

    for entry in reversed(detection_log):
        try:
            entry_time = datetime.strptime(entry["timestamp"], "%Y-%m-%d %H:%M:%S")
        except:
            continue

        if now - entry_time <= timedelta(seconds=active_threshold_seconds):
            node_id = entry["node_id"]
            node_counts[node_id] += 1
            if node_id not in recent_nodes:
                recent_nodes[node_id] = entry_time
        else:
            break

    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>Aktivni Nodesi</title>
        <style>
            :root {
                --bg-main: #121212;
                --bg-card: #1e1e1e;
                --bg-header: #1f1f1f;
                --text-main: #e0e0e0;
                --border: #333;
                --active-color: #388e3c33;
            }
            body {
                font-family: Arial, sans-serif;
                padding: 20px;
                background-color: var(--bg-main);
                color: var(--text-main);
            }
            h1 {
                text-align: center;
                margin-bottom: 2rem;
            }
            table {
                border-collapse: collapse;
                width: 60%;
                margin: auto;
                background-color: var(--bg-card);
                color: var(--text-main);
            }
            th, td {
                border: 1px solid var(--border);
                padding: 10px;
                text-align: center;
            }
            th {
                background-color: var(--bg-header);
                color: #fff;
            }
            .active {
                background-color: var(--active-color);
            }
        </style>
    </head>
    <body>
        <h1>Active Nodes (Last {{threshold}} seconds)</h1>
        <table>
            <thead>
                <tr>
                    <th>Node ID</th>
                    <th>Last seen</th>
                    <th>Detection Count</th>
                </tr>
            </thead>
            <tbody>
            {% if recent_nodes %}
                {% for node_id, last_seen in recent_nodes.items() %}
                <tr class="active">
                    <td>{{ node_id }}</td>
                    <td>{{ last_seen.strftime("%Y-%m-%d %H:%M:%S") }}</td>
                    <td>{{ node_counts[node_id] }}</td>
                </tr>
                {% endfor %}
            {% else %}
                <tr>
                    <td colspan="3" style="color: grey;">No active nodes in last {{threshold}} seconds.</td>
                </tr>
            {% endif %}
            </tbody>
        </table>
        <p style="text-align:center; margin-top: 20px;">
            Total num of active nodes: <b>{{ recent_nodes|length }}</b>
        </p>
    </body>
    </html>
    """

    template = Template(html)
    rendered_html = template.render(
        recent_nodes=recent_nodes,
        node_counts=node_counts,
        threshold=active_threshold_seconds
    )
    return HTMLResponse(content=rendered_html)


@app.get("/intruder_alerts")
def get_intruder_alerts():
    return JSONResponse(content=intruder_alerts)


@app.get("/intruder_alerts/html", response_class=HTMLResponse)
def intruder_alerts_html():
    html = """
    <!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta http-equiv="refresh" content="5">
    <title>Unknown Intruder Alerts</title>
    <style>
        :root {
            --bg-main: #121212;
            --bg-card: #1e1e1e;
            --bg-header: #1f1f1f;
            --text-main: #e0e0e0;
            --border: #333;
            --alert-color: #b00020; /* tamna crvena za alert */
            --alert-bg: #ff8a8033;  /* suptilna tamnija crvena s transparencijom */
            --no-alert-color: #aaa;
        }

        body {
            font-family: Arial, sans-serif;
            padding: 20px;
            background-color: var(--bg-main);
            color: var(--text-main);
        }

        h1 {
            text-align: center;
            margin-bottom: 2rem;
        }

        table {
            border-collapse: collapse;
            width: 70%;
            margin: auto;
            background-color: var(--bg-card);
            color: var(--text-main);
        }

        th, td {
            border: 1px solid var(--border);
            padding: 10px;
            text-align: center;
        }

        th {
            background-color: var(--bg-header);
            color: #fff;
        }

        .alert {
            background-color: var(--alert-bg);
            color: var(--alert-color);
            font-weight: bold;
        }

        td[colspan="3"] {
            color: var(--no-alert-color);
        }
    </style>
</head>
<body>
    <h1>Intruder Alert Log</h1>
    <table>
        <thead>
            <tr>
                <th>Timestamp</th>
                <th>Num of tries</th>
                <th>Nodes</th>
            </tr>
        </thead>
        <tbody>
            {% for alert in alerts %}
            <tr class="alert">
                <td>{{ alert.timestamp }}</td>
                <td>{{ alert.count }}</td>
                <td>{{ alert.nodes | join(", ") }}</td>
            </tr>
            {% endfor %}
            {% if alerts|length == 0 %}
            <tr>
                <td colspan="3">No intruder alerts noted</td>
            </tr>
            {% endif %}
        </tbody>
    </table>
</body>
</html>
    """
    template = Template(html)
    rendered_html = template.render(alerts=intruder_alerts)
    return HTMLResponse(content=rendered_html)



@app.get("/reload_dataset", response_class=PlainTextResponse)
def reload_dataset():
    global faiss_index, known_face_encodings, known_face_names

    known_face_encodings.clear()
    known_face_names.clear()
    detection_log.clear()

    logging.info("Reloading dataset...")
    load_dataset()

    logging.info("Rebuilding FAISS index...")
    build_index()

    return "Dataset and index reloaded!"


@app.on_event("startup")
def startup_event():
    global known_face_encodings, known_face_names

    known_face_encodings.clear()
    known_face_names.clear()

    logging.info("Starting up... loading dataset.")
    load_dataset()
    build_index()

    # Start Redis classify worker in a background thread
    worker_thread = threading.Thread(target=classify_worker, daemon=True)
    worker_thread.start()
    logging.info("Started Redis classify worker thread.")

if __name__ == "__main__":
    port = 8000
    logging.info(f"Server running on port: {port}")
    uvicorn.run("server_FASTAPI:app", host="0.0.0.0", port=8000, reload=True)

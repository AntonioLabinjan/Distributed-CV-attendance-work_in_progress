# Distributed Face Recognition System (CLIP + FAISS)

Autorizacija zahtjeva: Token šalješ u HTTP headeru pod nazivom x-node-token (npr. x-node-token: tvoj_token_ovdje), a ne u tijelu (body) zahtjeva. => za TC testing

Provjera tokena: Server uspoređuje taj header token s validnim tokenima učitanim iz JSON fajlova i samo ako je token prepoznat dopušta daljnju obradu.

This repository contains a scalable system for distributed face recognition using multiple client nodes and a central server. Each node extracts face embeddings using a CLIP model and sends them to the server, which handles classification and logging using a FAISS index of known faces.

---

## Table of Contents

* [Overview](#overview)
* [System Architecture](#system-architecture)
* [Core Components](#core-components)
* [Technology Stack](#technology-stack)
* [Setup and Usage](#setup-and-usage)

  * [1. Running the Central Server](#1-running-the-central-server)
  * [2. Running a Node](#2-running-a-node)
* [Data Flow](#data-flow)
* [Design Principles](#design-principles)
* [Privacy and Security](#privacy-and-security)
* [Logging Interface](#logging-interface)
* [Example Outputs](#example-outputs)
* [Future Improvements](#future-improvements)

---

## Overview

This project enables face recognition in a distributed setting where low-cost client nodes perform local embedding extraction and offload classification to a centralized server. The system is designed to be:

* **Lightweight on edge devices**
* **Scalable across multiple clients**
* **Centralized for efficient indexing and classification**
* **Privacy-aware**, avoiding raw image transfer
* **Face segmentation:** Nodes apply MediaPipe Face Mesh segmentation to isolate the face region within the detected bounding box before embedding extraction, improving embedding quality by removing background noise.
* **Dynamic threshold tuning** via live grid search and auto-reloading of new faces without needing server restarts.


---

## System Architecture

The system is composed of the following key components:

* **Nodes (Clients):** Capture face images using webcams and extract face embeddings using the CLIP model.
* **Embedding Queue:** Located on the server; acts as middleware between the nodes and the classification service, ensuring fair and stable request handling.
* **Central Server:** Hosts the face embedding database, performs nearest neighbor search via FAISS, classifies embeddings, and logs recognition results.

### Architecture Diagram

> Replace the image below with your architecture diagram

```
[Insert your architecture diagram here, e.g., docs/images/architecture.png]
```

---

## Core Components

### Nodes

Each node is responsible for:

* Capturing webcam input in real-time
* Detecting faces using Haar cascades
* Segmenting detected faces with MediaPipe Face Mesh to mask out background pixels inside the face region
* Generating normalized CLIP embeddings
* Avoiding redundant transmissions by comparing embeddings with the previous one and sending only above certain threshold
* Sending new embeddings to the server for classification

### Embedding Queue (Server-side)

* Ensures **balanced and sequential processing** of requests from all nodes
* Acts as a **load balancer and buffer**, protecting the server from bursts of requests
* Located on the server and managed using Python’s `Queue` module and a background thread

### Central Server

* Loads all known faces from a structured dataset at startup
* Builds a **FAISS index** from the embeddings
* Classifies incoming embeddings via **k-nearest neighbor search**
* Logs results with timestamps, node ID, predicted name, and confidence score
* Provides a live **HTML interface** for monitoring detections
* Automatically reloads all known faces and rebuilds the FAISS index on startup, ensuring the system always uses the latest dataset structure without requiring manual refresh.

### Live Threshold Tuner (Server-side)

* Evaluates multiple threshold values in real-time for each incoming embedding
* Stores the number of successful classifications per threshold
* Results accessible via `/threshold_stats` endpoint
* Helps identify the best-performing threshold dynamically, no manual tuning required


---

## Technology Stack

| Component        | Technology                            |
| ---------------- | ------------------------------------- |
| Embedding Model  | CLIP (`openai/clip-vit-base-patch32`) |
| Classifier       | FAISS (L2 similarity search)          |
| Web Interface    | FastAPI + JavaScript                    |
| Face Detection   | OpenCV Haar Cascade                   |
| Communication    | HTTP (JSON over REST API)             |
| Parallelism      | Python threading + Queue module       |
| Containerization | Docker (for server deployment)        |

---

## Setup and Usage

### 1. Running the Central Server

You can run the central server directly or via Docker.

**Option A: Docker (Recommended)**

```bash
docker pull antoniolabinjan/face-rec-central_server:latest
docker run -p 6010:6010 -v $(pwd)/dataset:/app/dataset antoniolabinjan/face-rec-central_server:latest
```

> Ensure the dataset is available locally in the structure:
> `dataset/<person_name>/<subdir>/<images>.jpg`

**Option B: Manual (Python)**

```bash
pip install -r requirements.txt
python server.py
```

### 2. Running a Node

```bash
python node.py
```

Each node will:

* Start webcam capture
* Detect faces
* Generate embeddings using CLIP
* Send embeddings only when significantly different from the last one

---

## Data Flow

1. Each node detects a face → extracts an embedding.
2. Segments the face region using MediaPipe Face Mesh to remove background pixels within the face bounding box
3. Embedding is sent via POST to `/classify` on the server.
4. Server enqueues the request for processing.
5. A background thread processes embeddings sequentially:

   * Classification using FAISS
   * Result stored in `detection_log`
6. Server responds with formatted message (e.g., `node 2: detected John Doe`).

---

## Design Principles

* **Separation of concerns:** Nodes only extract embeddings; classification is centralized.
* **Fairness:** Queue ensures balanced handling across all nodes.
* **Efficiency:** No image transfer, just vector data.
* **Responsiveness:** Live updates via `/log/html`.
 * **Adaptability:** Real-time evaluation of multiple classification thresholds allows system to adapt to changing lighting, background, or model drift.


---

## Privacy and Security

* No images or video frames are transmitted over the network.
* Only normalized face embeddings (numeric vectors) are sent.
* All classification and identity resolution happens server-side.
* This design supports **GDPR-compliant** face recognition deployments.

---

## Logging Interface

Live classification results can be accessed at:

```
http://<server_ip>:6010/log/html
```

Live threshold grid results (HTML view):

```
http://<server_ip>:6010/threshold_stats/html
```

Output includes timestamp, predicted name, score, and node ID. Rows are color-coded based on confidence.

---

## Example Outputs

> Add a screenshot of the `/log/html` interface here.

```
[Insert live log screenshot here: docs/images/live_log.png]
```

Threshold statistics available at:

http://<server_ip>:6010/threshold_stats


Provides JSON summary of successful recognitions per threshold (auto-updated live).

---

## Future Improvements

* WebSocket-based real-time communication
* Embedded SQLite/DB backend for persistent logging
* Centralized monitoring dashboard
* Add authentication per node
* Auto-retraining of face database from new captures

---


DOCS DRAFT:
- samo ću si izlistat najvažnije rute i funkcije
- pa ću ben složit kako treba
Server
- add_known_face
- load_dataset
- build_index
- classify_face
- check_for_intruder_alert
- classify_worker
- /classify
- /log
- /log/html
- /
- /threshold_stats
- /ping
- /active_nodes/html
- /intruder_alerts
- /reload_dataset

Node
- segment_face
- classify_worker

## Maintainer

Antonio Labinjan
[GitHub: AntonioLabinjan](https://github.com/AntonioLabinjan)

Dockerhub deployment: https://hub.docker.com/repository/docker/antoniolabinjan/face-rec-central_server/general - > deprecated

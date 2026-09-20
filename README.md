# Multimodal Image & Text Embedding Server

Lightweight embedding microservices powered by **OpenAI CLIP (`clip-vit-base-patch32`)**. Designed to run smoothly on low-cost VPS instances (1 vCPU, 1GB–4GB RAM) with **in-memory SHA-256 deduplication** and **automatic idle RAM cleanup**.

---

## Architecture & Project Structure

```
server/
├── python/                      # Python (FastAPI + PyTorch)
│   ├── main.py
│   ├── requirements.txt
│   ├── Dockerfile
│   ├── .env.example
│   └── test_server.py
├── js/                          # JavaScript (Node.js + ONNX Runtime) ⭐ Fastest & Lowest RAM
│   ├── server.js
│   ├── package.json
│   ├── Dockerfile
│   └── .env.example
├── docker-compose.yml           # Unified Docker Compose
├── integration.md               # API Integration Guide for Developers
└── README.md
```

---

## Comparison Summary

| Metric | Python Server (`/python`) | Node.js Server (`/js`) ⭐ |
| :--- | :--- | :--- |
| **Framework** | FastAPI + PyTorch | Express + ONNX Runtime |
| **Active RAM** | ~600 – 800 MB | **~400 MB (INT8 Quantized)** |
| **Idle RAM (after 30 min)** | **~50 MB (Auto-unloaded)** | **~68 MB (Auto-unloaded)** |
| **Image Inference Latency** | ~120 – 370 ms | **~41.9 ms (24 img/sec)** |
| **Text Query Latency** | ~32 – 150 ms | **~2.0 ms (496 q/sec)** |
| **Default Port** | `8000` | `8001` |

---

## Quick Start with Docker

### Run Node.js Server (Recommended for 1 vCPU / 4GB VPS):
```bash
docker compose up -d --build embed-server-js
```
Endpoint: `http://localhost:8001/embed`

### Run Python Server:
```bash
docker compose up -d --build embed-server-python
```
Endpoint: `http://localhost:8000/embed`

### Run Both:
```bash
docker compose up -d --build
```

---

## Running Locally

### 1. Python Server:
```bash
cd python
python -m venv venv
# Linux / macOS: source venv/bin/activate | Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
python main.py
```

### 2. Node.js Server:
```bash
cd js
npm install
cp .env.example .env
npm start
```

---

## Integration Guide
For client integration code examples (cURL, JavaScript, Python, PostgreSQL/pgvector), see [`integration.md`](./integration.md).

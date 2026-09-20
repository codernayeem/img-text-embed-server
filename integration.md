# Embedding Server Integration Guide

A lightweight microservice that converts **Images**, **Text**, or **Image + Text (Fused)** into **512-dimensional vector arrays** using OpenAI's CLIP model.

---

## 1. Connection & Auth

* **Base URL:** `http://<SERVER_IP>:8000`
* **Auth Header:** `X-API-Key: <YOUR_API_KEY>` (or `?api_key=<KEY>`)
* **Vector Dimension:** `512` (Array of float32 values)

---

## 2. API Endpoints

### `POST /embed` (Unified Embedding Endpoint)
Accepts `multipart/form-data` or `application/x-www-form-urlencoded`.

| Parameter | Type | Required | Description |
| :--- | :--- | :--- | :--- |
| `file` | Binary File | Optional | Image file (`.jpg`, `.png`, `.webp`) |
| `text` | String | Optional | Text query / product name / description |
| `image_weight` | Float | Optional (Default: `0.6`) | Weight of image in fused vector |
| `text_weight` | Float | Optional (Default: `0.4`) | Weight of text in fused vector |

> **Note:** You must provide at least `file`, `text`, or both.

#### Response Format:
```json
{
  "success": true,
  "type": "image" | "text" | "image+text",
  "sha256": "8daf25a6ad1fa4b7...",
  "dim": 512,
  "cached": false,
  "latency_ms": 118.4,
  "embedding": [-0.0858, -0.0172, 0.0580, "... 512 numbers ..."]
}
```

---

### `POST /embed/text` (JSON Convenience Endpoint)
**Headers:** `Content-Type: application/json`

```json
{
  "text": "baby shampoo and gentle wash"
}
```

---

### `GET /health` (System Status)
No auth required. Returns uptime, RAM, CPU usage, and request counts.

---

## 3. Usage Examples

### A. cURL

```bash
# 1. Image only (Visual Search / Uploads)
curl -X POST "http://<SERVER_IP>:8000/embed" \
  -H "X-API-Key: YOUR_API_KEY" \
  -F "file=@product.jpg"

# 2. Text only (Semantic Search Query)
curl -X POST "http://<SERVER_IP>:8000/embed" \
  -H "X-API-Key: YOUR_API_KEY" \
  -F "text=herbal green tea"

# 3. Image + Text (Fused Product Vector)
curl -X POST "http://<SERVER_IP>:8000/embed" \
  -H "X-API-Key: YOUR_API_KEY" \
  -F "file=@product.jpg" \
  -F "text=Cetaphil Baby Massage Oil"
```

---

### B. JavaScript / TypeScript (Fetch / Node.js)

```javascript
async function getEmbedding({ file, text }) {
  const formData = new FormData();
  if (file) formData.append("file", file);
  if (text) formData.append("text", text);

  const res = await fetch("http://<SERVER_IP>:8000/embed", {
    method: "POST",
    headers: { "X-API-Key": "YOUR_API_KEY" },
    body: formData,
  });

  const data = await res.json();
  return data.embedding; // 512-dim float array
}
```

---

### C. Python

```python
import requests

def get_embedding(image_path=None, text=None):
    headers = {"X-API-Key": "YOUR_API_KEY"}
    files = {"file": open(image_path, "rb")} if image_path else None
    data = {"text": text} if text else None

    res = requests.post("http://<SERVER_IP>:8000/embed", headers=headers, files=files, data=data)
    return res.json()["embedding"]
```

---

## 4. Database Query Quick Reference

Because all embeddings share the same 512-dimensional coordinate space, vector search is a standard **Cosine Similarity** lookup:

### PostgreSQL (`pgvector`)
```sql
-- Search Top 10 closest items to query vector
SELECT id, name, 1 - (embedding <=> '[0.034, -0.129, ...]'::vector) AS similarity
FROM items
ORDER BY embedding <=> '[0.034, -0.129, ...]'::vector ASC
LIMIT 10;
```

### Qdrant / Pinecone / Milvus / Chroma
* **Dimension:** `512`
* **Distance Metric:** `Cosine`

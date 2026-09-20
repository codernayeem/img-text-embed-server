# Image & Text Embedding Server (CLIP Microservice)

A production-ready, lightweight embedding microservice powered by **OpenAI CLIP (`clip-vit-base-patch32`)** and **FastAPI**. Designed to run smoothly on low-cost VPS instances (1 vCPU, 4GB RAM) with **~450MB RAM footprint** and **~120ms latency**.

---

## Features

- ⚡ **Unified Embedding API (`POST /embed`)**: Supports:
  - **Image only** (multipart file upload).
  - **Text only** (natural language search query).
  - **Image + Text** (combined/fused vector with customizable weights, e.g. 0.6 image + 0.4 text).
- 🔒 **Security & Authentication**:
  - `API_KEY` header (`X-API-Key`) required for all embedding endpoints.
  - `ADMIN_KEY` header (`X-Admin-Key`) required for administrative endpoints.
- 🚀 **In-Memory History & Deduplication Cache**:
  - Keeps the last **100 embedding requests** in memory with Client IP, Timestamp, SHA-256 hash, and vectors.
  - **Deduplication:** Repeated requests with the exact same image/text return instantly (**0.0 ms**) without re-running CPU inference.
- 🩺 **Health & Monitoring (`GET /health`)**:
  - Real-time memory usage (RSS MB), CPU %, uptime, total requests, and cache hit metrics.
- 🐳 **Docker & Docker Compose Ready**.

---

## Quick Start

### 1. Installation

```bash
# Clone or navigate to server directory
cd server

# Create and activate virtual environment
python -m venv venv
# Linux / macOS:
source venv/bin/activate
# Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment

Copy `.env.example` to `.env` and set your secret keys:

```bash
cp .env.example .env
```

```ini
API_KEY=my_secure_api_key_12345
ADMIN_KEY=my_secure_admin_key_67890
HOST=0.0.0.0
PORT=8000
MODEL_NAME=openai/clip-vit-base-patch32
DEVICE=cpu
IMAGE_WEIGHT=0.6
TEXT_WEIGHT=0.4
HISTORY_LIMIT=100
```

### 3. Run the Server

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

Interactive Swagger API docs available at: `http://localhost:8000/docs`

---

## API Reference

### 1. Unified Embed Endpoint: `POST /embed`

**Headers:**
`X-API-Key: my_secure_api_key_12345`

#### Case A: Image Only
```bash
curl -X POST "http://localhost:8000/embed" \
  -H "X-API-Key: my_secure_api_key_12345" \
  -F "file=@/path/to/product.jpg"
```

#### Case B: Text Only (Search Query)
```bash
curl -X POST "http://localhost:8000/embed" \
  -H "X-API-Key: my_secure_api_key_12345" \
  -F "text=baby shampoo gentle wash"
```

#### Case C: Image + Text (Fused Vector)
```bash
curl -X POST "http://localhost:8000/embed" \
  -H "X-API-Key: my_secure_api_key_12345" \
  -F "file=@/path/to/product.jpg" \
  -F "text=Cetaphil Baby Massage Oil" \
  -F "image_weight=0.6" \
  -F "text_weight=0.4"
```

#### Response:
```json
{
  "success": true,
  "type": "image+text",
  "sha256": "8daf25a6ad1fa4b7...",
  "dim": 512,
  "cached": false,
  "latency_ms": 83.06,
  "embedding": [-0.0858, -0.0172, 0.0580, ...]
}
```

---

### 2. Health & Status: `GET /health`

```bash
curl -X GET "http://localhost:8000/health"
```

```json
{
  "status": "healthy",
  "model": "openai/clip-vit-base-patch32",
  "device": "cpu",
  "uptime_seconds": 124.5,
  "memory_rss_mb": 465.7,
  "cpu_usage_percent": 8.1,
  "total_requests": 14,
  "cache_hits": 3,
  "in_memory_history_count": 11
}
```

---

### 3. Admin In-Memory History: `GET /admin/history`

**Headers:**
`X-Admin-Key: my_secure_admin_key_67890`

```bash
# Get compact log without full vector payloads
curl -X GET "http://localhost:8000/admin/history?include_vectors=false" \
  -H "X-Admin-Key: my_secure_admin_key_67890"
```

```json
[
  {
    "id": 3,
    "ip": "192.168.1.100",
    "timestamp": "2026-09-20 14:43:28",
    "type": "image+text",
    "sha256": "8daf25a6ad1f...",
    "text": "Premium Baby Care Set",
    "dim": 512
  }
]
```

---

## Next.js (TypeScript) Integration Example

```typescript
// app/api/search/route.ts
import { NextRequest, NextResponse } from "next/server";
import { prisma } from "@/lib/prisma";

export async function POST(req: NextRequest) {
  const formData = await req.formData();
  const file = formData.get("file") as File | null;
  const text = formData.get("text") as string | null;

  // 1. Forward request to Python Embedding Server
  const embedFormData = new FormData();
  if (file) embedFormData.append("file", file);
  if (text) embedFormData.append("text", text);

  const embedRes = await fetch("http://YOUR_VPS_IP:8000/embed", {
    method: "POST",
    headers: {
      "X-API-Key": process.env.EMBED_API_KEY!,
    },
    body: embedFormData,
  });

  const { embedding } = await embedRes.json();
  const vectorString = `[${embedding.join(",")}]`;

  // 2. Query Postgres with pgvector
  const products = await prisma.$queryRaw`
    SELECT id, name, price, "imageUrl",
           1 - (embedding <=> ${vectorString}::vector) AS similarity
    FROM "Product"
    ORDER BY embedding <=> ${vectorString}::vector
    LIMIT 10;
  `;

  return NextResponse.json({ products });
}
```

---

## Docker Deployment on VPS

```bash
# Build & start container in background
docker compose up -d --build

# View logs
docker compose logs -f
```

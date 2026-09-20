import express from 'express';
import multer from 'multer';
import cors from 'cors';
import crypto from 'crypto';
import dotenv from 'dotenv';
import { RawImage, AutoProcessor, AutoTokenizer, CLIPVisionModelWithProjection, CLIPTextModelWithProjection } from '@xenova/transformers';

dotenv.config();

const app = express();
app.use(cors());
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

const PORT = parseInt(process.env.PORT || '8001');
const API_KEY = process.env.API_KEY || 'my_secure_api_key_12345';
const ADMIN_KEY = process.env.ADMIN_KEY || 'my_secure_admin_key_67890';
const MODEL_NAME = process.env.MODEL_NAME || 'Xenova/clip-vit-base-patch32';
const DEFAULT_IMAGE_WEIGHT = parseFloat(process.env.IMAGE_WEIGHT || '0.6');
const DEFAULT_TEXT_WEIGHT = parseFloat(process.env.TEXT_WEIGHT || '0.4');
const HISTORY_LIMIT = parseInt(process.env.HISTORY_LIMIT || '100');
const IDLE_TIMEOUT_MS = parseInt(process.env.IDLE_TIMEOUT_MINUTES || '30') * 60 * 1000;

const upload = multer({ storage: multer.memoryStorage() });

// Server state
const startTime = Date.now();
let lastActivityTime = Date.now();
let totalRequests = 0;
let cacheHits = 0;

// In-Memory history & cache
const history = [];
const cacheMap = new Map();
let historyId = 0;

// Model instances (Loaded lazily, auto-unloaded on idle)
let visionModel = null;
let textModel = null;
let processor = null;
let tokenizer = null;
let loadingPromise = null;

async function getOrLoadModel() {
  lastActivityTime = Date.now();
  if (visionModel && textModel && processor && tokenizer) {
    return { visionModel, textModel, processor, tokenizer };
  }

  if (loadingPromise) return loadingPromise;

  loadingPromise = (async () => {
    console.log(`[${new Date().toISOString()}] Loading ONNX CLIP model '${MODEL_NAME}' into RAM...`);
    const t0 = Date.now();
    processor = await AutoProcessor.from_pretrained(MODEL_NAME);
    tokenizer = await AutoTokenizer.from_pretrained(MODEL_NAME);
    visionModel = await CLIPVisionModelWithProjection.from_pretrained(MODEL_NAME, { quantized: true });
    textModel = await CLIPTextModelWithProjection.from_pretrained(MODEL_NAME, { quantized: true });
    console.log(`[${new Date().toISOString()}] ONNX Model loaded in ${(Date.now() - t0)/1000}s.`);
    loadingPromise = null;
    return { visionModel, textModel, processor, tokenizer };
  })();

  return loadingPromise;
}

function unloadModelIfIdle() {
  if ((visionModel || textModel) && (Date.now() - lastActivityTime) >= IDLE_TIMEOUT_MS) {
    console.log(`[${new Date().toISOString()}] Idle timeout reached. Unloading ONNX models from RAM...`);
    visionModel = null;
    textModel = null;
    processor = null;
    tokenizer = null;
    if (global.gc) global.gc();
    const mem = process.memoryUsage();
    console.log(`[${new Date().toISOString()}] RAM freed. Heap: ${(mem.heapUsed / 1024 / 1024).toFixed(1)} MB, RSS: ${(mem.rss / 1024 / 1024).toFixed(1)} MB`);
  }
}

// Watchdog timer for idle RAM cleanup
setInterval(unloadModelIfIdle, 30000);

// Authentication Middleware
function authMiddleware(req, res, next) {
  const key = req.headers['x-api-key'] || req.query.api_key;
  if (!key || key !== API_KEY) {
    return res.status(401).json({ detail: "Invalid or missing API Key. Provide 'X-API-Key' header." });
  }
  next();
}

function adminMiddleware(req, res, next) {
  const key = req.headers['x-admin-key'] || req.query.admin_key;
  if (!key || key !== ADMIN_KEY) {
    return res.status(403).json({ detail: "Invalid or missing Admin Key. Provide 'X-Admin-Key' header." });
  }
  next();
}

// Vector normalization helper
function normalize(vector) {
  let sumSq = 0;
  for (let i = 0; i < vector.length; i++) sumSq += vector[i] * vector[i];
  const norm = Math.sqrt(sumSq);
  if (norm === 0) return vector;
  const out = new Array(vector.length);
  for (let i = 0; i < vector.length; i++) out[i] = vector[i] / norm;
  return out;
}

function computeSha256(buffer, text) {
  const hash = crypto.createHash('sha256');
  if (buffer) {
    hash.update('IMG:');
    hash.update(buffer);
  }
  if (text) {
    hash.update('|TXT:');
    hash.update(text.trim().toLowerCase());
  }
  return hash.digest('hex');
}

function recordHistory(ip, reqType, sha, vector, text) {
  historyId++;
  const entry = {
    id: historyId,
    ip: ip || 'unknown',
    timestamp: new Date().toISOString(),
    type: reqType,
    sha256: sha,
    text: text || null,
    vector
  };

  if (history.length >= HISTORY_LIMIT) {
    const oldest = history.shift();
    cacheMap.delete(oldest.sha256);
  }

  history.push(entry);
  cacheMap.set(sha, entry);
  return entry;
}

// --- ENDPOINTS ---

app.get('/health', (req, res) => {
  const mem = process.memoryUsage();
  res.json({
    status: 'healthy',
    runtime: 'Node.js ' + process.version,
    model: MODEL_NAME,
    uptime_seconds: Math.round((Date.now() - startTime) / 1000),
    memory_rss_mb: Math.round(mem.rss / 1024 / 1024),
    heap_used_mb: Math.round(mem.heapUsed / 1024 / 1024),
    total_requests: totalRequests,
    cache_hits: cacheHits,
    in_memory_history_count: history.length
  });
});

const uploadMiddleware = (req, res, next) => {
  const ct = req.headers['content-type'] || '';
  if (ct.includes('multipart/form-data')) {
    return upload.single('file')(req, res, next);
  }
  next();
};

app.post('/embed', authMiddleware, uploadMiddleware, async (req, res) => {
  totalRequests++;
  const ip = req.ip || req.connection.remoteAddress;
  const imageBuffer = req.file ? req.file.buffer : null;
  const rawText = req.body && req.body.text ? req.body.text : null;
  const text = rawText && typeof rawText === 'string' && rawText.trim() ? rawText.trim() : null;
  const imageWeight = (req.body && req.body.image_weight) ? parseFloat(req.body.image_weight) : DEFAULT_IMAGE_WEIGHT;
  const textWeight = (req.body && req.body.text_weight) ? parseFloat(req.body.text_weight) : DEFAULT_TEXT_WEIGHT;

  if (!imageBuffer && !text) {
    return res.status(400).json({ detail: "Must provide either an image file, a text string, or both." });
  }

  const reqType = (imageBuffer && text) ? 'image+text' : (imageBuffer ? 'image' : 'text');
  const sha = computeSha256(imageBuffer, text);

  // Cache lookup
  if (cacheMap.has(sha)) {
    cacheHits++;
    const cached = cacheMap.get(sha);
    return res.json({
      success: true,
      type: reqType,
      sha256: sha,
      dim: cached.vector.length,
      cached: true,
      latency_ms: 0.0,
      embedding: cached.vector
    });
  }

  const t0 = Date.now();
  try {
    const { visionModel, textModel, processor, tokenizer } = await getOrLoadModel();
    let imageVec = null;
    let textVec = null;

    if (imageBuffer) {
      const rawImage = await RawImage.fromBlob(new Blob([imageBuffer]));
      const imageInputs = await processor(rawImage);
      const { image_embeds } = await visionModel(imageInputs);
      imageVec = Array.from(normalize(image_embeds.data));
    }

    if (text) {
      const textInputs = tokenizer(text);
      const { text_embeds } = await textModel(textInputs);
      textVec = Array.from(normalize(text_embeds.data));
    }

    let finalVec = null;
    if (imageVec && textVec) {
      const fused = new Array(imageVec.length);
      for (let i = 0; i < imageVec.length; i++) {
        fused[i] = (imageWeight * imageVec[i]) + (textWeight * textVec[i]);
      }
      finalVec = normalize(fused);
    } else if (imageVec) {
      finalVec = imageVec;
    } else {
      finalVec = textVec;
    }

    const latencyMs = Date.now() - t0;
    recordHistory(ip, reqType, sha, finalVec, text);

    return res.json({
      success: true,
      type: reqType,
      sha256: sha,
      dim: finalVec.length,
      cached: false,
      latency_ms: latencyMs,
      embedding: finalVec
    });
  } catch (err) {
    console.error("Inference error:", err);
    return res.status(500).json({ detail: `Model inference error: ${err.message}` });
  }
});

app.post('/embed/text', authMiddleware, async (req, res) => {
  req.body = { text: req.body.text };
  return app._router.handle(req, res);
});

app.get('/admin/history', adminMiddleware, (req, res) => {
  const includeVectors = req.query.include_vectors !== 'false';
  const limit = Math.min(parseInt(req.query.limit || '100'), 100);
  const records = history.slice(-limit).reverse();

  if (!includeVectors) {
    return res.json(records.map(r => ({
      id: r.id,
      ip: r.ip,
      timestamp: r.timestamp,
      type: r.type,
      sha256: r.sha256,
      text: r.text,
      dim: r.vector.length
    })));
  }

  return res.json(records);
});

app.listen(PORT, '0.0.0.0', () => {
  console.log(`Node.js Embedding Server running on http://0.0.0.0:${PORT}`);
});

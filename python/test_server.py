import os
import sys
import time
import requests

BASE_URL = "http://127.0.0.1:8000"
API_KEY = "my_secure_api_key_12345"
ADMIN_KEY = "my_secure_admin_key_67890"

def test_health():
    print("\n--- 1. Testing GET /health ---")
    r = requests.get(f"{BASE_URL}/health")
    print(f"Status: {r.status_code}")
    print("Response:", r.json())
    assert r.status_code == 200

def test_unauthorized():
    print("\n--- 2. Testing Unauthorized Access (No API Key) ---")
    r = requests.post(f"{BASE_URL}/embed", data={"text": "test query"})
    print(f"Status: {r.status_code} (Expected 401)")
    print("Response:", r.json())
    assert r.status_code == 401

def test_image_embed(sample_image_path):
    print("\n--- 3. Testing POST /embed (Image Only) ---")
    headers = {"X-API-Key": API_KEY}
    with open(sample_image_path, "rb") as f:
        files = {"file": (os.path.basename(sample_image_path), f, "image/jpeg")}
        t0 = time.time()
        r = requests.post(f"{BASE_URL}/embed", headers=headers, files=files)
        
    print(f"Status: {r.status_code}")
    data = r.json()
    print(f"Type: {data.get('type')}, Dim: {data.get('dim')}, Cached: {data.get('cached')}, Latency: {data.get('latency_ms')} ms")
    print(f"SHA-256: {data.get('sha256')[:16]}...")
    print(f"Vector Sample (first 5 values): {data.get('embedding')[:5]}")
    assert r.status_code == 200
    assert data.get("dim") == 512
    assert not data.get("cached")

    print("\n--- 4. Testing Deduplication Cache (Same Image Repeat) ---")
    with open(sample_image_path, "rb") as f:
        files = {"file": (os.path.basename(sample_image_path), f, "image/jpeg")}
        r2 = requests.post(f"{BASE_URL}/embed", headers=headers, files=files)
    data2 = r2.json()
    print(f"Status: {r2.status_code}")
    print(f"Cached Hit: {data2.get('cached')} (Latency: {data2.get('latency_ms')} ms)")
    assert data2.get("cached") == True

def test_text_embed():
    print("\n--- 5. Testing POST /embed (Text Only) ---")
    headers = {"X-API-Key": API_KEY}
    r = requests.post(f"{BASE_URL}/embed", headers=headers, data={"text": "herbal organic shampoo"})
    print(f"Status: {r.status_code}")
    data = r.json()
    print(f"Type: {data.get('type')}, Dim: {data.get('dim')}, Cached: {data.get('cached')}, Latency: {data.get('latency_ms')} ms")
    assert r.status_code == 200

def test_image_plus_text_embed(sample_image_path):
    print("\n--- 6. Testing POST /embed (Image + Text Fused) ---")
    headers = {"X-API-Key": API_KEY}
    with open(sample_image_path, "rb") as f:
        files = {"file": (os.path.basename(sample_image_path), f, "image/jpeg")}
        data_fields = {
            "text": "Premium Baby Care Set for New Borns",
            "image_weight": "0.6",
            "text_weight": "0.4"
        }
        r = requests.post(f"{BASE_URL}/embed", headers=headers, files=files, data=data_fields)
        
    print(f"Status: {r.status_code}")
    data = r.json()
    print(f"Type: {data.get('type')}, Dim: {data.get('dim')}, Cached: {data.get('cached')}, Latency: {data.get('latency_ms')} ms")
    assert r.status_code == 200
    assert data.get("type") == "image+text"

def test_admin_history():
    print("\n--- 7. Testing Admin History Unauthorized (No Admin Key) ---")
    r = requests.get(f"{BASE_URL}/admin/history")
    print(f"Status: {r.status_code} (Expected 403)")
    assert r.status_code == 403

    print("\n--- 8. Testing GET /admin/history (With Admin Key) ---")
    headers = {"X-Admin-Key": ADMIN_KEY}
    r = requests.get(f"{BASE_URL}/admin/history?include_vectors=false", headers=headers)
    print(f"Status: {r.status_code}")
    history = r.json()
    print(f"Total History Items in Memory: {len(history)}")
    for item in history:
        print(f"  ID #{item['id']} | IP: {item['ip']} | Type: {item['type']:<10} | SHA: {item['sha256'][:12]}... | Text: {item['text']}")
    assert r.status_code == 200

if __name__ == "__main__":
    sample_img = os.path.join("..", "products", "baby-products", "baby-care-set.jpg")
    if not os.path.exists(sample_img):
        sample_img = os.path.join("products", "baby-products", "baby-care-set.jpg")
    
    test_health()
    test_unauthorized()
    test_image_embed(sample_img)
    test_text_embed()
    test_image_plus_text_embed(sample_img)
    test_admin_history()
    print("\n>>> ALL TESTS PASSED SUCCESSFULLY! <<<\n")

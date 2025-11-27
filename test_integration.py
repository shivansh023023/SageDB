import requests
import json
import time
import sys

BASE_URL = "http://localhost:8000"

def log(msg):
    print(f"[TEST] {msg}")

def test_health():
    log("Testing Health Check...")
    try:
        resp = requests.get(f"{BASE_URL}/health")
        resp.raise_for_status()
        print(json.dumps(resp.json(), indent=2))
    except Exception as e:
        log(f"Health check failed: {e}")
        sys.exit(1)

def test_create_nodes():
    log("Testing Node Creation...")
    nodes = [
        {"text": "Python is a programming language", "type": "concept", "metadata": {"category": "tech"}},
        {"text": "FastAPI is a web framework for Python", "type": "entity", "metadata": {"category": "tech"}},
        {"text": "Graph databases store relationships", "type": "concept", "metadata": {"category": "db"}},
        {"text": "Vector databases store embeddings", "type": "concept", "metadata": {"category": "db"}}
    ]
    
    created_ids = []
    for n in nodes:
        resp = requests.post(f"{BASE_URL}/v1/nodes", json=n)
        if resp.status_code == 200:
            data = resp.json()
            created_ids.append(data['uuid'])
            print(f"Created Node: {data['uuid']} - {data['text'][:30]}...")
        else:
            print(f"Failed to create node: {resp.text}")
    return created_ids

def test_create_edges(node_ids):
    log("Testing Edge Creation...")
    if len(node_ids) < 4:
        log("Not enough nodes to test edges")
        return

    # Python -> FastAPI
    e1 = {"source_id": node_ids[0], "target_id": node_ids[1], "relation": "has_framework", "weight": 0.9}
    # Graph DB -> Vector DB (just a relation)
    e2 = {"source_id": node_ids[2], "target_id": node_ids[3], "relation": "related_to", "weight": 0.5}
    
    for e in [e1, e2]:
        resp = requests.post(f"{BASE_URL}/v1/edges", json=e)
        if resp.status_code == 200:
            print(f"Created Edge: {e['source_id']} -> {e['target_id']}")
        else:
            print(f"Failed to create edge: {resp.text}")

def test_hybrid_search():
    log("Testing Hybrid Search...")
    query = {
        "text": "python web framework",
        "top_k": 5,
        "alpha": 0.7,
        "beta": 0.3
    }
    
    resp = requests.post(f"{BASE_URL}/v1/search/hybrid", json=query)
    if resp.status_code == 200:
        results = resp.json()['results']
        print(f"Found {len(results)} results:")
        for r in results:
            print(f" - [{r['score']:.4f}] {r['text']} (Graph Score: {r['graph_score']:.4f})")
    else:
        print(f"Search failed: {resp.text}")

def main():
    # Wait for server to start
    time.sleep(2)
    
    test_health()
    ids = test_create_nodes()
    test_create_edges(ids)
    test_hybrid_search()
    log("Tests Completed.")

if __name__ == "__main__":
    main()

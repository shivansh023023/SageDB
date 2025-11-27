import requests
import time
import json
import sys

BASE_URL = "http://localhost:8000"

def wait_for_server(retries=10, delay=2):
    print("Waiting for server to be ready...")
    for i in range(retries):
        try:
            response = requests.get(f"{BASE_URL}/health")
            if response.status_code == 200:
                print("Server is ready!")
                return True
        except requests.exceptions.ConnectionError:
            pass
        time.sleep(delay)
    print("Server failed to start.")
    return False

def test_create_node(text, node_type, metadata):
    url = f"{BASE_URL}/v1/nodes"
    payload = {
        "text": text,
        "type": node_type,
        "metadata": metadata
    }
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        print(f"Created node: {response.json()['uuid']}")
        return response.json()
    else:
        print(f"Failed to create node: {response.text}")
        return None

def test_create_edge(source_id, target_id, relation, weight):
    url = f"{BASE_URL}/v1/edges"
    payload = {
        "source_id": source_id,
        "target_id": target_id,
        "relation": relation,
        "weight": weight
    }
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        print(f"Created edge from {source_id} to {target_id}")
        return response.json()
    else:
        print(f"Failed to create edge: {response.text}")
        return None

def test_hybrid_search(query_text, alpha=0.7, beta=0.3, top_k=5):
    url = f"{BASE_URL}/v1/search/hybrid"
    payload = {
        "text": query_text,
        "alpha": alpha,
        "beta": beta,
        "top_k": top_k
    }
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        print(f"Search results for '{query_text}' (alpha={alpha}, beta={beta}):")
        results = response.json()['results']
        for res in results:
            print(f" - {res['uuid']} (Score: {res['score']}): {res['text'][:50]}...")
        return results
    else:
        print(f"Search failed: {response.text}")
        return None

def test_graph_traversal(start_id, depth=2):
    url = f"{BASE_URL}/v1/search/graph"
    params = {
        "start_id": start_id,
        "depth": depth
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        print(f"Graph traversal from {start_id} (depth={depth}):")
        print(json.dumps(response.json(), indent=2))
        return response.json()
    else:
        print(f"Graph traversal failed: {response.text}")
        return None

def test_delete_node(node_uuid):
    url = f"{BASE_URL}/v1/nodes/{node_uuid}"
    response = requests.delete(url)
    if response.status_code == 200:
        print(f"Deleted node: {node_uuid}")
        return True
    else:
        print(f"Failed to delete node: {response.text}")
        return False

def main():
    if not wait_for_server():
        sys.exit(1)

    print("\n--- Testing Node Creation ---")
    # Create nodes representing concepts/entities
    ai_node = test_create_node("Artificial Intelligence is a field of computer science.", "concept", {"category": "CS"})
    ml_node = test_create_node("Machine Learning is a subset of AI.", "concept", {"category": "CS"})
    dl_node = test_create_node("Deep Learning uses neural networks.", "concept", {"category": "CS"})
    python_node = test_create_node("Python is a popular programming language for AI.", "entity", {"category": "Language"})
    
    if not all([ai_node, ml_node, dl_node, python_node]):
        print("Failed to create some nodes. Exiting.")
        sys.exit(1)

    print("\n--- Testing Edge Creation ---")
    test_create_edge(ai_node['uuid'], ml_node['uuid'], "includes", 0.9)
    test_create_edge(ml_node['uuid'], dl_node['uuid'], "includes", 0.9)
    test_create_edge(python_node['uuid'], ai_node['uuid'], "used_for", 0.8)
    test_create_edge(python_node['uuid'], ml_node['uuid'], "used_for", 0.8)

    print("\n--- Testing Vector Search (Alpha=1.0, Beta=0.0) ---")
    test_hybrid_search("neural networks", alpha=1.0, beta=0.0)

    print("\n--- Testing Hybrid Search (Alpha=0.5, Beta=0.5) ---")
    test_hybrid_search("programming language", alpha=0.5, beta=0.5)

    print("\n--- Testing Graph Traversal ---")
    test_graph_traversal(ai_node['uuid'], depth=2)

    print("\n--- Testing Node Deletion ---")
    test_delete_node(dl_node['uuid'])
    
    print("\n--- Verifying Deletion (Search should not find it) ---")
    test_hybrid_search("neural networks", alpha=1.0, beta=0.0)

if __name__ == "__main__":
    main()

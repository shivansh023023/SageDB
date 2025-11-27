import requests
import random
import time

API_URL = "http://localhost:8000"

def create_node(text, node_type, metadata):
    try:
        response = requests.post(f"{API_URL}/v1/nodes", json={
            "text": text,
            "type": node_type,
            "metadata": metadata
        })
        if response.status_code == 200:
            return response.json()['uuid']
        else:
            print(f"Failed to create node: {response.text}")
            return None
    except Exception as e:
        print(f"Error creating node: {e}")
        return None

def create_edge(source, target, relation, weight):
    try:
        response = requests.post(f"{API_URL}/v1/edges", json={
            "source_id": source,
            "target_id": target,
            "relation": relation,
            "weight": weight
        })
        if response.status_code != 200:
            print(f"Failed to create edge: {response.text}")
    except Exception as e:
        print(f"Error creating edge: {e}")

def main():
    print("Starting database population...")
    
    # Topics for mock data
    topics = ["Artificial Intelligence", "Machine Learning", "Neural Networks", "Deep Learning", 
              "Natural Language Processing", "Computer Vision", "Reinforcement Learning", 
              "Data Science", "Big Data", "Cloud Computing"]
    
    node_ids = []
    
    # Create Nodes
    print("Creating nodes...")
    for i, topic in enumerate(topics):
        # Create a main concept node
        uuid = create_node(
            text=f"{topic} is a rapidly evolving field in technology.",
            node_type="concept",
            metadata={"category": "technology", "importance": "high"}
        )
        if uuid:
            node_ids.append(uuid)
            print(f"Created concept: {topic}")
            
        # Create related entity nodes
        for j in range(3):
            sub_topic = f"{topic} - Subconcept {j+1}"
            uuid = create_node(
                text=f"Details about {sub_topic} which is related to {topic}.",
                node_type="entity",
                metadata={"category": "detail", "parent": topic}
            )
            if uuid:
                node_ids.append(uuid)
    
    print(f"Created {len(node_ids)} nodes.")
    
    # Create Edges
    print("Creating edges...")
    edge_count = 0
    
    # Connect nodes randomly but with some logic (clustering)
    for source in node_ids:
        # Connect to 2-4 random other nodes
        targets = random.sample(node_ids, k=random.randint(2, 4))
        for target in targets:
            if source != target:
                weight = random.uniform(0.1, 0.9)
                relation = random.choice(["related_to", "part_of", "depends_on", "influences"])
                create_edge(source, target, relation, weight)
                edge_count += 1
                
    print(f"Created {edge_count} edges.")
    print("Database population complete!")

if __name__ == "__main__":
    main()

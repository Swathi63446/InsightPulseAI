import os
import chromadb
from chromadb.utils import embedding_functions

# Setup paths
base_dir = os.path.dirname(os.path.abspath(__file__))
insights_dir = os.path.join(base_dir, "data", "insights")
db_path = os.path.join(base_dir, "data", "chroma_db")

# 1. Initialize ChromaDB
client = chromadb.PersistentClient(path=db_path)
# Use a high-quality, local embedding model
emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
collection = client.get_or_create_collection(name="sales_playbook", embedding_function=emb_fn)

# 2. Index the insights
for filename in os.listdir(insights_dir):
    if filename.endswith(".md"):
        path = os.path.join(insights_dir, filename)
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
            
        # Store in DB with the filename as the ID
        collection.add(
            documents=[content],
            ids=[filename],
            metadatas=[{"source": filename}]
        )
        print(f"Indexed: {filename}")

print("\nKnowledge base build complete!")
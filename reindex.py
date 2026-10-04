"""
Re-index script: reads all existing vectors from the Pinecone index (which were
embedded with sentence-transformers/all-roberta-large-v1), re-embeds the text
chunks using Pinecone hosted inference (multilingual-e5-large), and upserts
them back so the index matches what the app now uses for querying.

Run once locally:
    pip install pinecone python-dotenv
    python reindex.py
"""

import os
import time
from pinecone import Pinecone
from dotenv import load_dotenv

load_dotenv()

BATCH_SIZE = 100   # vectors to fetch per list call
EMBED_BATCH = 96   # Pinecone inference max batch size

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index = pc.Index(name=os.environ["INDEX_NAME"], host=os.environ["HOST_NAME"])

stats = index.describe_index_stats()
total = stats["total_vector_count"]
print(f"Index has {total} vectors.")

# --- 1. Fetch all IDs ---
print("Fetching all vector IDs...")
all_ids = []
for ns_stats in stats.get("namespaces", {""}).values():
    pass  # namespace handling if needed

# Pinecone list() paginates by prefix
for id_batch in index.list(limit=BATCH_SIZE):
    all_ids.extend(id_batch)

print(f"Fetched {len(all_ids)} IDs.")

# --- 2. Fetch metadata in batches ---
print("Fetching metadata...")
chunks = {}  # id -> content string
for i in range(0, len(all_ids), BATCH_SIZE):
    batch_ids = all_ids[i : i + BATCH_SIZE]
    result = index.fetch(ids=batch_ids)
    for vec_id, vec in result["vectors"].items():
        content = vec.get("metadata", {}).get("content", "")
        chunks[vec_id] = content

print(f"Retrieved {len(chunks)} chunks with metadata.")

# --- 3. Re-embed and upsert ---
items = list(chunks.items())
print(f"Re-embedding {len(items)} chunks with multilingual-e5-large...")

for i in range(0, len(items), EMBED_BATCH):
    batch = items[i : i + EMBED_BATCH]
    ids   = [b[0] for b in batch]
    texts = [b[1] for b in batch]

    embeddings = pc.inference.embed(
        model="multilingual-e5-large",
        inputs=texts,
        parameters={"input_type": "passage"},
    )

    vectors = [
        {
            "id": ids[j],
            "values": embeddings[j]["values"],
            "metadata": {"content": texts[j]},
        }
        for j in range(len(ids))
    ]

    index.upsert(vectors=vectors)
    print(f"  Upserted {i + len(batch)}/{len(items)}")
    time.sleep(0.5)  # stay within rate limits

print("Re-indexing complete.")

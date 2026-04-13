import json
import faiss
import sqlite3
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import torch

from core.config import config

# --- 1. SETTINGS ---
DB_FILE = config.DB_FILE
INDEX_FILE = config.VECTORDCONFIG_PATH
MODEL_PATH = config.EMBEDDING_MODEL_PATH
INPUT_JSON = config.RAW_DATA_PATH
BATCH_SIZE = 64  
D = 384          

# --- 2. GPU HANDSHAKE ---
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Running on: {device.upper()}")

model = SentenceTransformer(MODEL_PATH, device=device)

# --- 3. SQLITE SETUP ---
conn = sqlite3.connect(DB_FILE)
cursor = conn.cursor()
# Create table: 'id' matches the FAISS sequential index (0, 1, 2...)
cursor.execute('''CREATE TABLE IF NOT EXISTS papers 
                  (id INTEGER PRIMARY KEY, arxiv_id TEXT, title TEXT , abstract TEXT)''')
conn.commit()

# --- 4. FAISS SETUP ---
quantizer = faiss.IndexFlatL2(D)
index = faiss.IndexIVFFlat(quantizer, D, 2048, faiss.METRIC_L2)

def stream_arxiv():
    with open(INPUT_JSON, 'r') as f:
        for line in f:
            yield json.loads(line)

# --- PHASE 1: TRAINING ---
print("Phase 1: Training on 50k samples...")
train_texts = []
for i, data in enumerate(stream_arxiv()):
    train_texts.append(f"{data['title']}. {data['abstract']}")
    if i >= 50000: break

train_embeddings = model.encode(train_texts, batch_size=128, show_progress_bar=True, convert_to_numpy=True)
index.train(train_embeddings)
del train_texts, train_embeddings

# --- PHASE 2: INGESTION (FAISS + SQLITE) ---
print("Phase 2: Ingesting vectors and metadata...")
batch_texts = []
batch_meta = []
current_id = 0

# Get total lines once for the progress bar
total_lines = sum(1 for _ in open(INPUT_JSON, 'r'))

for data in tqdm(stream_arxiv(), total=total_lines):
    batch_texts.append(f"{data['title']}. {data['abstract']}")
    # Store minimal info to keep DB small: (SequentialID, ArXivID, Title)
    batch_meta.append((current_id, data['id'], data['title'] , data['abstract']))
    current_id += 1
    
    if len(batch_texts) == BATCH_SIZE:
        # 1. Update FAISS using Autocast for memory efficiency
        with torch.amp.autocast(device_type='cuda'):
            # Note: we removed precision="fp16" from here
            embeddings = model.encode(batch_texts, convert_to_numpy=True)
        
        index.add(embeddings)
        
        # 2. Update SQLite
        cursor.executemany("INSERT INTO papers VALUES (?, ?, ? , ?)", batch_meta)
        conn.commit()
        
        batch_texts, batch_meta = [], []

# Final partial batch
if batch_texts:
    index.add(model.encode(batch_texts, convert_to_numpy=True))
    cursor.executemany("INSERT INTO papers VALUES (?, ?, ? , ?)", batch_meta)
    conn.commit()

# --- 5. SAVE & CLOSE ---
faiss.write_index(index, INDEX_FILE)
conn.close()
print(f"Done! Index and Database are ready for Phase 3.")

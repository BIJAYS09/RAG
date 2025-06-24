# rag_chat_app.py

import os
import faiss
import numpy as np
from bs4 import BeautifulSoup
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from typing import List
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import torch

# Initialize app
app = FastAPI()

# Allow CORS for all (for frontend communication)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === CONFIG ===
HTML_DIR = './Dataset_ABA/en-GB'
INDEX_PATH = './vector.index'
MODEL_NAME = './model/phi-1_5'  # Path to downloaded Phi-1.5 model
K = 3  # Number of relevant chunks to retrieve

# === Load Models ===
print("Loading embedding model...")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

print("Loading Phi-1.5 model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32,
    device_map={"": "cpu"},
    low_cpu_mem_usage=True
)

generator = pipeline(
    'text-generation',
    model=model,
    tokenizer=tokenizer,
)

# === Load or Build FAISS Index ===
all_chunks = []
chunk_embeddings = []

print("Reading and chunking HTML files...")

def extract_chunks_from_html(directory: str) -> List[str]:
    chunks = []
    for filename in os.listdir(directory):
        if filename.endswith(".html"):
            path = os.path.join(directory, filename)
            with open(path, 'r', encoding='utf-8') as f:
                soup = BeautifulSoup(f, 'html.parser')
                text = soup.get_text(separator=' ', strip=True)
                words = text.split()
                for i in range(0, len(words), 300):
                    chunk = ' '.join(words[i:i+300])
                    chunks.append(chunk)
    return chunks

all_chunks = extract_chunks_from_html(HTML_DIR)
chunk_embeddings = embedding_model.encode(all_chunks)

if os.path.exists(INDEX_PATH):
    print("Loading existing FAISS index...")
    index = faiss.read_index(INDEX_PATH)
else:
    print("Creating FAISS index...")
    d = chunk_embeddings[0].shape[0]
    index = faiss.IndexFlatL2(d)
    index.add(np.array(chunk_embeddings))
    faiss.write_index(index, INDEX_PATH)

# === Helper Functions ===

def retrieve_context(query: str, k=K) -> str:
    query_embedding = embedding_model.encode([query])
    _, indices = index.search(np.array(query_embedding), k)
    context = "\n\n".join([all_chunks[i] for i in indices[0]])
    return context

def build_prompt(context: str, query: str) -> str:
    base = "You are an assistant. Use the context below to answer the question. If the answer isn't in the context, say you don't know.\n\nContext:\n"
    question = f"\n\nQuestion: {query}\nAnswer:"
    max_chars = 2000  # ~512 tokens
    context_trimmed = context[:(max_chars - len(base + question))]
    return base + context_trimmed + question

# === API Models ===
class Query(BaseModel):
    message: str

@app.post("/chat")
def chat(query: Query):
    context = retrieve_context(query.message)
    prompt = build_prompt(context, query.message)
    print("Generating response...")
    output = generator(prompt, max_new_tokens=512, temperature=0.2)[0]['generated_text']
    answer = output.split('Answer:')[-1].strip()
    return {"response": answer}

# === Run App ===
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)

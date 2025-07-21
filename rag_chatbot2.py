import os
import subprocess
import time
from bs4 import BeautifulSoup
from llama_index.core import VectorStoreIndex, Settings, StorageContext, load_index_from_storage
from langchain_huggingface import HuggingFaceEmbeddings
from llama_index.core import Document
from rapidfuzz import fuzz
from llama_index.core.node_parser import SentenceSplitter


import json
import threading

LOG_FILE = "evaluation_logs.json"
LOG_LOCK = threading.Lock()

def load_logs():
    try:
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []

def save_logs(logs):
    with LOG_LOCK:
        with open(LOG_FILE, "w", encoding="utf-8") as f:
            json.dump(logs, f, ensure_ascii=False, indent=2)

# Path to persist index
INDEX_STORAGE_DIR = "./index_storage"

# Store evaluation samples
EVALUATION_LOGS = []

def load_clean_html(directory):
    text_docs = []
    image_docs = []

    for filename in os.listdir(directory):
        if filename.endswith(".html"):
            full_path = os.path.join(directory, filename)
            with open(full_path, "r", encoding="utf-8") as f:
                raw_html = f.read()
                soup = BeautifulSoup(raw_html, "html.parser")

                # Clean text
                clean_text = soup.get_text(separator="\n")
                text_docs.append(Document(text=clean_text, metadata={"source": full_path}))

                # Images
                image_tags = soup.find_all("img")
                for img in image_tags:
                    src = img.get("src")
                    if src:
                        img_path = os.path.abspath(os.path.join(os.path.dirname(full_path), src))
                        image_docs.append(Document(
                            text=f"Image: {os.path.basename(img_path)}",
                            metadata={"source": full_path, "image_path": img_path}
                        ))

    return text_docs, image_docs






# Load documents
# Load and split
text_documents, image_documents = load_clean_html("./Dataset_ABA/en-GB")
splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)

text_nodes = splitter.get_nodes_from_documents(text_documents)

# Initialize embeddings
local_embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

TEXT_INDEX_DIR = "./index_text"
IMAGE_INDEX_DIR = "./index_image"


# Apply settings
Settings.embed_model = local_embed_model
Settings.context_window = 2048
Settings.chunk_size = 512
Settings.chunk_overlap = 50

# Build or load indexes
if os.path.exists(TEXT_INDEX_DIR):
    text_index = load_index_from_storage(StorageContext.from_defaults(persist_dir=TEXT_INDEX_DIR))
else:
    text_index = VectorStoreIndex(text_nodes)
    text_index.storage_context.persist(persist_dir=TEXT_INDEX_DIR)

if os.path.exists(IMAGE_INDEX_DIR):
    image_index = load_index_from_storage(StorageContext.from_defaults(persist_dir=IMAGE_INDEX_DIR))
else:
    image_index = VectorStoreIndex(image_documents)
    image_index.storage_context.persist(persist_dir=IMAGE_INDEX_DIR)

def retrieve_text_documents(query, top_k):
    similarity_threshold=0.3
    fuzzy_threshold=50
    retriever = text_index.as_retriever(similarity_top_k=top_k)
    nodes = retriever.retrieve(query)

    semantic_results = []
    for node in nodes:
        score = node.score if hasattr(node, 'score') else None
        if score is not None and score >= similarity_threshold:
            semantic_results.append((score, node.get_text(), node.metadata.get("source", "Unknown")))

    fuzzy_results = []
    all_docs = text_index.docstore.docs.values()
    for doc in all_docs:
        text = doc.get_text()
        similarity = fuzz.partial_ratio(query.lower(), text.lower())
        if similarity >= fuzzy_threshold:
            fuzzy_results.append((similarity / 100, text, doc.metadata.get("source", "Unknown")))

    # Combine and sort
    combined = semantic_results + fuzzy_results
    combined.sort(reverse=True, key=lambda x: x[0])

    # Remove duplicates by text
    seen = set()
    unique_contexts = []
    for _, text, source in combined:
        if text not in seen:
            image_list = []
            for doc in all_docs:
                if doc.get_text() == text:
                    image_list = doc.metadata.get("images", [])
                    break
            unique_contexts.append({
                "text": text,
                "source": source,
                "images": image_list
            })
            seen.add(text)
        if len(unique_contexts) >= top_k:
            break

    return unique_contexts

def retrieve_image_documents(query, top_k):
    fuzzy_threshold=60
    # Step 1: Semantic retrieval (OCR + caption stored in text)
    semantic_retriever = image_index.as_retriever(similarity_top_k=top_k)
    semantic_results = semantic_retriever.retrieve(query)

    # Step 2: Fuzzy matching on image filenames (useful when user says "photo of diagram.png")
    all_docs = image_index.docstore.docs.values()
    fuzzy_results = []
    for doc in all_docs:
        image_path = doc.metadata.get("image_path", "")
        if not image_path:
            continue

        filename = os.path.basename(image_path).lower()
        similarity = fuzz.partial_ratio(query.lower(), filename)
        if similarity >= fuzzy_threshold:
            # fuzzy_results.append((similarity / 100.0, doc))
            fuzzy_results.append(doc)  # just append the doc

    # Combine and deduplicate
    combined_nodes = semantic_results[:]
    seen_texts = set([node.get_text().strip() for node in semantic_results])

    for score, doc in sorted(fuzzy_results, key=lambda x: x[0], reverse=True):
        if doc.get_text().strip() not in seen_texts:
            combined_nodes.append(doc)
            seen_texts.add(doc.get_text().strip())

        if len(combined_nodes) >= top_k:
            break

    return combined_nodes



# Call Ollama model locally
def call_ollama(prompt):
    try:
        result = subprocess.run(
            ['ollama', 'run', 'llama3'],
            input=prompt,
            capture_output=True,
            text=True,
            encoding='utf-8',
            check=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        return f"Error calling Ollama: {e.stderr}"

def rag_chatbot(query):
    start_time = time.time()

    # Retrieve from both indexes
    text_nodes = retrieve_text_documents(query, top_k=10)
    image_nodes = retrieve_image_documents(query, top_k=5)

    all_nodes = text_nodes + image_nodes

    if not all_nodes:
        return "I don't know. No relevant content found.", 0, []

    # Build combined context
    context_parts = []
    reference_info = []
    seen_texts = set()
    context_list_for_ragas = []

    for i, node in enumerate(all_nodes, 1):
        # Differentiate between dicts (from text) and Nodes (from image)
        if isinstance(node, dict):
            text = node["text"].strip()
            context_list_for_ragas.append(text)  # For RAGAS
            source = node.get("source", "Unknown")
            image_paths = node.get("images", [])
        else:
            text = node.get_text().strip()
            context_list_for_ragas.append(text)  # For RAGAS
            source = node.metadata.get("source", "Unknown")
            image_path = node.metadata.get("image_path", None)
            image_paths = [image_path] if image_path else []

        if text in seen_texts:
            continue
        seen_texts.add(text)

        entry = f"[Source {i}] ({source}):\n{text}"
        if image_paths:
            entry += "\n" + "\n".join([f"Image Path: {img}" for img in image_paths])

        context_parts.append(entry)

        reference_info.append({
            "label": f"Source {i}",
            "path": source,
            "images": image_paths
        })

    context = "\n".join(context_parts)

    prompt = (
        f"Use the following context to answer the question accurately and only based on the provided information.\n"
        f"If image references are present, only refer to them if they contain useful visual information like diagrams or figures (not logos or generic visuals).\n\n"
        f"{context}\n\n"
        f"Question: {query}\n"
        f"Answer clearly and directly:"
    )

    response = call_ollama(prompt)
    elapsed_time = time.time() - start_time

    logs = load_logs()
    logs.append({
        "question": query,
        "answer": response,
        "contexts": context_list_for_ragas,
        "ground_truth": "HEllo"
    })
    save_logs(logs)

    return response, elapsed_time, reference_info



import os
import subprocess
import time
from bs4 import BeautifulSoup
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, StorageContext, load_index_from_storage
from langchain_huggingface import HuggingFaceEmbeddings
from llama_index.core import Document
from rapidfuzz import fuzz
from llama_index.core.node_parser import SentenceSplitter

# Path to persist index
INDEX_STORAGE_DIR = "./index_storage"

def load_clean_html(directory):
    documents = []
    for filename in os.listdir(directory):
        if filename.endswith(".html"):
            with open(os.path.join(directory, filename), "r", encoding="utf-8") as f:
                raw_html = f.read()
                soup = BeautifulSoup(raw_html, "html.parser")
                clean_text = soup.get_text(separator="\n")  # Get clean text only
                documents.append(Document(text=clean_text))
    return documents

# Load documents
#documents = SimpleDirectoryReader("./Dataset_ABA/en-GB").load_data()
parser = SentenceSplitter()
documents = load_clean_html("./Dataset_ABA/en-GB")
splitter = SentenceSplitter(chunk_size=256, chunk_overlap=50)
nodes = splitter.get_nodes_from_documents(documents)

# Initialize local embedding model
local_embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Apply settings
Settings.embed_model = local_embed_model
Settings.context_window = 2048
Settings.chunk_size = 256
Settings.chunk_overlap = 50

# Load or build index
if os.path.exists(INDEX_STORAGE_DIR):
    print("Loading existing index...")
    storage_context = StorageContext.from_defaults(persist_dir=INDEX_STORAGE_DIR)
    index = load_index_from_storage(storage_context)
else:
    print("Building new index...")
    index = VectorStoreIndex(nodes)
    index.storage_context.persist(persist_dir=INDEX_STORAGE_DIR)
    print("Index saved.")

# Retrieval function
def retrieve_documents(query, index=index, top_k=3, similarity_threshold=0.2, fuzzy_threshold=50):
    retriever = index.as_retriever(similarity_top_k=top_k)
    nodes = retriever.retrieve(query)

    # Semantic vector matches
    semantic_results = []
    for node in nodes:
        score = node.score if hasattr(node, 'score') else None
        if score is not None and score >= similarity_threshold:
            semantic_results.append((score, node.get_text()))

    # Fuzzy string matches
    fuzzy_results = []
    all_docs = index.docstore.docs.values()
    for doc in all_docs:
        text = doc.get_text()
        similarity = fuzz.partial_ratio(query.lower(), text.lower())
        if similarity >= fuzzy_threshold:
            fuzzy_results.append((similarity / 100, text)) # Normalize to 0–1 to match vector scores

    # Combine and sort both
    combined = semantic_results + fuzzy_results
    combined.sort(reverse=True, key=lambda x: x[0]) # sort by similarity score

    # Remove duplicates while preserving order
    seen = set()
    unique_contexts = []
    for _, text in combined:
        if text not in seen:
            unique_contexts.append(text)
            seen.add(text)
        if len(unique_contexts) >= top_k:
            break

    return unique_contexts


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

# RAG chatbot core logic
def rag_chatbot(query):
    start_time = time.time()
    contexts = retrieve_documents(query)

    # If no documents pass the similarity threshold
    if not contexts:
        return "I don't know. The answer is not found in the dataset.", 0

    context = "\n\n".join(contexts)
    # prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
    prompt = (
        f"Use the following context to answer the question accurately and only based on the provided information.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n"
        f"Answer clearly and directly:"
      )
    response = call_ollama(prompt)

    elapsed_time = time.time() - start_time

    if response.lower() in ["i don't know.", "i'm not sure.", "sorry, i don't know.", "sorry, i am not sure."]:
        return "I don't know. The answer is not found in the dataset.", elapsed_time

    return response, elapsed_time


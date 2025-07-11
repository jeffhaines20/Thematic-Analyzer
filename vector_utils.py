import faiss
import nltk
nltk.download('punkt_tab')
import hdbscan
from nltk.tokenize import sent_tokenize
from collections import defaultdict
from sentence_transformers import SentenceTransformer

embedder = None


def load_embedder():
    global embedder
    if embedder is None:
        print("[DBG] Loading SentenceTransformer...")
        embedder = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")  # force CPU
        print("[DBG] SentenceTransformer is loaded")


def prepare_chunks(text, window_size=3):
    print("Entered prepare_chunks")
    sentences = sent_tokenize(text)
    chunks = []
    chunk_map = []

    for size in range(1, window_size + 1):
        for i in range(len(sentences) - size + 1):
            chunk = " ".join(sentences[i:i+size])
            chunks.append(chunk)
            chunk_map.append((i, size))  # index + how many sentences

    return chunks, chunk_map


def build_faiss_index(chunks, embedder):
    print("Entered build_faiss_index")
    embeddings = embedder.encode(chunks, convert_to_numpy=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    return index, embeddings


@GPU(duration=120)
def match_quote_fast(quote, chunks, index, embeddings, embedder, threshold=0.65):
    quote_embedding = embedder.encode([quote], convert_to_numpy=True)

    # Use FAISS to get the nearest neighbor
    distances, indices = index.search(quote_embedding, k=1)
    best_idx = indices[0][0]
    best_dist = distances[0][0]

    # FAISS uses L2 distance — convert to cosine similarity approximation
    # For normalized vectors, cosine_sim ≈ 1 - (L2^2 / 2)
    similarity = 1 - (best_dist / 2)

    if similarity >= threshold:
        return chunks[best_idx], similarity
    else:
        return None, similarity


def cluster_chunks_hdbscan(embeddings, min_cluster_size=3):
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, metric='euclidean')
    cluster_labels = clusterer.fit_predict(embeddings)

    return cluster_labels


def group_chunks_by_cluster(chunks, labels):
    clustered = defaultdict(list)
    for label, chunk in zip(labels, chunks):
        if label != -1:  # HDBSCAN marks noise as -1
            clustered[label].append(chunk)

    return dict(clustered)


@GPU(duration=120)
def vectorize_text(text: str, window_size: int=2):
    global embedder
    print("Entered vectorize_text")
    chunks, chunk_map = prepare_chunks(text, window_size=window_size)
    index, embeddings = build_faiss_index(chunks, embedder)

    return chunks, index, embeddings, embedder


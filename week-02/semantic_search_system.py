"""Week 02 - Mini Semantic Search System

Loads FAQ data from CSV/JSON using pandas, cleans and splits the text into
chunks (question + answer), converts text into embeddings, stores them in
ChromaDB, and performs semantic search to display the top 3 most relevant matches.

Run it with:
    pip install -r requirements.txt
    python semantic_search_system.py
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import List, Tuple

import chromadb
import pandas as pd
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
CSV_PATH = DATA_DIR / "faqs.csv"
JSON_PATH = DATA_DIR / "faqs.json"
CHROMA_DB_DIR = BASE_DIR / "chroma_db"
COLLECTION_NAME = "faqs_semantic_search"
DEFAULT_MODEL = "all-MiniLM-L6-v2"


def load_faq_data(csv_path: Path = CSV_PATH, json_path: Path = JSON_PATH) -> pd.DataFrame:
    """Load FAQs from CSV if available, otherwise fall back to JSON."""
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} FAQs from {csv_path.name}")
        return df
    if json_path.exists():
        with json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        df = pd.DataFrame(data)
        print(f"Loaded {len(df)} FAQs from {json_path.name}")
        return df
    raise FileNotFoundError(f"No FAQ data found. Tried {csv_path} and {json_path}")


def clean_and_prepare_chunks(df: pd.DataFrame) -> pd.DataFrame:
    """Clean text and create chunks combining question and answer."""
    cleaned = df.copy()
    cleaned["question"] = cleaned["question"].astype(str).str.strip()
    cleaned["answer"] = cleaned["answer"].astype(str).str.strip()
    cleaned["chunk"] = cleaned["question"] + " " + cleaned["answer"]
    cleaned["chunk"] = cleaned["chunk"].str.replace(r"\s+", " ", regex=True)
    print(f"Prepared {len(cleaned)} text chunks (question + answer)")
    return cleaned


def build_index(
    df: pd.DataFrame,
    persist_dir: Path = CHROMA_DB_DIR,
    collection_name: str = COLLECTION_NAME,
    model_name: str = DEFAULT_MODEL,
) -> Tuple[chromadb.Collection, SentenceTransformer]:
    """Generate embeddings and store them in ChromaDB."""
    print(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)

    persist_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(persist_dir))

    # Use sentence-transformers embedding function for consistency
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=model_name
    )

    # Create or get collection
    try:
        client.delete_collection(name=collection_name)
    except Exception:
        pass

    collection = client.create_collection(
        name=collection_name,
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"},
    )

    documents = df["chunk"].tolist()
    metadatas = []
    ids = []
    for idx, row in df.iterrows():
        metadatas.append({"question": row["question"], "answer": row["answer"]})
        ids.append(f"faq_{idx}")

    print(f"Generating embeddings and storing {len(documents)} documents in ChromaDB...")
    collection.add(documents=documents, metadatas=metadatas, ids=ids)
    print("Index built successfully.")
    return collection, model


def semantic_search(
    collection: chromadb.Collection,
    query: str,
    top_k: int = 3,
) -> List[dict]:
    """Perform semantic search and return top_k results."""
    results = collection.query(query_texts=[query], n_results=top_k)
    matches = []
    if results and results.get("documents"):
        docs = results["documents"][0]
        metas = results["metadatas"][0] if results.get("metadatas") else []
        distances = results["distances"][0] if results.get("distances") else []
        for i in range(len(docs)):
            matches.append(
                {
                    "rank": i + 1,
                    "question": metas[i].get("question", "") if i < len(metas) else "",
                    "answer": metas[i].get("answer", "") if i < len(metas) else "",
                    "chunk": docs[i],
                    "distance": distances[i] if i < len(distances) else None,
                }
            )
    return matches


def display_results(query: str, matches: List[dict]) -> None:
    """Display search results in a readable format."""
    print(f"\nQuery: {query}")
    print("=" * 60)
    if not matches:
        print("No results found.")
        return
    for match in matches:
        print(f"\nTop {match['rank']}:")
        print(f"Question: {match['question']}")
        print(f"Answer: {match['answer']}")
        if match["distance"] is not None:
            print(f"Similarity distance: {match['distance']:.4f}")
    print("\n" + "=" * 60)


def main() -> None:
    """Run the full semantic search pipeline."""
    parser = argparse.ArgumentParser(description="Mini Semantic Search System")
    parser.add_argument(
        "--query",
        type=str,
        default="What is machine learning?",
        help="Query to search for (default: What is machine learning?)",
    )
    parser.add_argument("--top-k", type=int, default=3, help="Number of results (default: 3)")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Embedding model name")
    args = parser.parse_args()

    df = load_faq_data()
    df = clean_and_prepare_chunks(df)
    collection, _ = build_index(df, model_name=args.model)

    matches = semantic_search(collection, args.query, top_k=args.top_k)
    display_results(args.query, matches)


if __name__ == "__main__":
    main()

"""
rag_adapter.py — Lightweight RAG wrapper for Lab 24.

Replaces Day 18's Qdrant dependency with FAISS (in-memory).
Adds proper LLM generation using OpenAI gpt-4o-mini.

Usage:
    from rag_adapter import RAGPipeline
    rag = RAGPipeline()
    answer, contexts = rag.query("What is ...?")
"""

import os
import sys
import pickle
import asyncio
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Config ────────────────────────────────────────────────────────
DAY18_DATA = Path(__file__).parent.parent / "day018" / "2A202600424-TaVinhPhuc-Day18" / "data"
DOCS_DIR = Path(__file__).parent / "docs"
EMBEDDING_MODEL = "text-embedding-3-small"
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64
TOP_K = 5
CACHE_PATH = Path(__file__).parent / ".rag_index_cache.pkl"


class RAGPipeline:
    """Simple FAISS-based RAG pipeline using OpenAI embeddings + gpt-4o-mini."""

    def __init__(self, docs_dir: Path | None = None, force_rebuild: bool = False):
        import faiss
        import numpy as np
        from openai import OpenAI

        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self.faiss = faiss
        self.np = np
        self.docs_dir = docs_dir or self._pick_docs_dir()
        self.chunks: list[str] = []
        self.index = None

        if not force_rebuild and CACHE_PATH.exists():
            self._load_cache()
        else:
            self._build_index()
            self._save_cache()

    def _pick_docs_dir(self) -> Path:
        """Use Day 18 data if available, otherwise fall back to local docs/."""
        if DAY18_DATA.exists() and any(DAY18_DATA.iterdir()):
            print(f"[RAG] Using Day 18 corpus: {DAY18_DATA}")
            return DAY18_DATA
        DOCS_DIR.mkdir(exist_ok=True)
        print(f"[RAG] Using local docs/: {DOCS_DIR}")
        return DOCS_DIR

    def _load_docs(self) -> list[str]:
        """Load text from .md and .pdf files."""
        texts = []

        # Markdown files
        for f in self.docs_dir.glob("**/*.md"):
            texts.append(f.read_text(encoding="utf-8", errors="ignore"))

        # PDF files
        for f in self.docs_dir.glob("**/*.pdf"):
            try:
                from pypdf import PdfReader
                reader = PdfReader(str(f))
                pdf_text = "\n".join(
                    page.extract_text() or "" for page in reader.pages
                )
                texts.append(pdf_text)
                print(f"  [RAG] Loaded PDF: {f.name} ({len(reader.pages)} pages)")
            except Exception as e:
                print(f"  [RAG] Skip PDF {f.name}: {e}")

        if not texts:
            # Fallback: create minimal demo corpus so lab can run
            print("  [RAG] WARNING: No docs found. Creating minimal demo corpus.")
            texts = self._demo_corpus()

        print(f"[RAG] Loaded {len(texts)} documents")
        return texts

    def _demo_corpus(self) -> list[str]:
        """Minimal corpus for testing when no real docs available."""
        return [
            """Large Language Models (LLMs) are neural networks trained on vast text corpora.
They use transformer architecture with self-attention mechanisms.
Popular LLMs include GPT-4, Claude, and Gemini. They can generate text, answer questions,
summarize documents, and write code. Fine-tuning adapts pre-trained models to specific tasks.
RAG (Retrieval-Augmented Generation) combines LLMs with external knowledge bases.""",
            """RAG Pipeline Architecture: 1) Document ingestion and chunking. 2) Embedding generation.
3) Vector storage (FAISS, Qdrant, Pinecone). 4) Query embedding. 5) Similarity search.
6) Context injection into prompt. 7) LLM generation. Key metrics: faithfulness, answer relevancy,
context precision, context recall. Chunking strategies: fixed-size, recursive, semantic.""",
            """Evaluation metrics for RAG systems: Faithfulness measures if the answer is grounded
in the retrieved context. Answer Relevancy measures how well the answer addresses the question.
Context Precision measures the signal-to-noise ratio in retrieved chunks.
Context Recall measures if all relevant information was retrieved. RAGAS automates these metrics.""",
            """Guardrails in AI systems provide safety layers. Input guardrails: PII redaction,
topic validation, injection detection. Output guardrails: Llama Guard for safety classification,
hallucination detection. PII types: names, emails, phone numbers, ID numbers.
Presidio is Microsoft's PII detection library supporting multiple languages.""",
            """LLM-as-Judge uses language models to evaluate other models' outputs.
Pairwise comparison: present two answers, ask which is better. Absolute scoring: rate answers
on 1-5 scale. Biases to watch: position bias (prefer first answer), length bias (prefer longer),
self-preference bias. Mitigation: swap-and-average for position bias, length normalization.""",
        ]

    def _chunk_text(self, text: str) -> list[str]:
        """Simple overlapping chunk splitter."""
        words = text.split()
        chunks = []
        i = 0
        while i < len(words):
            chunk = " ".join(words[i: i + CHUNK_SIZE])
            if chunk.strip():
                chunks.append(chunk)
            i += CHUNK_SIZE - CHUNK_OVERLAP
        return chunks

    def _embed(self, texts: list[str]) -> "np.ndarray":
        """Embed texts using OpenAI text-embedding-3-small."""
        import time
        all_embeddings = []
        batch_size = 100
        for i in range(0, len(texts), batch_size):
            batch = texts[i: i + batch_size]
            # Retry on rate limit
            for attempt in range(3):
                try:
                    resp = self.client.embeddings.create(
                        model=EMBEDDING_MODEL, input=batch
                    )
                    all_embeddings.extend([e.embedding for e in resp.data])
                    break
                except Exception as e:
                    if attempt == 2:
                        raise
                    print(f"  [RAG] Embed retry {attempt+1}: {e}")
                    time.sleep(2 ** attempt)
        return self.np.array(all_embeddings, dtype="float32")

    def _build_index(self):
        """Build FAISS index from documents."""
        print("[RAG] Building FAISS index...")
        docs = self._load_docs()

        for doc in docs:
            self.chunks.extend(self._chunk_text(doc))

        print(f"[RAG] {len(self.chunks)} chunks, embedding...")
        vectors = self._embed(self.chunks)

        # Normalize for cosine similarity
        norms = self.np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = vectors / (norms + 1e-8)

        dim = vectors.shape[1]
        self.index = self.faiss.IndexFlatIP(dim)  # Inner product = cosine after normalize
        self.index.add(vectors)
        print(f"[RAG] Index built: {self.index.ntotal} vectors, dim={dim}")

    def _save_cache(self):
        with open(CACHE_PATH, "wb") as f:
            pickle.dump({"chunks": self.chunks, "index": self.faiss.serialize_index(self.index)}, f)
        print("[RAG] Index cached.")

    def _load_cache(self):
        with open(CACHE_PATH, "rb") as f:
            data = pickle.load(f)
        self.chunks = data["chunks"]
        self.index = self.faiss.deserialize_index(data["index"])
        print(f"[RAG] Loaded cached index: {len(self.chunks)} chunks")

    def retrieve(self, query: str, top_k: int = TOP_K) -> list[str]:
        """Retrieve top-k relevant chunks for query."""
        q_vec = self._embed([query])
        q_vec = q_vec / (self.np.linalg.norm(q_vec) + 1e-8)
        scores, indices = self.index.search(q_vec, top_k)
        return [self.chunks[i] for i in indices[0] if i < len(self.chunks)]

    def generate(self, query: str, contexts: list[str]) -> str:
        """Generate answer using gpt-4o-mini with retrieved contexts."""
        context_str = "\n\n---\n\n".join(contexts)
        resp = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful assistant. Answer the question ONLY based on the "
                        "provided context. If the context doesn't contain enough information, "
                        "say 'I don't have enough information to answer this.' "
                        "Be concise and accurate."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Context:\n{context_str}\n\nQuestion: {query}",
                },
            ],
            temperature=0,
            max_tokens=512,
        )
        return resp.choices[0].message.content.strip()

    def query(self, question: str, top_k: int = TOP_K) -> tuple[str, list[str]]:
        """Full RAG query: retrieve + generate. Returns (answer, contexts)."""
        contexts = self.retrieve(question, top_k=top_k)
        answer = self.generate(question, contexts)
        return answer, contexts

    async def query_async(self, question: str, top_k: int = TOP_K) -> tuple[str, list[str]]:
        """Async wrapper for use in full_pipeline.py."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.query, question, top_k)


# ── RAG v2: same pipeline with slight tweak for pairwise comparison ──────────
class RAGPipelineV2(RAGPipeline):
    """Version 2: top_k=3 (vs v1's top_k=5). Used for pairwise judge comparison."""

    def query(self, question: str, top_k: int = 3) -> tuple[str, list[str]]:
        contexts = self.retrieve(question, top_k=top_k)
        # V2 uses a slightly different system prompt (more concise)
        context_str = "\n\n---\n\n".join(contexts)
        resp = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Answer concisely using only the provided context. "
                        "One paragraph maximum. If unsure, say so."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Context:\n{context_str}\n\nQuestion: {question}",
                },
            ],
            temperature=0.2,
            max_tokens=256,
        )
        return resp.choices[0].message.content.strip(), contexts


if __name__ == "__main__":
    rag = RAGPipeline()
    q = "What is RAG and how does it work?"
    answer, contexts = rag.query(q)
    print(f"\nQ: {q}")
    print(f"A: {answer}")
    print(f"\nContexts ({len(contexts)}):")
    for i, c in enumerate(contexts):
        print(f"  [{i+1}] {c[:100]}...")

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

class VectorStore:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.index = faiss.IndexFlatL2(384)
        self.texts = []

    def add(self, texts):
        embeddings = self.model.encode(texts)
        self.index.add(np.array(embeddings))
        self.texts.extend(texts)

    def search(self, query, k=5):
        q_emb = self.model.encode([query])
        _, idx = self.index.search(np.array(q_emb), k)
        return [self.texts[i] for i in idx[0]]

    @classmethod
    def load(cls, kb_store_dir: str):
        """Load VectorStore from directory containing vector.index and vector_texts.jsonl"""
        import faiss
        import json
        from pathlib import Path

        kb_store = Path(kb_store_dir)
        index_path = kb_store / "vector.index"
        texts_path = kb_store / "vector_texts.jsonl"

        if not index_path.exists() or not texts_path.exists():
            raise FileNotFoundError(f"Required files not found in {kb_store_dir}")

            # Create new instance
        vs = cls()

        # Load FAISS index
        vs.index = faiss.read_index(str(index_path))

        # Load texts
        vs.texts = []
        with open(texts_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                vs.texts.append(data['text'])

        return vs  # Make sure this return statement exists
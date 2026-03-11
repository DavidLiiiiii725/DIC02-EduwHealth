import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

class VectorStore:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.index = faiss.IndexFlatL2(384)
        self.texts = []
        self.meta = []   # parallel list of metadata dicts (source, chunk_id, kb_domain)

    def add(self, texts, meta=None):
        """Add *texts* to the index with optional metadata.

        Parameters
        ----------
        texts : list[str]
        meta  : list[dict] | None
            Per-chunk metadata dicts.  Defaults to empty dicts when omitted.
        """
        if meta is None:
            meta = [{}] * len(texts)
        embeddings = self.model.encode(texts)
        self.index.add(np.array(embeddings))
        self.texts.extend(texts)
        self.meta.extend(meta)

    def search(self, query, k=5, domain: str = None):
        """Return top-k texts matching *query*.

        Parameters
        ----------
        query : str
            Natural-language query string.
        k : int
            Number of results to return.
        domain : str, optional
            If provided, restrict results to chunks whose ``kb_domain``
            metadata field equals *domain*.  When filtering would yield fewer
            than *k* results the search falls back to the unfiltered index.
        """
        q_emb = self.model.encode([query])

        if domain and self.meta:
            # Collect indices that belong to the requested domain
            domain_indices = [
                i for i, m in enumerate(self.meta)
                if m.get("kb_domain") == domain
            ]
            if len(domain_indices) >= k:
                # Build a temporary flat index over the domain subset
                subset_vecs = np.array(
                    [self.index.reconstruct(i) for i in domain_indices],
                    dtype=np.float32,
                )
                tmp_index = faiss.IndexFlatL2(subset_vecs.shape[1])
                tmp_index.add(subset_vecs)
                _, local_idx = tmp_index.search(np.array(q_emb, dtype=np.float32), k)
                # filter out FAISS placeholder indices (-1) that appear when
                # k exceeds the number of available vectors
                return [
                    self.texts[domain_indices[i]]
                    for i in local_idx[0]
                    if i >= 0
                ]

        # Default: search the full index
        _, idx = self.index.search(np.array(q_emb, dtype=np.float32), k)
        # FAISS returns -1 as a placeholder when fewer than k vectors are indexed
        return [self.texts[i] for i in idx[0] if i >= 0]

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

        vs = cls()

        # Load FAISS index
        vs.index = faiss.read_index(str(index_path))

        # Load texts and metadata
        vs.texts = []
        vs.meta = []
        with open(texts_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                vs.texts.append(data['text'])
                vs.meta.append(data.get('meta', {}))

        return vs
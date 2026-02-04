from pathlib import Path
import re
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import torch

RAW_FILE = Path("kb.txt")
OUT_DIR = Path("kb_store")

MODEL_NAME = "all-MiniLM-L6-v2"
DIM = 384

# ✅ 优化后的内存控制参数
CHUNK_MAX_CHARS = 800  # 减小单块大小
CHUNK_OVERLAP = 100  # 减小重叠
ENCODE_BATCH_SIZE = 8  # 减小编码批次
ADD_BATCH_CHUNKS = 128  # 减少添加批次
NORMALIZE = True  # normalize embeddings（便于相似度更稳定）


def topic_aware_chunk_text(text: str, max_chars: int = 800, overlap: int = 100):
    """按主题结构进行分块，保持主题完整性"""
    import re

    # 先按 --- 分割主题
    topic_blocks = re.split(r'\n---\n', text.strip())
    chunks = []

    for block in topic_blocks:
        block = block.strip()
        if not block:
            continue

            # 提取主题标题和内容
        topic_match = re.match(r'## Topic:(.+?)(?=\n\n|\n$)', block, re.DOTALL)
        if not topic_match:
            continue

        topic_title = topic_match.group(1).strip()
        topic_content = block[topic_match.end():].strip()

        # 如果主题内容较短，作为一个整体块
        if len(block) <= max_chars:
            chunks.append(block)
            continue

            # 如果主题内容较长，需要进一步分块
        # 将主题标题添加到每个块的开始
        full_topic_header = f"## Topic:{topic_title}\n\n"

        # 按段落分割内容
        paragraphs = re.split(r'\n\s*\n', topic_content)
        current_chunk = full_topic_header
        current_content = ""

        for para in paragraphs:
            # 检查添加这个段落后是否会超过限制
            test_content = current_content + ("\n\n" if current_content else "") + para
            test_chunk = full_topic_header + test_content

            if len(test_chunk) <= max_chars:
                current_content = test_content
            else:
                # 保存当前块
                if current_content.strip():
                    chunks.append(full_topic_header + current_content)

                    # 开始新块，保留重叠
                if current_content:
                    # 找到最后一个完整段落作为重叠
                    para_lines = current_content.split('\n\n')
                    if len(para_lines) > 1:
                        overlap_content = "\n\n".join(para_lines[-1:])  # 保留最后一段
                    else:
                        overlap_content = current_content[-overlap:] if len(
                            current_content) > overlap else current_content
                    current_content = overlap_content + "\n\n" + para
                else:
                    current_content = para

                    # 添加最后一个块
        if current_content.strip():
            chunks.append(full_topic_header + current_content)

    return chunks


def load_or_create_index(index_path: Path):
    if index_path.exists():
        return faiss.read_index(str(index_path))
    return faiss.IndexFlatL2(DIM)


def count_existing_items(jsonl_path: Path) -> int:
    if not jsonl_path.exists():
        return 0
        # 统计已写入条数（用于断点续跑）
    n = 0
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for _ in f:
            n += 1
    return n


def main():
    print(torch.cuda.is_available())
    if not RAW_FILE.exists():
        raise FileNotFoundError(f"Raw KB file not found: {RAW_FILE}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    index_path = OUT_DIR / "vector.index"
    jsonl_path = OUT_DIR / "vector_texts.jsonl"

    print("[1/5] Loading raw KB...")
    text = RAW_FILE.read_text(encoding="utf-8")
    chunks = topic_aware_chunk_text(text, max_chars=CHUNK_MAX_CHARS, overlap=CHUNK_OVERLAP)
    print(f"[2/5] Total chunks: {len(chunks)}")

    print("[3/5] Loading embedding model...")
    model = SentenceTransformer(MODEL_NAME, device="cuda")

    print("[4/5] Loading/creating FAISS index...")
    index = load_or_create_index(index_path)

    # 断点续跑：如果 jsonl 已有 N 行，则跳过前 N 个 chunks
    already = count_existing_items(jsonl_path)
    if already > 0:
        print(f"[RESUME] Detected {already} existing items in {jsonl_path.name}.")
        if already >= len(chunks):
            print("[DONE] Nothing to add.")
            return
        chunks_to_add = chunks[already:]
        start_id = already
    else:
        chunks_to_add = chunks
        start_id = 0

    print(f"[5/5] Encoding & adding to index (chunks to add: {len(chunks_to_add)})...")

    with open(jsonl_path, "a", encoding="utf-8") as f_jsonl:
        for base in range(0, len(chunks_to_add), ADD_BATCH_CHUNKS):
            batch_chunks = chunks_to_add[base: base + ADD_BATCH_CHUNKS]
            batch_ids = range(start_id + base, start_id + base + len(batch_chunks))

            # 分批 encode
            emb = model.encode(
                batch_chunks,
                batch_size=ENCODE_BATCH_SIZE,
                normalize_embeddings=NORMALIZE,
                show_progress_bar=True
            )
            emb = np.array(emb, dtype=np.float32)

            index.add(emb)
            del emb
            torch.cuda.empty_cache()

            for cid, ch in zip(batch_ids, batch_chunks):
                obj = {"text": ch, "meta": {"source": RAW_FILE.name, "chunk_id": cid}}
                f_jsonl.write(json.dumps(obj, ensure_ascii=False) + "\n")

                # 每批落盘
            faiss.write_index(index, str(index_path))

            done = base + len(batch_chunks)
            print(f"  - Added {done}/{len(chunks_to_add)} new chunks (total indexed: {index.ntotal})")

    print("\n[OK] Vector KB built/updated successfully.")
    print(f"- index:  {index_path}")
    print(f"- texts:  {jsonl_path}")
    print(f"- total vectors: {index.ntotal}")


if __name__ == "__main__":
    main()
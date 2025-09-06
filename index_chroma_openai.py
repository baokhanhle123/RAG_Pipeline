#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Indexing & Retrieval với ChromaDB + OpenAI Embeddings — phiên bản chống 400 (token limit)
& phù hợp Chroma 0.5.x (PersistentClient tự động persist, KHÔNG gọi client.persist()).

- Đếm token bằng tiktoken, cắt theo token nếu vượt limit, embed từng đoạn rồi MEAN-POOL.
- sanitize_metadata(): ép metadata về kiểu nguyên thủy.
- enforce_unique_ids(): tránh DuplicateIDError trong batch upsert.
"""

from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Any, Optional, Iterable, Tuple
import os, json, hashlib, re
import chromadb
from openai import OpenAI


# Hợp lệ cho tham số `include` khi query Chroma
VALID_INCLUDE = {"documents", "embeddings", "metadatas", "distances", "uris", "data"}

def _safe_include(items):
    """Lọc danh sách include theo whitelist để tránh ValueError."""
    if not items:
        return None
    out = [x for x in items if x in VALID_INCLUDE]
    return out or None


# ============ Cấu hình ============

EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small").strip()
PERSIST_DIR = Path(os.getenv("CHROMA_DIR", f"./chroma_pccc_{EMBED_MODEL.replace('-', '_')}"))
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION", f"pccc_vi_{EMBED_MODEL.replace('-', '_')}")
INPUT_JSONL = Path(os.getenv("PCCC_CHUNKS", "./pccc_chunks.jsonl"))

BATCH = int(os.getenv("EMBED_BATCH", "256"))
TOPK  = int(os.getenv("RETRIEVER_TOPK", "5"))

# Giới hạn token cho embeddings v3 (8,192 theo tài liệu). Để dư biên an toàn.
DEFAULT_TOKEN_LIMIT = 8192
TOKEN_MARGIN = int(os.getenv("EMBED_TOKEN_MARGIN", "256"))

# ============ Tiktoken (đếm token thật) ============

try:
    import tiktoken  # pip install tiktoken
except Exception:
    tiktoken = None

def _get_encoder_for_embedding_model(model: str):
    # embeddings v3 dùng 'cl100k_base'
    if tiktoken is None:
        return None
    return tiktoken.get_encoding("cl100k_base")

def _token_limit_for_model(model: str) -> int:
    env_lim = os.getenv("EMBED_MAX_TOKENS", "").strip()
    if env_lim.isdigit():
        return int(env_lim)
    return DEFAULT_TOKEN_LIMIT

def _count_tokens(text: str, enc) -> int:
    if enc is None:
        return max(1, len(text) // 2)  # fallback siêu an toàn
    return len(enc.encode(text))

def _split_into_token_slices(text: str, enc, max_tokens: int) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    if _count_tokens(text, enc) <= max_tokens:
        return [text]

    paragraphs = re.split(r"\n{2,}", text)
    pieces: List[str] = []
    buf = ""
    buf_tokens = 0

    def flush_buf():
        nonlocal buf, buf_tokens
        if buf.strip():
            pieces.append(buf.strip())
        buf = ""
        buf_tokens = 0

    def add_part(part: str):
        nonlocal buf, buf_tokens
        t = _count_tokens(part, enc)
        if buf_tokens + t <= max_tokens:
            buf = (buf + ("\n\n" if buf else "") + part).strip()
            buf_tokens += t
        else:
            if buf:
                flush_buf()
            if _count_tokens(part, enc) <= max_tokens:
                pieces.append(part.strip())
            else:
                sentences = re.split(r"(?<=[\.?!…])\s+", part)
                cur = ""
                cur_tok = 0
                for s in sentences:
                    ts = _count_tokens(s, enc)
                    if ts > max_tokens:
                        ids = enc.encode(s) if enc else list(s)
                        for i in range(0, len(ids), max_tokens):
                            seg_ids = ids[i:i+max_tokens]
                            seg_txt = enc.decode(seg_ids) if enc else "".join(seg_ids)
                            pieces.append(seg_txt.strip())
                        cur = ""
                        cur_tok = 0
                    elif cur_tok + ts <= max_tokens:
                        cur = (cur + " " + s).strip()
                        cur_tok += ts
                    else:
                        if cur.strip():
                            pieces.append(cur.strip())
                        cur = s.strip()
                        cur_tok = ts
                if cur.strip():
                    pieces.append(cur.strip())

    for p in paragraphs:
        p = p.strip()
        if not p:
            continue
        add_part(p)

    if buf:
        pieces.append(buf.strip())

    out: List[str] = []
    for seg in pieces:
        if not seg:
            continue
        if _count_tokens(seg, enc) <= max_tokens:
            out.append(seg)
        else:
            ids = enc.encode(seg) if enc else list(seg)
            for i in range(0, len(ids), max_tokens):
                seg_ids = ids[i:i+max_tokens]
                seg_txt = enc.decode(seg_ids) if enc else "".join(seg_ids)
                out.append(seg_txt.strip())
    return out

def _mean_pool(vectors: List[List[float]]) -> List[float]:
    if not vectors:
        return []
    dim = len(vectors[0])
    acc = [0.0] * dim
    for v in vectors:
        if len(v) != dim:
            raise ValueError(f"Vector dim mismatch: {len(v)} vs {dim}")
        for i in range(dim):
            acc[i] += v[i]
    for i in range(dim):
        acc[i] /= len(vectors)
    return acc

# ============ OpenAI Embeddings ============

def get_openai_client() -> OpenAI:
    key = os.getenv("OPENAI_API_KEY", "").strip()
    assert key, "Thiếu OPENAI_API_KEY"
    return OpenAI(api_key=key)

def detect_embedding_dim(client: OpenAI, model: str) -> int:
    vec = client.embeddings.create(model=model, input=["__dimension_probe__"]).data[0].embedding
    return len(vec)

def embed_texts_safe(client: OpenAI, texts: List[str], model: str) -> List[List[float]]:
    enc = _get_encoder_for_embedding_model(model)
    limit = _token_limit_for_model(model) - TOKEN_MARGIN
    out: List[List[float]] = []

    for item in texts:
        txt = (item or "").strip()
        if not txt:
            out.append([])
            continue

        if _count_tokens(txt, enc) <= limit:
            r = client.embeddings.create(model=model, input=[txt])
            out.append(r.data[0].embedding)
            continue

        segments = _split_into_token_slices(txt, enc, max_tokens=limit)
        seg_vecs: List[List[float]] = []
        for seg in segments:
            r1 = client.embeddings.create(model=model, input=[seg])
            seg_vecs.append(r1.data[0].embedding)
        out.append(_mean_pool(seg_vecs))

    return out

# ============ Chroma helpers ============

def get_or_create_collection(client: chromadb.PersistentClient, name: str, space: str = "cosine", meta: Optional[Dict[str, Any]] = None):
    md = {"hnsw:space": space}
    if meta: md.update(meta)
    try:
        col = client.get_or_create_collection(name=name, metadata=md)
    except Exception:
        col = client.get_collection(name=name)
    return col

def chunked(iterable, n: int):
    buf = []
    for x in iterable:
        buf.append(x)
        if len(buf) >= n:
            yield buf
            buf = []
    if buf:
        yield buf

# ============ Chống trùng ID trong batch ============

def enforce_unique_ids(ids: List[str], docs: List[str], metas: List[Dict[str, Any]]) -> List[str]:
    seen = set()
    out_ids = []
    for i, cid in enumerate(ids):
        new_id = cid
        if new_id in seen:
            md = metas[i] or {}
            base = f"{cid}|{md.get('doc_id','')}|{md.get('section_key','')}|{md.get('page_start')}|{md.get('page_end')}|{i}"
            suffix = hashlib.sha1(base.encode("utf-8")).hexdigest()[:8]
            new_id = f"{cid}__{suffix}"
            while new_id in seen:
                base += "_"
                suffix = hashlib.sha1(base.encode("utf-8")).hexdigest()[:8]
                new_id = f"{cid}__{suffix}"
        seen.add(new_id)
        out_ids.append(new_id)
    return out_ids

# ============ Sanitize metadata ============

DROP_META_KEYS = {
    "text_as_html", "orig_elements", "detection_class_prob", "coordinates",
    "parent_id", "parent_index", "links", "link_urls"
}

def _scalarize(v: Any, max_len: int = 4000) -> Any:
    if v is None or isinstance(v, (str, int, float, bool)):
        return v
    try:
        if isinstance(v, (list, tuple)):
            if all(isinstance(x, (str, int, float, bool, type(None))) for x in v):
                s = " | ".join("" if x is None else str(x) for x in v)
            else:
                s = json.dumps(v, ensure_ascii=False)
            return s[:max_len]
        if isinstance(v, dict):
            s = json.dumps(v, ensure_ascii=False)
            return s[:max_len]
        s = str(v)
        return s[:max_len]
    except Exception:
        return str(v)[:max_len]

def sanitize_metadata(md: Dict[str, Any]) -> Dict[str, Any]:
    clean: Dict[str, Any] = {}
    for k, v in (md or {}).items():
        if k in DROP_META_KEYS:
            continue
        if not isinstance(k, str):
            k = str(k)
        clean[k] = _scalarize(v)
    return clean

# ============ I/O JSONL ============

def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    assert path.exists(), f"Không thấy file: {path}"
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if not line: continue
            out.append(json.loads(line))
    return out

# ============ Ingest vào Chroma (upsert) ============

def ingest_chunks_to_chroma(
    jsonl_path: Path,
    collection_name: str,
    persist_dir: Path,
    embed_model: str,
    batch_size: int = 256,
) -> Tuple[int, int, int]:
    rows = read_jsonl(jsonl_path)
    assert rows, f"File rỗng: {jsonl_path}"

    client_oai = get_openai_client()
    embed_dim = detect_embedding_dim(client_oai, embed_model)

    # Dùng PersistentClient: tự động persist xuống đĩa (không gọi client.persist())
    persist_dir.mkdir(parents=True, exist_ok=True)
    client_chroma = chromadb.PersistentClient(path=str(persist_dir))
    collection = get_or_create_collection(
        client_chroma,
        collection_name,
        space="cosine",
        meta={"embedding_model": embed_model, "embedding_dim": embed_dim},
    )

    ids: List[str] = []
    docs: List[str] = []
    metas: List[Dict[str, Any]] = []

    for r in rows:
        cid = r["chunk_id"]
        text = r["text"]
        md = r.get("metadata", {}) or {}
        md.setdefault("doc_id", r.get("doc_id", ""))
        md.setdefault("source_sha1", r.get("source_sha1",""))
        md.setdefault("embedding_model", embed_model)
        md.setdefault("embedding_dim", embed_dim)
        md.setdefault("is_pccc", True)
        md = sanitize_metadata(md)

        ids.append(cid)
        docs.append(text)
        metas.append(md)

    ids = enforce_unique_ids(ids, docs, metas)

    n_total = len(ids)
    n_upserted = 0

    for idxs in chunked(range(n_total), batch_size):
        batch_texts = [docs[i] for i in idxs]
        batch_ids   = [ids[i]  for i in idxs]
        batch_meta  = [metas[i] for i in idxs]

        vectors = embed_texts_safe(client_oai, batch_texts, embed_model)

        collection.upsert(
            ids=batch_ids,
            documents=batch_texts,
            metadatas=batch_meta,
            embeddings=vectors,
        )
        n_upserted += len(batch_ids)

    # KHÔNG gọi client_chroma.persist() ở Chroma 0.5.x; PersistentClient tự động lưu.
    # Tham khảo: Persistent Client docs & storage layout.
    print(f"✅ Indexed {n_upserted}/{n_total} chunks → collection '{collection_name}' at '{persist_dir}' | dim={embed_dim}")
    return n_total, n_upserted, embed_dim

# ============ Retrieval demo ============

def build_where_filter(
    van_ban: Optional[str] = None,
    dieu: Optional[str] = None,
    khoan: Optional[str] = None,
    phu_luc: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    w = {}
    if van_ban: w["van_ban"] = {"$eq": van_ban}
    if dieu:    w["dieu"]    = {"$eq": dieu}
    if khoan:   w["khoan"]   = {"$eq": khoan}
    if phu_luc: w["phu_luc"] = {"$eq": phu_luc}
    if extra:   w.update(extra)
    return w or None

def retrieve(
    query: str,
    collection_name: str,
    persist_dir: Path,
    embed_model: str,
    top_k: int = 5,
    where: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    client_oai = get_openai_client()
    qvec = embed_texts_safe(client_oai, [query], embed_model)[0]

    client_chroma = chromadb.PersistentClient(path=str(persist_dir))
    collection = client_chroma.get_collection(name=collection_name)

    # Chỉ include các trường hợp lệ (ids luôn có sẵn, không cần request)
    include_fields = _safe_include(["documents", "metadatas", "distances"])

    res = collection.query(
        query_embeddings=[qvec],
        n_results=top_k,
        where=where,
        include=include_fields,  # KHÔNG có "ids" ở đây
    )

    # 'ids' vẫn luôn có trong kết quả theo tài liệu
    out = []
    ids_ = res.get("ids", [[]])[0]
    docs_ = res.get("documents", [[]])[0] if res.get("documents") else []
    metas_ = res.get("metadatas", [[]])[0] if res.get("metadatas") else []
    dists_ = res.get("distances", [[]])[0] if res.get("distances") else []

    n = max(len(ids_), len(docs_), len(metas_), len(dists_))
    for i in range(n):
        m = metas_[i] if i < len(metas_) and metas_ else {}
        out.append({
            "id": ids_[i] if i < len(ids_) else None,
            "distance": dists_[i] if i < len(dists_) else None,
            "text": docs_[i] if i < len(docs_) else None,
            "citation": (m or {}).get("citation") or "",
            "doc_id": (m or {}).get("doc_id") or "",
            "van_ban": (m or {}).get("van_ban") or "",
            "dieu": (m or {}).get("dieu"),
            "khoan": (m or {}).get("khoan"),
            "diem": (m or {}).get("diem"),
            "page_start": (m or {}).get("page_start"),
            "page_end": (m or {}).get("page_end"),
        })
    return {"query": query, "top_k": top_k, "results": out}


# ============ Main ============

if __name__ == "__main__":
    n_total, n_upserted, dim = ingest_chunks_to_chroma(
        jsonl_path=INPUT_JSONL,
        collection_name=COLLECTION_NAME,
        persist_dir=PERSIST_DIR,
        embed_model=EMBED_MODEL,
        batch_size=BATCH,
    )

    example_q = "Bình chữa cháy xách tay phải kiểm định định kỳ bao lâu một lần?"
    where = build_where_filter(van_ban=None)
    res = retrieve(
        query=example_q,
        collection_name=COLLECTION_NAME,
        persist_dir=PERSIST_DIR,
        embed_model=EMBED_MODEL,
        top_k=TOPK,
        where=where,
    )
    print("\n=== DEMO RESULTS ===")
    for r in res["results"]:
        preview = (r['text'] or "").replace("\n", " ")[:200]
        print(f"- {r['citation']} | dist={r['distance']:.4f}\n  → {preview}\n")

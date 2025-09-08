#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
index_chroma_openai.py — Ingest incremental cho PCCC

Mục tiêu:
- Đọc pccc_chunks.jsonl (đầu ra từ bước chunking "smart")
- Nhóm theo nguồn (source_sha1)
- Chỉ ingest những nguồn CHƯA có trong Chroma (skip trùng)
- Upsert theo batch (embed bằng OpenAI), log số lượng
- (Tuỳ chọn) Demo truy hồi + in citations

Yêu cầu:
  pip install -U "chromadb>=0.5" "openai>=1.30.0" "tiktoken>=0.6"
Biến môi trường:
  OPENAI_API_KEY=...
  CHROMA_DIR=./chroma_pccc_text_embedding_3_small
  CHROMA_COLLECTION=pccc_vi_text_embedding_3_small
  EMBED_MODEL=text-embedding-3-small
  DEMO_QUERY="..."   # tuỳ chọn, để chạy demo retrieve sau khi ingest
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Any
import argparse
import json
import os
import sys
import math
import time
import hashlib

import chromadb
from openai import OpenAI

# =========================
# Cấu hình & constants
# =========================

PERSIST_DIR = Path(os.getenv("CHROMA_DIR", "./chroma_pccc_text_embedding_3_small"))
COLLECTION  = os.getenv("CHROMA_COLLECTION", "pccc_vi_text_embedding_3_small")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")

BATCH_EMBED = int(os.getenv("BATCH_EMBED", "96"))        # batch khi gọi embedding
MAX_EMBED_TOKENS = int(os.getenv("MAX_EMBED_TOKENS", "8000"))  # cắt trước khi embed
PRINT_EVERY = int(os.getenv("PRINT_EVERY", "1000"))      # log tiến độ

JSONL_CHUNKS = Path(os.getenv("PCCC_CHUNKS", "./pccc_chunks.jsonl"))

# include hợp lệ cho query (ids luôn có sẵn nếu dùng query/get)
VALID_INCLUDE = {"documents", "embeddings", "metadatas", "distances", "uris", "data"}

# =========================
# OpenAI & tiktoken helpers
# =========================

def get_openai_client() -> OpenAI:
    key = os.getenv("OPENAI_API_KEY", "").strip()
    assert key, "Thiếu biến môi trường OPENAI_API_KEY"
    return OpenAI(api_key=key)

try:
    import tiktoken
    ENC = tiktoken.get_encoding("cl100k_base")
except Exception:
    ENC = None

def tok_len(s: str) -> int:
    if ENC is None:
        return max(1, len(s)//2)  # fallback
    return len(ENC.encode(s or ""))

def truncate_tokens(s: str, max_tokens: int) -> str:
    """Cắt text theo số token để không vượt giới hạn input embedding."""
    if tok_len(s) <= max_tokens:
        return s
    if ENC is None:
        return s[: max_tokens * 2]
    ids = ENC.encode(s or "")
    return ENC.decode(ids[:max_tokens])

# =========================
# JSONL I/O
# =========================

def iter_jsonl(path: Path) -> Iterable[dict]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                sys.stderr.write(f"[WARN] Bỏ dòng JSONL hỏng: {path}:{ln}\n")

def group_chunks_by_source_sha1(path: Path) -> Dict[str, List[dict]]:
    """
    Nhóm các record theo source_sha1; mỗi record mong đợi có:
      - chunk_id (id duy nhất)
      - text
      - metadata: chứa 'citation', 'van_ban', 'dieu', 'khoan', ...
      - source_sha1
    """
    groups: Dict[str, List[dict]] = {}
    for row in iter_jsonl(path):
        sha1 = row.get("source_sha1") or (row.get("metadata") or {}).get("source_sha1")
        if not sha1:
            # Bỏ qua dòng không xác định nguồn
            continue
        groups.setdefault(sha1, []).append(row)
    return groups

# =========================
# Chroma helpers (incremental)
# =========================

def get_persistent_client() -> chromadb.ClientAPI:
    # PersistentClient lưu DB tại thư mục path
    return chromadb.PersistentClient(path=str(PERSIST_DIR))

def get_or_create_collection(client: chromadb.ClientAPI, name: str):
    # Nếu đã tồn tại sẽ trả về collection cũ; nếu chưa có sẽ tạo mới
    # (get_or_create hành vi chuẩn của Chroma, docs Getting Started)
    return client.get_or_create_collection(name=name)

def chroma_has_source(collection, source_sha1: str) -> bool:
    """
    Kiểm tra nhanh xem source (theo SHA-1) đã được ingest chưa.
    Dùng filter metadata với Collection.get(where=..., limit=1).
    """
    res = collection.get(where={"source_sha1": {"$eq": source_sha1}}, limit=1)
    ids = res.get("ids") or []
    return len(ids) > 0

def chroma_has_any(collection, source_sha1_list: List[str]) -> Dict[str, bool]:
    """
    Kiểm tra theo lô: với $in để xác định các SHA-1 đã có.
    Lưu ý: có thể phải phân lô nếu danh sách quá dài.
    """
    present: Dict[str, bool] = {}
    if not source_sha1_list:
        return present
    # Phân lô an toàn (tránh where quá dài)
    step = 512
    for i in range(0, len(source_sha1_list), step):
        batch = source_sha1_list[i:i+step]
        res = collection.get(where={"source_sha1": {"$in": batch}}, limit=10_000)
        # ids trả về là danh sách phẳng (get), còn metadatas trả về song song
        metas = res.get("metadatas") or []
        # Lấy tập các sha1 xuất hiện
        for md in metas:
            sha1 = (md or {}).get("source_sha1")
            if sha1:
                present[sha1] = True
    return present

# =========================
# Sanitize metadata (đúng kiểu Chroma)
# =========================

def _to_scalar(v: Any) -> Any:
    # Chroma chỉ nhận str, int, float, bool, None làm metadata value
    # Nếu gặp list/dict -> chuyển sang JSON string
    if v is None or isinstance(v, (str, int, float, bool)):
        return v
    try:
        return json.dumps(v, ensure_ascii=False)
    except Exception:
        return str(v)

def sanitize_metadata(md: dict) -> dict:
    if not md:
        return {}
    out = {}
    for k, v in md.items():
        out[str(k)] = _to_scalar(v)
    return out

# =========================
# Embedding
# =========================

def embed_texts_safe(client: OpenAI, texts: List[str], model: str) -> List[List[float]]:
    """
    - Cắt từng text theo MAX_EMBED_TOKENS để tránh lỗi "prompt too long".
    - Gọi embeddings theo batch.
    """
    tr_texts = [truncate_tokens(t or "", MAX_EMBED_TOKENS) for t in texts]
    out: List[List[float]] = []
    for i in range(0, len(tr_texts), BATCH_EMBED):
        sub = tr_texts[i:i+BATCH_EMBED]
        r = client.embeddings.create(model=model, input=sub)
        # data giữ nguyên thứ tự input
        out.extend([d.embedding for d in r.data])
    return out

# =========================
# Ingest incremental
# =========================

def ingest_chunks_to_chroma(
    chunks_jsonl: Path,
    collection_name: str,
) -> Tuple[int, int, int]:
    """
    Trả về: (n_sources_total, n_sources_new, n_chunks_upserted)
    - n_sources_* tính theo source_sha1
    """
    assert chunks_jsonl.exists(), f"Không thấy file: {chunks_jsonl}"
    client_oai = get_openai_client()

    client_chroma = get_persistent_client()
    collection = get_or_create_collection(client_chroma, collection_name)

    groups = group_chunks_by_source_sha1(chunks_jsonl)
    source_sha1_list = sorted(groups.keys())
    if not source_sha1_list:
        print(f"[INFO] Không có record hợp lệ trong {chunks_jsonl}")
        return (0, 0, 0)

    # Kiểm tra các nguồn đã có trong Chroma (dùng $in)
    present_map = chroma_has_any(collection, source_sha1_list)  # dùng where với $in theo cookbook filters
    # (tài liệu Filters cho biết $eq/$in... dùng được trong get/query)  # noqa

    n_sources_total = len(source_sha1_list)
    n_sources_new = 0
    n_upserted = 0

    for idx, sha1 in enumerate(source_sha1_list, 1):
        if present_map.get(sha1) or chroma_has_source(collection, sha1):
            print(f"[SKIP] Source đã tồn tại (SHA1={sha1[:12]}...)")
            continue

        rows = groups[sha1]
        # de-dup theo chunk_id trong nhóm (phòng lặp dòng JSONL)
        uniq_by_id: Dict[str, dict] = {}
        for r in rows:
            rid = r.get("chunk_id") or hashlib.sha1((sha1 + "|" + (r.get("text") or "")).encode("utf-8")).hexdigest()[:16]
            uniq_by_id[rid] = r
        rows = list(uniq_by_id.values())

        ids: List[str] = []
        docs: List[str] = []
        metas: List[dict] = []

        for r in rows:
            cid = r.get("chunk_id") or hashlib.sha1((sha1 + "|" + (r.get("text") or "")).encode("utf-8")).hexdigest()[:16]
            text = r.get("text") or ""
            md = r.get("metadata") or {}
            # thêm source_sha1 vào metadata để filter nhanh
            md["source_sha1"] = sha1
            # chuẩn hoá metadata đúng kiểu
            md = sanitize_metadata(md)

            ids.append(str(cid))
            docs.append(text)
            metas.append(md)

        # Embed & upsert theo batch
        for i in range(0, len(ids), BATCH_EMBED):
            sub_ids = ids[i:i+BATCH_EMBED]
            sub_docs = docs[i:i+BATCH_EMBED]
            sub_metas = metas[i:i+BATCH_EMBED]
            vecs = embed_texts_safe(client_oai, sub_docs, EMBED_MODEL)

            collection.upsert(
                ids=sub_ids,
                embeddings=vecs,
                documents=sub_docs,
                metadatas=sub_metas,
            )
            n_upserted += len(sub_ids)

        n_sources_new += 1
        if n_upserted % PRINT_EVERY == 0:
            print(f"[PROGRESS] upserted={n_upserted} (sources processed={n_sources_new}/{n_sources_total})")

    print(f"✅ Indexed sources {n_sources_new}/{n_sources_total} → collection '{collection_name}' at '{PERSIST_DIR.name}'")
    return (n_sources_total, n_sources_new, n_upserted)

# =========================
# Retrieve (demo)
# =========================

def _safe_include(items):
    if not items: return None
    return [x for x in items if x in VALID_INCLUDE] or None

def embed_query(client: OpenAI, q: str, model: str) -> List[float]:
    return client.embeddings.create(model=model, input=[q]).data[0].embedding

def retrieve(
    query: str,
    collection_name: str,
    top_k: int = 5,
    where: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    client_oai = get_openai_client()
    qvec = embed_query(client_oai, query, EMBED_MODEL)

    client_chroma = get_persistent_client()
    collection = client_chroma.get_collection(name=collection_name)

    include_fields = _safe_include(["documents", "metadatas", "distances"])
    res = collection.query(
        query_embeddings=[qvec],
        n_results=top_k,
        where=where,
        include=include_fields,   # ids sẽ có sẵn trong kết quả query
    )

    out = []
    ids_ = res.get("ids", [[]])[0]
    docs_ = res.get("documents", [[]])[0] if res.get("documents") else []
    metas_ = res.get("metadatas", [[]])[0] if res.get("metadatas") else []
    dists_ = res.get("distances", [[]])[0] if res.get("distances") else []

    n = max(len(ids_), len(docs_), len(metas_), len(dists_))
    for i in range(n):
        m = metas_[i] if i < len(metas_) else {}
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

# =========================
# CLI
# =========================

def parse_args():
    p = argparse.ArgumentParser(description="Index incremental chunks vào Chroma (OpenAI embeddings).")
    p.add_argument("--chunks", type=str, default=str(JSONL_CHUNKS), help="Đường dẫn pccc_chunks.jsonl")
    p.add_argument("--collection", type=str, default=COLLECTION, help="Tên collection Chroma")
    p.add_argument("--demo_query", type=str, default=os.getenv("DEMO_QUERY", ""), help="Chuỗi truy vấn demo sau khi ingest")
    p.add_argument("--top_k", type=int, default=5, help="Số kết quả lấy khi demo retrieve")
    return p.parse_args()

def main():
    args = parse_args()
    chunks_path = Path(args.chunks)
    assert chunks_path.exists(), f"Không thấy file: {chunks_path}"

    n_total, n_new, n_up = ingest_chunks_to_chroma(
        chunks_jsonl=chunks_path,
        collection_name=args.collection,
    )

    if args.demo_query.strip():
        print("\n=== DEMO RESULTS ===")
        res = retrieve(args.demo_query, args.collection, top_k=args.top_k)
        for r in res["results"]:
            vb = r.get("van_ban") or "Văn bản"
            segs = []
            if r.get("dieu"): segs.append(f"Điều {r['dieu']}")
            if r.get("khoan"): segs.append(f"Khoản {r['khoan']}")
            if r.get("diem"): segs.append(f"Điểm {r['diem']}")
            label = f"{vb} — {', '.join(segs)}" if segs else vb
            dist = r.get("distance")
            print(f"- {label} | dist={dist:.4f}" if dist is not None else f"- {label}")
            txt = (r.get("text") or "").strip()
            if txt:
                print("  →", txt[:240].replace("\n"," ") + ("..." if len(txt) > 240 else ""))
        print()

if __name__ == "__main__":
    main()

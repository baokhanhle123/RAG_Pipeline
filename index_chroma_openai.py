#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Index incremental chunks (DOC/DOCX/PDF) vào Chroma bằng OpenAI Embeddings.
- Batching theo ngân sách token + fallback chia đôi khi dính 'max_tokens_per_request'.
- Retry/backoff khi gặp RateLimit/Timeout/5xx.
- Tự gán metadata 'modality' (table/image/text) + (tùy chọn) collection phụ cho TABLE/IMAGE (tái dùng vector).
- Kiểm tra 'citation' để đảm bảo trích dẫn chuẩn từ smart chunking.

Tham khảo:
- Chroma where / where_document filter khi truy vấn.  # docs.trychroma.com
- New OpenAI embeddings (text-embedding-3-small/large). # openai.com blog
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Any
import argparse, json, os, sys, hashlib, time, math, random

import chromadb
import openai as openai_root
from openai import OpenAI, BadRequestError  # dùng BadRequestError riêng
import math 
# Các lỗi còn lại lấy từ module openai_root để tương thích nhiều phiên bản

# -----------------------------
# Cấu hình & biến môi trường
# -----------------------------

PERSIST_DIR = Path(os.getenv("CHROMA_DIR", "./chroma_pccc_text_embedding_3_small"))
COLLECTION  = os.getenv("CHROMA_COLLECTION", "pccc_vi_text_embedding_3_small")
SPLIT_BY_MODALITY = os.getenv("SPLIT_BY_MODALITY", "0") == "1"
JSONL_CHUNKS = Path(os.getenv("PCCC_CHUNKS", "./pccc_chunks.jsonl"))
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")

BATCH_EMBED = int(os.getenv("BATCH_EMBED", "96"))
MAX_EMBED_TOTAL_TOKENS = int(os.getenv("MAX_EMBED_TOTAL_TOKENS", "280000"))  # an toàn < ~300k community-observed
MAX_EMBED_TOKENS = int(os.getenv("MAX_EMBED_TOKENS", "8000"))  # an toàn ~8k/input
PRINT_EVERY = int(os.getenv("PRINT_EVERY", "1000"))
SCHEMA_VERSION = os.getenv("PCCC_SCHEMA_VERSION", "pccc_chunks/v1.1")

VALID_INCLUDE = {"documents", "embeddings", "metadatas", "distances", "uris", "data"}

# -----------------------------
# OpenAI client & tokenizer
# -----------------------------

def get_openai_client() -> OpenAI:
    key = os.getenv("OPENAI_API_KEY", "").strip()
    assert key, "Thiếu OPENAI_API_KEY"
    return OpenAI(api_key=key)

try:
    import tiktoken
    ENC = tiktoken.get_encoding("cl100k_base")
except Exception:
    ENC = None

def tok_len(s: str) -> int:
    if ENC is None:
        return max(1, len(s)//2)
    return len(ENC.encode(s or ""))

def truncate_tokens(s: str, max_tokens: int) -> str:
    if tok_len(s) <= max_tokens:
        return s
    if ENC is None:
        return s[:max_tokens*2]
    ids = ENC.encode(s or "")
    return ENC.decode(ids[:max_tokens])

# -----------------------------
# JSONL helpers
# -----------------------------

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
    groups: Dict[str, List[dict]] = {}
    for row in iter_jsonl(path):
        sha1 = row.get("source_sha1") or (row.get("metadata") or {}).get("source_sha1")
        if not sha1:
            continue
        groups.setdefault(sha1, []).append(row)
    return groups

# -----------------------------
# Chroma helpers
# -----------------------------

def get_persistent_client() -> chromadb.ClientAPI:
    PERSIST_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(PERSIST_DIR))

def get_or_create_collection(client: chromadb.ClientAPI, name: str):
    return client.get_or_create_collection(name=name)

def chroma_has_any(collection, sha1_list: List[str]) -> Dict[str, bool]:
    present: Dict[str, bool] = {}
    if not sha1_list:
        return present
    step = 512
    for i in range(0, len(sha1_list), step):
        batch = sha1_list[i:i+step]
        res = collection.get(where={"source_sha1": {"$in": batch}}, limit=10000)
        for md in (res.get("metadatas") or []):
            sha = (md or {}).get("source_sha1")
            if sha:
                present[sha] = True
    return present

# -----------------------------
# Metadata sanitize & modality
# -----------------------------

def _to_scalar(v: Any) -> Any:
    if v is None or isinstance(v, (str, int, float, bool)):
        return v
    try:
        return json.dumps(v, ensure_ascii=False)
    except Exception:
        return str(v)


def sanitize_metadata(md: dict) -> dict:
    """
    Chuẩn hóa metadata để tương thích Chroma:
    - Bỏ key có value None
    - Bỏ float NaN/Inf
    - Giữ nguyên str/int/float/bool
    - Với list/dict/... -> json.dumps(...) (str)
    """
    if not md:
        return {"schema_version": SCHEMA_VERSION}

    out = {}
    for k, v in md.items():
        key = str(k)

        # 1) loại None
        if v is None:
            continue

        # 2) số đặc biệt: NaN/Inf -> bỏ
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            continue

        # 3) các kiểu được phép
        if isinstance(v, (str, int, float, bool)):
            out[key] = v
            continue

        # 4) mọi thứ khác -> stringify an toàn
        try:
            out[key] = json.dumps(v, ensure_ascii=False)
        except Exception:
            out[key] = str(v)

    out["schema_version"] = SCHEMA_VERSION
    return out


def _starts_with(s: str, prefix: str) -> bool:
    return (s or "").lstrip().startswith(prefix)

def detect_modality(text: str, md: dict) -> str:
    ht = md.get("has_table_html")
    if isinstance(ht, str):
        ht = ht.lower() in ("1", "true", "yes")
    if ht or _starts_with(text, "[BẢNG]") or _starts_with(text, "[BANG]"):
        return "table"
    hi = md.get("has_image")
    if isinstance(hi, str):
        hi = hi.lower() in ("1", "true", "yes")
    if hi or _starts_with(text, "[HÌNH]") or _starts_with(text, "[HINH]") or _starts_with(text, "[FIGURE]"):
        return "image"
    return "text"

# -----------------------------
# Embedding: token-budget batching + retry/backoff
# -----------------------------

def _call_embeddings(client: OpenAI, model: str, inputs: List[str]) -> List[List[float]]:
    r = client.embeddings.create(model=model, input=inputs)
    return [d.embedding for d in r.data]

def _call_embeddings_with_split_on_error(client: OpenAI, model: str, inputs: List[str]) -> List[List[float]]:
    try:
        return _call_embeddings(client, model, inputs)
    except BadRequestError as e:
        msg = (getattr(e, "message", None) or str(e) or "").lower()
        if ("max_tokens_per_request" in msg) or ("per request" in msg and "tokens" in msg):
            if len(inputs) == 1:
                # input đơn vẫn quá to → để dev điều chỉnh MAX_EMBED_TOKENS
                raise
            mid = len(inputs) // 2
            left  = _call_embeddings_with_split_on_error(client, model, inputs[:mid])
            right = _call_embeddings_with_split_on_error(client, model, inputs[mid:])
            return left + right
        raise

def _retryable_embed_batch(client: OpenAI, model: str, inputs: List[str], max_retries: int = 6) -> List[List[float]]:
    """
    Retry/backoff cho RateLimit/Timeout/5xx. Jitter để tránh thác yêu cầu.
    """
    base = 1.2
    for attempt in range(max_retries):
        try:
            return _call_embeddings_with_split_on_error(client, model, inputs)
        except (openai_root.RateLimitError,
                openai_root.APIConnectionError,
                openai_root.APITimeoutError) as e:
            sleep = base ** attempt + random.uniform(0, 0.6)
            print(f"[WARN] Embed retry ({type(e).__name__}) in {sleep:.1f}s (attempt {attempt+1}/{max_retries})")
            time.sleep(sleep)
            continue
        except openai_root.APIError as e:
            # 5xx → retry; 4xx (khác BadRequest đã xử lý ở trên) → raise
            status = getattr(e, "status_code", None) or getattr(e, "status", None)
            if status and int(status) >= 500:
                sleep = base ** attempt + random.uniform(0, 0.6)
                print(f"[WARN] Embed 5xx retry in {sleep:.1f}s")
                time.sleep(sleep); continue
            raise
    raise RuntimeError("Embeddings failed after retries")

def embed_texts_safe(client: OpenAI, texts: List[str], model: str) -> List[List[float]]:
    cut_texts = [truncate_tokens(t or "", MAX_EMBED_TOKENS) for t in texts]
    lengths = [tok_len(t) for t in cut_texts]
    out: List[List[float]] = []
    i = 0
    while i < len(cut_texts):
        total_tok = 0; cnt = 0; j = i
        while j < len(cut_texts) and cnt < BATCH_EMBED:
            need = lengths[j]
            if need > MAX_EMBED_TOTAL_TOKENS:
                if cnt == 0:
                    j = j + 1
                break
            if total_tok + need > MAX_EMBED_TOTAL_TOKENS:
                break
            total_tok += need; cnt += 1; j += 1
        if j == i:  # không nhét nổi → gửi đơn lẻ
            j = i + 1
        sub_inputs = cut_texts[i:j]
        vecs = _retryable_embed_batch(client, model, sub_inputs)
        out.extend(vecs)
        i = j
    return out

# -----------------------------
# Ingest incremental (+ modality split optional, reuse vectors)
# -----------------------------

def _validate_row(r: dict) -> Optional[str]:
    if not r.get("text"):
        return "missing text"
    md = r.get("metadata") or {}
    if not md.get("citation"):
        return "missing citation"
    return None

def ingest_chunks_to_chroma(
    chunks_jsonl: Path,
    collection_name: str,
) -> Tuple[int, int, int, Optional[int]]:
    assert chunks_jsonl.exists(), f"Không thấy file: {chunks_jsonl}"

    client_oai = get_openai_client()
    client_chroma = get_persistent_client()

    base_col = get_or_create_collection(client_chroma, collection_name)

    tables_col = images_col = None
    if SPLIT_BY_MODALITY:
        tables_col = get_or_create_collection(client_chroma, f"{collection_name}_tables")
        images_col = get_or_create_collection(client_chroma, f"{collection_name}_images")

    groups = group_chunks_by_source_sha1(chunks_jsonl)
    sha1_list = sorted(groups.keys())
    if not sha1_list:
        print(f"[INFO] Không có record hợp lệ trong {chunks_jsonl}")
        return (0, 0, 0, None)

    present_map = chroma_has_any(base_col, sha1_list)

    n_sources_total = len(sha1_list)
    n_sources_new = 0
    n_upserted = 0
    emb_dim: Optional[int] = None

    for idx, sha1 in enumerate(sha1_list, 1):
        if present_map.get(sha1):
            print(f"[SKIP] Source đã tồn tại (SHA1={sha1[:12]}...)")
            continue

        rows = groups[sha1]
        uniq: Dict[str, dict] = {}
        bad = 0
        for r in rows:
            err = _validate_row(r)
            if err:
                bad += 1
                continue
            cid = r.get("chunk_id") or hashlib.sha1((sha1 + "|" + (r.get("text") or "")).encode("utf-8")).hexdigest()[:16]
            uniq[cid] = r
        rows = list(uniq.values())
        if bad:
            print(f"[WARN] Bỏ {bad} dòng thiếu metadata/citation ở source {sha1[:12]}")

        ids: List[str] = []; docs: List[str] = []; metas: List[dict] = []; modalities: List[str] = []
        ids_table: List[str] = []; docs_table: List[str] = []; metas_table: List[dict] = []; vecs_table: List[List[float]] = []
        ids_image: List[str] = []; docs_image: List[str] = []; metas_image: List[dict] = []; vecs_image: List[List[float]] = []

        for r in rows:
            cid = r.get("chunk_id") or hashlib.sha1((sha1 + "|" + (r.get("text") or "")).encode("utf-8")).hexdigest()[:16]
            text = r.get("text") or ""
            md_raw = (r.get("metadata") or {})
            md = sanitize_metadata(md_raw)
            md["source_sha1"] = sha1
            if r.get("doc_id"): md["doc_id"] = r["doc_id"]
            modality = detect_modality(text, md_raw)
            md["modality"] = modality
            ids.append(str(cid)); docs.append(text); metas.append(md); modalities.append(modality)

        # Embed + upsert theo batch
        for i in range(0, len(ids), BATCH_EMBED):
            sub_ids, sub_docs, sub_metas, sub_mods = ids[i:i+BATCH_EMBED], docs[i:i+BATCH_EMBED], metas[i:i+BATCH_EMBED], modalities[i:i+BATCH_EMBED]
            vecs = embed_texts_safe(client_oai, sub_docs, EMBED_MODEL)
            if emb_dim is None and vecs:
                emb_dim = len(vecs[0])

            base_col.upsert(ids=sub_ids, embeddings=vecs, documents=sub_docs, metadatas=sub_metas)
            n_upserted += len(sub_ids)

            if SPLIT_BY_MODALITY:
                for k, _mod in enumerate(sub_mods):
                    if _mod == "table":
                        ids_table.append(sub_ids[k]); docs_table.append(sub_docs[k]); metas_table.append(sub_metas[k]); vecs_table.append(vecs[k])
                    elif _mod == "image":
                        ids_image.append(sub_ids[k]); docs_image.append(sub_docs[k]); metas_image.append(sub_metas[k]); vecs_image.append(vecs[k])

        if SPLIT_BY_MODALITY and tables_col and ids_table:
            for i in range(0, len(ids_table), 2048):
                tables_col.upsert(
                    ids=ids_table[i:i+2048],
                    embeddings=vecs_table[i:i+2048],
                    documents=docs_table[i:i+2048],
                    metadatas=metas_table[i:i+2048],
                )
        if SPLIT_BY_MODALITY and images_col and ids_image:
            for i in range(0, len(ids_image), 2048):
                images_col.upsert(
                    ids=ids_image[i:i+2048],
                    embeddings=vecs_image[i:i+2048],
                    documents=docs_image[i:i+2048],
                    metadatas=metas_image[i:i+2048],
                )

        n_sources_new += 1
        if n_upserted % PRINT_EVERY == 0:
            print(f"[PROGRESS] upserted={n_upserted} (sources processed={n_sources_new}/{n_sources_total})")

    print(f"✅ Indexed {n_sources_new}/{n_sources_total} sources → collection '{collection_name}' at '{PERSIST_DIR.name}'"
          + (f" | dim={emb_dim}" if emb_dim else ""))
    if SPLIT_BY_MODALITY:
        print(f"   (Đã tạo/bổ sung collection phụ: '{collection_name}_tables' và '{collection_name}_images')")

    return (n_sources_total, n_sources_new, n_upserted, emb_dim)

# -----------------------------
# Retrieve (demo) — ưu tiên bảng/hình
# -----------------------------

VALID_INCLUDE = {"documents", "metadatas", "distances"}
def _safe_include(items):
    if not items: return None
    return [x for x in items if x in VALID_INCLUDE] or None

def embed_query(client: OpenAI, q: str, model: str) -> List[float]:
    q = truncate_tokens(q, MAX_EMBED_TOKENS)
    return client.embeddings.create(model=model, input=[q]).data[0].embedding

def retrieve(
    query: str,
    collection_name: str,
    top_k: int = 5,
    prefer_modality: Optional[str] = None,   # None|"table"|"image"|"text"
    only_modality: bool = False,
) -> Dict[str, Any]:
    client_oai = get_openai_client()
    qvec = embed_query(client_oai, query, EMBED_MODEL)
    client_chroma = get_persistent_client()
    collection = client_chroma.get_or_create_collection(name=collection_name)

    where = None; where_document = None
    if prefer_modality in {"table", "image", "text"}:
        where = {"modality": {"$eq": prefer_modality}} if only_modality else {"$or": [{"modality": {"$eq": prefer_modality}}]}
        if prefer_modality == "table":
            where_document = {"$contains": "[BẢNG]"}  # full-text filter vào nội dung
        elif prefer_modality == "image":
            where_document = {"$contains": "[HÌNH]"}

    include_fields = _safe_include(["documents", "metadatas", "distances"])
    res = collection.query(
        query_embeddings=[qvec],
        n_results=top_k,
        where=where,
        where_document=where_document,
        include=include_fields,
    )

    out = []
    ids_ = res.get("ids", [[]])[0]
    docs_ = res.get("documents", [[]])[0] if res.get("documents") else []
    metas_ = res.get("metadatas", [[]])[0] if res.get("metadatas") else []
    dists_ = res.get("distances", [[]])[0] if res.get("distances") else []

    for i in range(max(len(ids_), len(docs_), len(metas_), len(dists_))):
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
            "source_sha1": (m or {}).get("source_sha1"),
            "modality": (m or {}).get("modality"),
            "has_table_html": (m or {}).get("has_table_html"),
            "has_image": (m or {}).get("has_image"),
        })
    return {"query": query, "top_k": top_k, "results": out}

# -----------------------------
# CLI
# -----------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Index incremental chunks vào Chroma (ưu tiên bảng/hình)")
    p.add_argument("--chunks", type=str, default=str(JSONL_CHUNKS), help="Đường dẫn pccc_chunks.jsonl")
    p.add_argument("--collection", type=str, default=COLLECTION, help="Tên collection Chroma")
    p.add_argument("--demo_query", type=str, default=os.getenv("DEMO_QUERY", ""), help="Truy vấn demo sau khi ingest")
    p.add_argument("--top_k", type=int, default=5)
    p.add_argument("--prefer_modality", type=str, choices=["table","image","text"], default=None)
    p.add_argument("--only_modality", action="store_true")
    return p.parse_args()

def main():
    args = parse_args()
    chunks_path = Path(args.chunks)
    assert chunks_path.exists(), f"Không thấy file: {chunks_path}"

    n_total, n_new, n_up, dim = ingest_chunks_to_chroma(
        chunks_jsonl=chunks_path,
        collection_name=args.collection,
    )

    if args.demo_query.strip():
        print("\n=== DEMO RESULTS ===")
        res = retrieve(args.demo_query, args.collection, top_k=args.top_k,
                       prefer_modality=args.prefer_modality, only_modality=args.only_modality)
        for r in res["results"]:
            label_parts = []
            if r.get("van_ban"): label_parts.append(r["van_ban"])
            segs = []
            if r.get("dieu"): segs.append(f"Điều {r['dieu']}")
            if r.get("khoan"): segs.append(f"Khoản {r['khoan']}")
            if r.get("diem"): segs.append(f"Điểm {r['diem']}")
            label = " — ".join([label_parts[0], ", ".join(segs)]) if (label_parts and segs) else (label_parts[0] if label_parts else ", ".join(segs))
            dist = r.get("distance"); mod = r.get("modality") or "text"
            print(f"- [{mod.upper()}] {label or 'Văn bản'} | dist={dist:.4f}" if dist is not None else f"- [{mod.UPPER()}] {label or 'Văn bản'}")
            txt = (r.get("text") or "").strip().replace("\n", " ")
            print("  →", (txt[:240] + ("..." if len(txt) > 240 else "")))
        print()

if __name__ == "__main__":
    main()

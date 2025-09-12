#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Index incremental chunks (DOC/DOCX/PDF/PPTX) vào Chroma bằng OpenAI Embeddings
+ XÂY DỰNG CHỈ MỤC LEXICAL (FTS5) để hỗ trợ Hybrid search (BM25 + dense).

Tính năng:
- Batching theo ngân sách token + fallback chia đôi khi dính 'max_tokens_per_request'.
- Retry/backoff cho RateLimit/Timeout/5xx.
- Sanitize metadata an toàn cho Chroma (str/int/float/bool; list/dict -> JSON).
- Phát hiện modality (table/image/text) & upsert song song (tuỳ chọn) collection phụ.
- Xây FTS5 (SQLite) incremental: bảng 'chunks' + virtual table 'chunks_fts'.
- Demo Hybrid search: BM25 (FTS5) + Dense (Chroma) hợp nhất bằng RRF + trọng số.

ENV:
  OPENAI_API_KEY
  CHROMA_DIR=./chroma_pccc_text_embedding_3_small
  CHROMA_COLLECTION=pccc_vi_text_embedding_3_small
  EMBED_MODEL=text-embedding-3-small

  # Ingest
  PCCC_CHUNKS=./pccc_chunks.jsonl
  BATCH_EMBED=96
  MAX_EMBED_TOTAL_TOKENS=280000
  MAX_EMBED_TOKENS=8000
  PRINT_EVERY=1000
  PCCC_SCHEMA_VERSION=pccc_chunks/v1.2

  # Modality split (tuỳ chọn)
  SPLIT_BY_MODALITY=0

  # FTS (lexical index)
  BUILD_LEXICAL=1
  FTS_DB=<mặc định: {CHROMA_DIR}/lexical/{COLLECTION}_fts.sqlite>

  # Hybrid demo
  HYBRID_TOPK_LEX=50
  HYBRID_TOPK_VEC=30
  HYBRID_FINAL_K=12
  HYBRID_RRF_K=60
  W_LEX=0.45
  W_VEC=0.55
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Any
import argparse, json, os, sys, hashlib, time, math, random, sqlite3

import chromadb
import openai as openai_root
from openai import OpenAI, BadRequestError

# -----------------------------
# Cấu hình & biến môi trường
# -----------------------------

PERSIST_DIR = Path(os.getenv("CHROMA_DIR", "./chroma_pccc_text_embedding_3_small"))
COLLECTION  = os.getenv("CHROMA_COLLECTION", "pccc_vi_text_embedding_3_small")
JSONL_CHUNKS = Path(os.getenv("PCCC_CHUNKS", "./pccc_chunks.jsonl"))
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")

BATCH_EMBED = int(os.getenv("BATCH_EMBED", "96"))
MAX_EMBED_TOTAL_TOKENS = int(os.getenv("MAX_EMBED_TOTAL_TOKENS", "280000"))
MAX_EMBED_TOKENS = int(os.getenv("MAX_EMBED_TOKENS", "8000"))
PRINT_EVERY = int(os.getenv("PRINT_EVERY", "1000"))
SCHEMA_VERSION = os.getenv("PCCC_SCHEMA_VERSION", "pccc_chunks/v1.2")

SPLIT_BY_MODALITY = os.getenv("SPLIT_BY_MODALITY", "0") == "1"

# FTS (lexical)
BUILD_LEXICAL = os.getenv("BUILD_LEXICAL", "1") == "1"
_FTS_DEFAULT_DIR = PERSIST_DIR / "lexical"
FTS_DB = os.getenv("FTS_DB", str(_FTS_DEFAULT_DIR / f"{COLLECTION}_fts.sqlite"))

# Hybrid demo params
HYBRID_TOPK_LEX = int(os.getenv("HYBRID_TOPK_LEX", "50"))
HYBRID_TOPK_VEC = int(os.getenv("HYBRID_TOPK_VEC", "30"))
HYBRID_FINAL_K  = int(os.getenv("HYBRID_FINAL_K", "12"))
HYBRID_RRF_K    = int(os.getenv("HYBRID_RRF_K", "60"))
W_LEX = float(os.getenv("W_LEX", "0.45"))
W_VEC = float(os.getenv("W_VEC", "0.55"))

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
    - Bỏ None, NaN/Inf
    - Giữ str/int/float/bool
    - list/dict/... -> json.dumps(...)
    """
    if not md:
        return {"schema_version": SCHEMA_VERSION}

    out = {}
    for k, v in md.items():
        key = str(k)
        if v is None:
            continue
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            continue
        if isinstance(v, (str, int, float, bool)):
            out[key] = v
            continue
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
# Embedding helpers (batch + retry/backoff)
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
                raise
            mid = len(inputs) // 2
            left  = _call_embeddings_with_split_on_error(client, model, inputs[:mid])
            right = _call_embeddings_with_split_on_error(client, model, inputs[mid:])
            return left + right
        raise

def _retryable_embed_batch(client: OpenAI, model: str, inputs: List[str], max_retries: int = 6) -> List[List[float]]:
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
        if j == i:
            j = i + 1
        sub_inputs = cut_texts[i:j]
        vecs = _retryable_embed_batch(client, model, sub_inputs)
        out.extend(vecs)
        i = j
    return out

# -----------------------------
# FTS (SQLite) — lexical index
# -----------------------------

def _fts_connect() -> sqlite3.Connection:
    Path(FTS_DB).parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(FTS_DB)
    con.execute("PRAGMA journal_mode=WAL;")
    con.execute("PRAGMA synchronous=NORMAL;")
    con.execute("PRAGMA temp_store=MEMORY;")
    return con

def _fts_init_schema(con: sqlite3.Connection):
    con.executescript("""
    CREATE TABLE IF NOT EXISTS chunks (
        id TEXT PRIMARY KEY,
        text TEXT NOT NULL,
        doc_id TEXT,
        source_sha1 TEXT,
        modality TEXT,
        van_ban TEXT,
        dieu TEXT,
        khoan TEXT,
        diem TEXT,
        parent_id TEXT,
        ancestor_dieu_id TEXT
    );
    CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts
        USING fts5(text, content='chunks', content_rowid='rowid', tokenize='unicode61');
    CREATE INDEX IF NOT EXISTS idx_chunks_sha1 ON chunks(source_sha1);
    CREATE INDEX IF NOT EXISTS idx_chunks_parent ON chunks(parent_id);
    CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
        INSERT INTO chunks_fts(rowid, text) VALUES (new.rowid, new.text);
    END;
    CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
        INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES ('delete', old.rowid, old.text);
    END;
    CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
        INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES ('delete', old.rowid, old.text);
        INSERT INTO chunks_fts(rowid, text) VALUES (new.rowid, new.text);
    END;
    """)
    con.commit()

def fts_upsert_rows(rows: List[dict]):
    """rows: mỗi item gồm {id,text,doc_id,source_sha1,modality,van_ban,dieu,khoan,diem,parent_id,ancestor_dieu_id}"""
    if not BUILD_LEXICAL or not rows:
        return
    con = _fts_connect()
    try:
        _fts_init_schema(con)
        cur = con.cursor()
        cur.execute("BEGIN;")
        cur.executemany("""
            INSERT INTO chunks(id,text,doc_id,source_sha1,modality,van_ban,dieu,khoan,diem,parent_id,ancestor_dieu_id)
            VALUES(?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(id) DO UPDATE SET
                text=excluded.text,
                doc_id=excluded.doc_id,
                source_sha1=excluded.source_sha1,
                modality=excluded.modality,
                van_ban=excluded.van_ban,
                dieu=excluded.dieu,
                khoan=excluded.khoan,
                diem=excluded.diem,
                parent_id=excluded.parent_id,
                ancestor_dieu_id=excluded.ancestor_dieu_id;
        """, [
            (
                r.get("id"),
                r.get("text") or "",
                r.get("doc_id"),
                r.get("source_sha1"),
                r.get("modality"),
                r.get("van_ban"),
                r.get("dieu"),
                r.get("khoan"),
                r.get("diem"),
                r.get("parent_id"),
                r.get("ancestor_dieu_id"),
            )
            for r in rows
        ])
        con.commit()
    finally:
        con.close()

def fts_search(query: str, topk: int) -> List[Tuple[str, float]]:
    """
    Trả về [(id, score_lex)], score_lex đã chuẩn hóa về miền cao-là-tốt.
    FTS5 bm25() -> điểm thấp là tốt. Ta dùng: score = 1 / (1 + bm25)
    """
    if not BUILD_LEXICAL:
        return []
    con = _fts_connect()
    try:
        _fts_init_schema(con)
        cur = con.cursor()
        # match: dùng FTS5 unicode61; câu hỏi có dấu → ok.
        cur.execute(f"""
            SELECT c.id, bm25(chunks_fts) AS rank
            FROM chunks_fts
            JOIN chunks c ON c.rowid = chunks_fts.rowid
            WHERE chunks_fts MATCH ?
            ORDER BY rank ASC
            LIMIT ?;
        """, (query, int(topk)))
        rows = cur.fetchall()
        out = []
        for cid, bm in rows:
            try:
                bm = float(bm)
            except Exception:
                bm = 100.0
            out.append((cid, 1.0/(1.0 + bm)))
        return out
    finally:
        con.close()

# -----------------------------
# Ingest incremental (+ modality split, + FTS upsert)
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

    # chuẩn bị FTS schema sớm (nếu bật)
    if BUILD_LEXICAL:
        con = _fts_connect()
        try:
            _fts_init_schema(con)
        finally:
            con.close()

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

        # thu thập dữ liệu cho FTS
        fts_rows: List[dict] = []

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

            if BUILD_LEXICAL:
                fts_rows.append({
                    "id": str(cid),
                    "text": text,
                    "doc_id": r.get("doc_id"),
                    "source_sha1": sha1,
                    "modality": modality,
                    "van_ban": md_raw.get("van_ban") or md.get("van_ban"),
                    "dieu": (md_raw.get("dieu") or md.get("dieu")),
                    "khoan": (md_raw.get("khoan") or md.get("khoan")),
                    "diem": (md_raw.get("diem") or md.get("diem")),
                    "parent_id": (md_raw.get("parent_id") or md.get("parent_id")),
                    "ancestor_dieu_id": (md_raw.get("ancestor_dieu_id") or md.get("ancestor_dieu_id")),
                })

        # Upsert vào FTS trước (để demo hybrid có thể chạy ngay cả khi embedding đợi)
        if BUILD_LEXICAL and fts_rows:
            fts_upsert_rows(fts_rows)

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

        if SPLIT_BY_MODALITY and ids_table:
            tables_col = get_or_create_collection(client_chroma, f"{collection_name}_tables")
            for i in range(0, len(ids_table), 2048):
                tables_col.upsert(
                    ids=ids_table[i:i+2048],
                    embeddings=vecs_table[i:i+2048],
                    documents=docs_table[i:i+2048],
                    metadatas=metas_table[i:i+2048],
                )
        if SPLIT_BY_MODALITY and ids_image:
            images_col = get_or_create_collection(client_chroma, f"{collection_name}_images")
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
    if BUILD_LEXICAL:
        print(f"   (Đã xây FTS lexical tại: {FTS_DB})")

    return (n_sources_total, n_sources_new, n_upserted, emb_dim)

# -----------------------------
# Retrieve utils (dense + get by ids)
# -----------------------------

_VALID_INCLUDE = {"documents", "metadatas", "distances"}
def _safe_include(items):
    if not items: return None
    return [x for x in items if x in _VALID_INCLUDE] or None

def embed_query(client: OpenAI, q: str, model: str) -> List[float]:
    q = truncate_tokens(q, MAX_EMBED_TOKENS)
    return client.embeddings.create(model=model, input=[q]).data[0].embedding

def _normalize_distance_to_similarity(distances: List[Optional[float]]) -> List[float]:
    vals = [d for d in distances if d is not None]
    if not vals:
        return [0.0] * len(distances)
    mn, mx = min(vals), max(vals)
    if mx <= mn + 1e-12:
        return [1.0] * len(distances)
    return [1.0 - ((d - mn) / (mx - mn)) if d is not None else 0.0 for d in distances]

def chroma_query_vec(query: str, collection_name: str, top_k: int) -> List[dict]:
    client_oai = get_openai_client()
    qvec = embed_query(client_oai, query, EMBED_MODEL)
    client_chroma = get_persistent_client()
    col = client_chroma.get_or_create_collection(name=collection_name)
    res = col.query(
        query_embeddings=[qvec],
        n_results=top_k,
        include=_safe_include(["documents","metadatas","distances"]),
    )
    ids_ = res.get("ids", [[]])[0]
    docs_ = res.get("documents", [[]])[0] if res.get("documents") else []
    metas_ = res.get("metadatas", [[]])[0] if res.get("metadatas") else []
    dists_ = res.get("distances", [[]])[0] if res.get("distances") else []
    sims = _normalize_distance_to_similarity(dists_)
    out = []
    for i in range(max(len(ids_), len(docs_), len(metas_), len(dists_))):
        m = metas_[i] if i < len(metas_) else {}
        out.append({
            "id": ids_[i] if i < len(ids_) else None,
            "text": docs_[i] if i < len(docs_) else "",
            "metadata": (m or {}),
            "sim_vec": sims[i] if i < len(sims) else 0.0,
            "distance": dists_[i] if i < len(dists_) else None,
        })
    return out

def chroma_get_by_ids(collection_name: str, ids: List[str]) -> Dict[str, dict]:
    if not ids:
        return {}
    client = get_persistent_client()
    col = client.get_or_create_collection(name=collection_name)
    res = col.get(ids=ids, include=_safe_include(["documents","metadatas"]))
    out = {}
    for i, cid in enumerate(res.get("ids", [])):
        out[cid] = {
            "id": cid,
            "text": (res.get("documents") or [""])[i],
            "metadata": (res.get("metadatas") or [{}])[i],
        }
    return out

# -----------------------------
# Hybrid search (demo): FTS (BM25) + Dense (Chroma) + RRF
# -----------------------------

def _rrf_scores(ids: List[str], base: int = HYBRID_RRF_K) -> Dict[str, float]:
    return {cid: 1.0 / (base + rank) for rank, cid in enumerate(ids, start=1)}

def hybrid_retrieve(
    query: str,
    collection_name: str,
    topk_lex: int = HYBRID_TOPK_LEX,
    topk_vec: int = HYBRID_TOPK_VEC,
    final_k: int = HYBRID_FINAL_K,
) -> List[dict]:
    # 1) Lexical (FTS5)
    lex_hits = fts_search(query, topk_lex)  # [(id, score_lex)]
    lex_ids = [cid for cid, _ in lex_hits]
    # Lấy nội dung/metadata cho lexical ids từ Chroma (để thống nhất output)
    lex_docs = chroma_get_by_ids(collection_name, lex_ids)
    # Bổ sung điểm RRF cho lexical
    rrf_lex = _rrf_scores(lex_ids, base=HYBRID_RRF_K)
    lex_pack = []
    for cid, s_lex in lex_hits:
        obj = lex_docs.get(cid, {"id": cid, "text": "", "metadata": {}})
        obj["score_lex"] = float(s_lex)
        obj["rrf_lex"] = rrf_lex.get(cid, 0.0)
        lex_pack.append(obj)

    # 2) Dense (Chroma)
    vec_hits = chroma_query_vec(query, collection_name, topk_vec)  # [{"id","text","metadata","sim_vec"}]
    vec_ids = [h["id"] for h in vec_hits if h.get("id")]
    rrf_vec = _rrf_scores(vec_ids, base=HYBRID_RRF_K)
    for h in vec_hits:
        h["score_vec"] = float(h.get("sim_vec", 0.0))
        h["rrf_vec"] = rrf_vec.get(h["id"], 0.0)

    # 3) Hợp nhất theo id
    pool: Dict[str, dict] = {}
    for r in lex_pack:
        cid = r["id"]; pool[cid] = r
    for r in vec_hits:
        cid = r["id"]
        if cid in pool:
            # gộp điểm & giữ text/metadata từ vec (đủ hơn)
            pool[cid]["text"] = r.get("text") or pool[cid].get("text") or ""
            pool[cid]["metadata"] = r.get("metadata") or pool[cid].get("metadata") or {}
            pool[cid]["score_vec"] = r.get("score_vec", 0.0)
            pool[cid]["rrf_vec"]   = r.get("rrf_vec", 0.0)
        else:
            pool[cid] = r

    # 4) Tính fused score
    out = []
    for cid, it in pool.items():
        s_lex = float(it.get("score_lex", 0.0))
        s_vec = float(it.get("score_vec", 0.0))
        rrf = float(it.get("rrf_lex", 0.0)) + float(it.get("rrf_vec", 0.0))
        fused = W_LEX*s_lex + W_VEC*s_vec + rrf
        it["score_hybrid"] = fused
        out.append(it)

    out.sort(key=lambda x: x.get("score_hybrid", 0.0), reverse=True)
    return out[:final_k]

# -----------------------------
# CLI
# -----------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Index chunks vào Chroma + xây FTS (lexical) + demo Hybrid search")
    p.add_argument("--chunks", type=str, default=str(JSONL_CHUNKS), help="Đường dẫn pccc_chunks.jsonl")
    p.add_argument("--collection", type=str, default=COLLECTION, help="Tên collection Chroma")
    p.add_argument("--demo_query", type=str, default=os.getenv("DEMO_QUERY", ""), help="Truy vấn demo sau khi ingest")
    p.add_argument("--topk_lex", type=int, default=HYBRID_TOPK_LEX)
    p.add_argument("--topk_vec", type=int, default=HYBRID_TOPK_VEC)
    p.add_argument("--final_k", type=int, default=HYBRID_FINAL_K)
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
        print("\n=== HYBRID DEMO RESULTS (BM25 + Dense + RRF) ===")
        results = hybrid_retrieve(
            query=args.demo_query.strip(),
            collection_name=args.collection,
            topk_lex=args.topk_lex,
            topk_vec=args.topk_vec,
            final_k=args.final_k,
        )
        for i, r in enumerate(results, 1):
            md = r.get("metadata") or {}
            segs = []
            if md.get("van_ban"): segs.append(md["van_ban"])
            parts = []
            if md.get("dieu"):  parts.append(f"Điều {md['dieu']}")
            if md.get("khoan"): parts.append(f"Khoản {md['khoan']}")
            if md.get("diem"):  parts.append(f"Điểm {md['diem']}")
            label = (segs[0] + (" — " + ", ".join(parts) if parts else "")) if segs else ", ".join(parts)
            print(f"{i:>2}. {label or 'Văn bản'} | hybrid={r.get('score_hybrid',0):.4f} (lex={r.get('score_lex',0):.3f}, vec={r.get('score_vec',0):.3f})")
            txt = (r.get("text") or "").strip().replace("\n", " ")
            print("    →", (txt[:240] + ("..." if len(txt) > 240 else "")))
        print()

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
QA Answerer PCCC (VN) — HYBRID RETRIEVAL: lexical (BM25) + dense (emb) → RRF → parent-scope expand → reranker → answer
- Nếu câu hỏi về BẢNG: in bảng Markdown trước, rồi giải thích & trích dẫn.
- Ưu tiên văn bản mới; giảm trùng; chống lẫn Khoản/Điểm giữa các Điều bằng Parent-Document expansion + scope prior.
- Agent chuẩn hoá prompt & chống lệch chủ đề (prompt_guard).

ENV chính:
  OPENAI_API_KEY
  CHROMA_DIR=./chroma_pccc_text_embedding_3_small
  CHROMA_COLLECTION=pccc_vi_text_embedding_3_small
  EMBED_MODEL=text-embedding-3-small
  CHAT_MODEL=gpt-4o-mini

  # Hybrid search
  HYBRID_SEARCH=1
  LEXICAL_JSONL=./pccc_chunks.jsonl
  TOPK_LEXICAL=50
  TOPK_DENSE=20
  RRF_K=60

  # Parent expansion
  PARENT_EXPAND=1
  PARENT_TOP_GROUPS=3
  PARENT_LIMIT_PER_GROUP=12

  # Retrieve options
  PREFER_MODALITY=auto
  ONLY_MODALITY=0
  STRICT_TABLE_RETRIEVE=0

  # Rerank
  RERANK_BACKEND=cross_encoder
  RERANK_MODEL=BAAI/bge-reranker-v2-m3
  RERANK_MAX_LEN=512
  TOPK_RERANK=30
  FINAL_K=8
  W_CE=0.64 W_VEC=0.18 W_MOD=0.08 W_REC=0.07 W_COH=0.03
  # Priors mở rộng (lexical/scope/money)
  W_LEX=0.06 W_SCOPE=0.06 W_MONEY=0.04

  # Render bảng
  TABLE_RENDER_MAX_ROWS=30
  TABLE_RENDER_MAX_TABLES=2
  TABLE_RENDER_MAX_CHARS=12000

  # Prompt guard
  PROMPT_GUARD=1
  PROMPT_GUARD_USE_LLM=0
"""

from __future__ import annotations
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import os, re, json, argparse, unicodedata, threading

import chromadb
from openai import OpenAI
from prompt_guard import guard_prompt

# ================== Embedding tokenizer ==================
try:
    import tiktoken
    ENC = tiktoken.get_encoding("cl100k_base")
except Exception:
    ENC = None

def tok_len(s: str) -> int:
    if ENC is None: return max(1, len(s)//2)
    return len(ENC.encode(s or ""))

def truncate_tokens(s: str, max_tokens: int) -> str:
    if tok_len(s) <= max_tokens: return s
    if ENC is None: return s[: max_tokens*2]
    ids = ENC.encode(s or "")
    return ENC.decode(ids[:max_tokens])

# ================== OpenAI ==================
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
CHAT_MODEL  = os.getenv("CHAT_MODEL", "gpt-4o-mini")

def get_openai_client() -> OpenAI:
    key = os.getenv("OPENAI_API_KEY", "").strip()
    assert key, "Thiếu OPENAI_API_KEY"
    return OpenAI(api_key=key)

def embed_query(client: OpenAI, q: str) -> List[float]:
    q = truncate_tokens(q, int(os.getenv("MAX_EMBED_TOKENS", "8000")))
    return client.embeddings.create(model=EMBED_MODEL, input=[q]).data[0].embedding

# ================== Chroma ==================
PERSIST_DIR   = Path(os.getenv("CHROMA_DIR", "./chroma_pccc_text_embedding_3_small"))
COLLECTION    = os.getenv("CHROMA_COLLECTION", "pccc_vi_text_embedding_3_small")
VALID_INCLUDE = {"documents","metadatas","distances","embeddings","uris","data"}

def get_chroma_collection(name: str):
    client = chromadb.PersistentClient(path=str(PERSIST_DIR))
    return client.get_or_create_collection(name=name)

def _safe_include(items: List[str]):
    return [x for x in items if x in VALID_INCLUDE] or None

# ================== Heuristics modality ==================
_TABLE_PAT = re.compile(r"\b(bảng|bang|table|biểu|biểu)\b", re.I | re.U)
_IMAGE_PAT = re.compile(r"\b(hình|hinh|ảnh|anh|figure|sơ đồ|so do|biểu đồ|biểu đồ)\b", re.I | re.U)

def infer_query_modality(query: str) -> Optional[str]:
    if _TABLE_PAT.search(query or ""): return "table"
    if _IMAGE_PAT.search(query or ""): return "image"
    return None

def _infer_modality_from_text(txt: str) -> str:
    t = (txt or "").lstrip()
    if t.startswith("[BẢNG]"): return "table"
    if t.startswith("[HÌNH]"): return "image"
    return "text"

# ================== DENSE retrieve (Chroma) ==================
def retrieve_dense(
    query: str,
    collection_name: str,
    top_k: int = 20,
    prefer_modality: Optional[str] = None,   # None|"table"|"image"|"text"
    only_modality: bool = False,
) -> List[Dict[str,Any]]:
    col = get_chroma_collection(collection_name)
    oai = get_openai_client()
    qvec = embed_query(oai, query)

    where = None
    where_document = None
    if prefer_modality in {"table", "image"} and only_modality:
        if prefer_modality == "table":
            where_document = {"$contains": "[BẢNG]"}
        elif prefer_modality == "image":
            where_document = {"$contains": "[HÌNH]"}
    # prefer_modality="text" → không lọc cứng, để rerank xử lý.

    res = col.query(
        query_embeddings=[qvec],
        n_results=top_k,
        where=where,
        where_document=where_document,
        include=_safe_include(["documents","metadatas","distances"]),
    )

    out: List[Dict[str,Any]] = []
    ids_ = res.get("ids", [[]])[0]
    docs_ = res.get("documents", [[]])[0] if res.get("documents") else []
    metas_ = res.get("metadatas", [[]])[0] if res.get("metadatas") else []
    dists_ = res.get("distances", [[]])[0] if res.get("distances") else []

    for i in range(max(len(ids_), len(docs_), len(metas_), len(dists_))):
        doc_text = docs_[i] if i < len(docs_) else ""
        m = metas_[i] if i < len(metas_) else {}
        modality = (m or {}).get("modality") or _infer_modality_from_text(doc_text)
        out.append({
            "id": ids_[i] if i < len(ids_) else None,
            "distance": dists_[i] if i < len(dists_) else None,
            "text": doc_text,
            "citation": (m or {}).get("citation") or "",
            "doc_id": (m or {}).get("doc_id") or "",
            "van_ban": (m or {}).get("van_ban") or "",
            "dieu": (m or {}).get("dieu"),
            "khoan": (m or {}).get("khoan"),
            "diem": (m or {}).get("diem"),
            "page_start": (m or {}).get("page_start"),
            "page_end": (m or {}).get("page_end"),
            "source_sha1": (m or {}).get("source_sha1"),
            "modality": modality,
            "has_table_html": (m or {}).get("has_table_html"),
            "table_html_excerpt": (m or {}).get("table_html_excerpt"),
            "has_image": (m or {}).get("has_image"),
            # recency:
            "effective_date": (m or {}).get("effective_date") or (m or {}).get("ngay_hieu_luc"),
            "ban_hanh_date": (m or {}).get("ban_hanh_date") or (m or {}).get("ngay_ban_hanh"),
            "updated_at": (m or {}).get("updated_at"),
            # parent scope (nếu đã index)
            "ancestor_dieu_id": (m or {}).get("ancestor_dieu_id"),
        })
    return out

# ================== LEXICAL retrieve (BM25 over JSONL) ==================
try:
    from rank_bm25 import BM25Okapi
except Exception:
    BM25Okapi = None  # fallback dense-only

_WORD_RE = re.compile(r"[0-9A-Za-zÀ-ỹ]+", re.UNICODE)

def _strip_accents(s: str) -> str:
    if not s: return ""
    import unicodedata as _ud
    s = _ud.normalize("NFD", s)
    s = "".join(ch for ch in s if _ud.category(ch) != "Mn")
    return _ud.normalize("NFKC", s)

def _tokenize_vi(s: str) -> List[str]:
    s = _strip_accents((s or "").lower())
    return _WORD_RE.findall(s)

class _LexicalIndex:
    def __init__(self, jsonl_path: Path):
        self.path = jsonl_path
        self.mtime = 0.0
        self.docs: List[str] = []
        self.tokens: List[List[str]] = []
        self.meta: List[Dict[str,Any]] = []
        self.bm25: Optional[BM25Okapi] = None
        self._ensure_loaded(force=True)

    def _ensure_loaded(self, force: bool = False):
        if not self.path.exists():
            return
        mt = self.path.stat().st_mtime
        if (not force) and mt <= self.mtime:
            return
        docs, toks, meta = [], [], []
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line: continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                txt = (obj.get("text") or "").strip()
                if not txt: continue
                md  = obj.get("metadata") or {}
                doc_id = obj.get("doc_id") or md.get("doc_id") or ""
                cid = obj.get("chunk_id") or obj.get("id") or md.get("id") or None
                modality = (md.get("modality") or ("table" if txt.lstrip().startswith("[BẢNG]") else ("image" if txt.lstrip().startswith("[HÌNH]") else "text")))
                record = {
                    "id": cid,
                    "distance": None,
                    "text": txt,
                    "citation": md.get("citation") or "",
                    "doc_id": doc_id,
                    "van_ban": md.get("van_ban"),
                    "dieu": md.get("dieu"),
                    "khoan": md.get("khoan"),
                    "diem": md.get("diem"),
                    "page_start": md.get("page_start"),
                    "page_end": md.get("page_end"),
                    "source_sha1": obj.get("source_sha1") or md.get("source_sha1"),
                    "modality": modality,
                    "has_table_html": md.get("has_table_html"),
                    "table_html_excerpt": md.get("table_html_excerpt"),
                    "has_image": md.get("has_image"),
                    "effective_date": md.get("effective_date") or md.get("ngay_hieu_luc"),
                    "ban_hanh_date": md.get("ban_hanh_date") or md.get("ngay_ban_hanh"),
                    "updated_at": md.get("updated_at"),
                    "ancestor_dieu_id": md.get("ancestor_dieu_id"),
                }
                docs.append(txt); toks.append(_tokenize_vi(txt)); meta.append(record)
        if BM25Okapi is None:
            self.bm25 = None
        else:
            self.bm25 = BM25Okapi(toks) if toks else None
        self.docs, self.tokens, self.meta = docs, toks, meta
        self.mtime = mt

    def search(self, query: str, top_k: int = 50,
               prefer_modality: Optional[str] = None,
               only_modality: bool = False) -> List[Dict[str,Any]]:
        self._ensure_loaded()
        if not self.docs:
            return []
        q_tokens = _tokenize_vi(query)
        scores = self.bm25.get_scores(q_tokens) if self.bm25 is not None else [0.0]*len(self.docs)
        idxs = list(range(len(self.docs)))
        idxs.sort(key=lambda i: scores[i], reverse=True)
        results: List[Dict[str,Any]] = []
        for i in idxs[:top_k]:
            item = dict(self.meta[i])
            item["bm25"] = float(scores[i])
            results.append(item)
        if prefer_modality in {"table","image"} and only_modality:
            tag = "table" if prefer_modality == "table" else "image"
            results = [r for r in results if (r.get("modality") == tag)]
        return results

_LEX_IDX: Optional[_LexicalIndex] = None
def get_lexical_index() -> Optional[_LexicalIndex]:
    global _LEX_IDX
    path = Path(os.getenv("LEXICAL_JSONL", "./pccc_chunks.jsonl"))
    if _LEX_IDX is None or (_LEX_IDX.path != path):
        _LEX_IDX = _LexicalIndex(path)
    return _LEX_IDX

def retrieve_lexical(query: str, top_k: int, prefer_modality: Optional[str], only_modality: bool) -> List[Dict[str,Any]]:
    idx = get_lexical_index()
    if idx is None or not (idx.docs and idx.meta):
        return []
    return idx.search(query, top_k=top_k, prefer_modality=prefer_modality, only_modality=only_modality)

# ================== RRF fusion (hybrid) ==================
def rrf_fuse(dense: List[Dict[str,Any]], lexical: List[Dict[str,Any]], k: int = 60, limit: int = 50) -> List[Dict[str,Any]]:
    def _key(c: Dict[str,Any]) -> str:
        cid = c.get("id")
        if cid: return str(cid)
        sha = c.get("source_sha1") or ""
        tx  = (c.get("text") or "")[:64]
        return f"{sha}|{hash(tx)}"

    r_dense  = { _key(c): r+1 for r, c in enumerate(dense) }
    r_lex    = { _key(c): r+1 for r, c in enumerate(lexical) }
    keys = set(r_dense) | set(r_lex)

    fused: Dict[str, Dict[str,Any]] = {}
    for key in keys:
        base = next((c for c in dense if _key(c) == key), None) or next((c for c in lexical if _key(c) == key), None)
        fused[key] = dict(base)
        score = 0.0
        if key in r_dense: score += 1.0 / (k + r_dense[key])
        if key in r_lex:   score += 1.0 / (k + r_lex[key])
        fused[key]["score_rrf"] = score
        if key in r_lex:
            fused[key]["bm25"] = fused[key].get("bm25", 0.0)

    def _dense_sim(c: Dict[str,Any]) -> float:
        d = c.get("distance")
        return -float(d) if d is not None else 0.0

    all_items = list(fused.values())
    all_items.sort(key=lambda x: (x.get("score_rrf", 0.0),
                                  (1 if (x.get("bm25") is not None and x.get("distance") is not None) else 0),
                                  _dense_sim(x)), reverse=True)
    return all_items[:limit]

# ================== HYBRID retrieve orchestrator ==================
def retrieve_hybrid(
    query: str,
    collection_name: str,
    topk_dense: int,
    topk_lex: int,
    prefer_modality: Optional[str],
    only_modality: bool,
    final_limit: int,
) -> List[Dict[str,Any]]:
    dense_res: List[Dict[str,Any]] = []
    lex_res:   List[Dict[str,Any]] = []

    def _run_dense():
        nonlocal dense_res
        dense_res = retrieve_dense(query, collection_name, top_k=topk_dense, prefer_modality=prefer_modality, only_modality=only_modality)

    def _run_lex():
        nonlocal lex_res
        lex_res = retrieve_lexical(query, top_k=topk_lex, prefer_modality=prefer_modality, only_modality=only_modality)

    td = threading.Thread(target=_run_dense)
    tl = threading.Thread(target=_run_lex)
    td.start(); tl.start(); td.join(); tl.join()

    k_rrf = int(os.getenv("RRF_K", "60"))
    fused = rrf_fuse(dense_res, lex_res, k=k_rrf, limit=max(final_limit, max(topk_dense, topk_lex)))
    return fused

# ================== Parent-scope EXPANSION (Parent Document Retriever) ==================
def _make_item_from_get(doc: str, md: dict, cid: str) -> Dict[str,Any]:
    modality = (md or {}).get("modality")
    if not modality:
        modality = "table" if str(doc).lstrip().startswith("[BẢNG]") else ("image" if str(doc).lstrip().startswith("[HÌNH]") else "text")
    return {
        "id": cid,
        "distance": None,
        "text": doc or "",
        "citation": (md or {}).get("citation") or "",
        "doc_id": (md or {}).get("doc_id") or "",
        "van_ban": (md or {}).get("van_ban") or "",
        "dieu": (md or {}).get("dieu"),
        "khoan": (md or {}).get("khoan"),
        "diem": (md or {}).get("diem"),
        "page_start": (md or {}).get("page_start"),
        "page_end": (md or {}).get("page_end"),
        "source_sha1": (md or {}).get("source_sha1"),
        "modality": modality,
        "has_table_html": (md or {}).get("has_table_html"),
        "table_html_excerpt": (md or {}).get("table_html_excerpt"),
        "has_image": (md or {}).get("has_image"),
        "effective_date": (md or {}).get("effective_date") or (md or {}).get("ngay_hieu_luc"),
        "ban_hanh_date": (md or {}).get("ban_hanh_date") or (md or {}).get("ngay_ban_hanh"),
        "updated_at": (md or {}).get("updated_at"),
        "ancestor_dieu_id": (md or {}).get("ancestor_dieu_id"),
        "bm25": None,  # không có trong nhánh get()
    }

def expand_by_parent_scope(pool: List[Dict[str,Any]], collection_name: str,
                           top_groups: int = 3, limit_per_group: int = 12, max_extra: int = 36) -> List[Dict[str,Any]]:
    """
    Lấy thêm các chunk cùng 'Điều' (ancestor_dieu_id hoặc dieu) trong cùng doc_id để giữ ngữ cảnh pháp lý nhất quán.
    """
    if not pool:
        return pool
    try:
        col = get_chroma_collection(collection_name)
    except Exception:
        return pool

    # Gom nhóm theo (doc_id, ancestor_dieu_id or dieu)
    def gkey(c):
        return (c.get("doc_id"), c.get("ancestor_dieu_id") or ("DIEU", c.get("dieu")))
    groups: Dict[Tuple[str,Any], List[Dict[str,Any]]] = {}
    for c in pool:
        groups.setdefault(gkey(c), []).append(c)

    # Chọn N nhóm mạnh nhất theo tổng score_rrf/bm25 (nếu có)
    def gscore(items: List[Dict[str,Any]]) -> float:
        s = 0.0
        for it in items:
            s += float(it.get("score_rrf") or 0.0) + float((it.get("bm25") or 0.0) * 1e-4)
        return s

    keys_sorted = sorted(groups.keys(), key=lambda k: gscore(groups[k]), reverse=True)[:max(1, top_groups)]

    # Thu thập id đã có để tránh trùng
    have_ids = set([str(c.get("id")) for c in pool if c.get("id")])

    extra: List[Dict[str,Any]] = []
    for k in keys_sorted:
        doc_id, dieu_key = k
        where = {"$and": [{"doc_id": {"$eq": doc_id}}]}
        if isinstance(dieu_key, tuple) and dieu_key and dieu_key[0] == "DIEU":
            # fallback theo 'dieu'
            where["$and"].append({"dieu": {"$eq": dieu_key[1]}})
        else:
            where["$and"].append({"ancestor_dieu_id": {"$eq": dieu_key}})
        try:
            got = col.get(where=where, limit=limit_per_group, include=_safe_include(["documents", "metadatas"]))
        except Exception:
            continue
        docs = (got or {}).get("documents") or []
        mds  = (got or {}).get("metadatas") or []
        ids  = (got or {}).get("ids") or []
        for doc, md, cid in zip(docs, mds, ids):
            cid_s = str(cid)
            if cid_s in have_ids:
                continue
            item = _make_item_from_get(doc, md, cid_s)
            extra.append(item)
            have_ids.add(cid_s)
            if len(extra) >= max_extra:
                break
        if len(extra) >= max_extra:
            break

    # Trộn: ưu tiên pool trước, rồi extra
    return pool + extra

# ================== Reranking (module riêng) ==================
from rerankers import rerank_candidates, infer_query_modality as rr_infer_modality

# ================== Utilities: render bảng ==================
_HTML_TAG = re.compile(r"<[^>]+>")

def is_markdown_table(text: str) -> bool:
    if not text: return False
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if len(lines) < 2: return False
    has_bar = any(line.count("|") >= 2 for line in lines[:5])
    has_sep = any(re.match(r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$", ln) for ln in lines[:6])
    return has_bar and has_sep

def extract_markdown_table_from_text(text: str) -> Optional[str]:
    if not text: return None
    t = text.lstrip()
    if t.startswith("[BẢNG]"): t = t[len("[BẢNG]"):].lstrip()
    lines = t.splitlines()
    start = -1
    for i, ln in enumerate(lines):
        if "|" in ln and i + 1 < len(lines) and re.search(r"-{3,}", lines[i+1]):
            start = i; break
    if start < 0: return None
    block, blanks = [], 0
    for j in range(start, len(lines)):
        if not lines[j].strip():
            blanks += 1
            if blanks >= 2: break
        else:
            blanks = 0
        block.append(lines[j])
    md = "\n".join(block).strip()
    return md if is_markdown_table(md) else None

def truncate_markdown_table(md: str, max_rows: int = 30) -> str:
    lines = [ln for ln in md.splitlines()]
    if len(lines) <= 2: return md
    sep_idx = -1
    for i, ln in enumerate(lines[:6]):
        if re.search(r"-{3,}", ln): sep_idx = i; break
    if sep_idx < 0: return md
    header = lines[:sep_idx]
    sep = lines[sep_idx:sep_idx+1]
    body = lines[sep_idx+1:]
    if len(body) <= max_rows:
        return md
    body_trunc = body[:max_rows] + [f"... (cắt bớt, còn {len(body) - max_rows} dòng)"]
    return "\n".join(header + sep + body_trunc)

def build_citation_label(item: Dict[str,Any]) -> str:
    segs = []
    if item.get("van_ban"): segs.append(item["van_ban"])
    parts = []
    if item.get("dieu"):  parts.append(f"Điều {item['dieu']}")
    if item.get("khoan"): parts.append(f"Khoản {item['khoan']}")
    if item.get("diem"):  parts.append(f"Điểm {item['diem']}")
    right = ", ".join(parts)
    label = segs[0] + (" — " + right if right else "") if segs else right
    p1, p2 = item.get("page_start"), item.get("page_end")
    if p1 is not None or p2 is not None:
        a = p1 if p1 is not None else p2
        b = p2 if p2 is not None else p1
        label += f" | trang {a}–{b}" if (a is not None and b is not None and a != b) else (f" | trang {a}" if a is not None else "")
    return label or (item.get("citation") or "")

def uniq_preserve(seq: List[str]) -> List[str]:
    seen = set(); out = []
    for x in seq:
        if not x or x in seen: continue
        seen.add(x); out.append(x)
    return out

def pick_table_blocks_from_ranked(
    ranked: List[Dict[str,Any]],
    max_tables: int,
    max_rows: int,
    max_chars: int
) -> List[Tuple[str, str]]:
    out: List[Tuple[str,str]] = []
    for item in ranked:
        if len(out) >= max_tables: break
        txt = item.get("text") or ""
        mod = (item.get("modality") or "").lower()
        md = extract_markdown_table_from_text(txt)
        if not md and mod == "table":
            html_ex = item.get("table_html_excerpt")
            if isinstance(html_ex, list) and html_ex: html_ex = html_ex[0]
            if isinstance(html_ex, str) and html_ex.strip():
                md = "```html\n" + html_ex.strip() + "\n```"
        if not md: continue
        md = truncate_markdown_table(md, max_rows=max_rows)
        if max_chars and len(md) > max_chars:
            md = md[:max_chars] + "\n... (cắt bớt theo giới hạn ký tự)"
        out.append((md, build_citation_label(item)))
    return out

# ================== Compose answer ==================
REFUSAL = "Không đủ thông tin từ nguồn đã lập chỉ mục."

def render_sources(ranked: List[Dict[str,Any]]) -> List[str]:
    labs = [ build_citation_label(x) for x in ranked ]
    return uniq_preserve(labs)

def make_system_prompt(is_table_query: bool) -> str:
    base = (
        "Bạn là trợ lý pháp lý PCCC. Trả lời bằng tiếng Việt, đầy đủ, có cấu trúc, "
        f"chỉ dùng NGỮ CẢNH cung cấp; nếu không đủ căn cứ, trả: \"{REFUSAL}\".\n"
        "- Chèn trích dẫn dạng [n] trỏ tới danh sách SOURCES ở cuối.\n"
        "- Với câu hỏi về HÌNH: tóm lược nội dung/caption, nêu điều/khoản/điểm và trang.\n"
    )
    if is_table_query:
        base += (
            "- Với câu hỏi về BẢNG: TRƯỚC HẾT hãy in BẢNG (nguyên văn) ở định dạng Markdown (giữ nguyên tiêu đề cột), "
            "sau đó tóm lược điểm chính và nêu căn cứ (điều/khoản/điểm, trang). Nếu context chứa HTML bảng trong "
            "khối ```html```, hãy chuyển sang bảng Markdown trước khi trả lời.\n"
        )
    return base

def make_user_prompt(query: str, contexts: List[str], sources: List[str], table_blocks: List[Tuple[str,str]], is_table_query: bool):
    ctx_blocks = []
    for i, (ctx, src) in enumerate(zip(contexts, sources), 1):
        ctx_blocks.append(f"[{i}] SOURCE: {src}\nEXCERPT:\n{ctx}")
    ctx_text = "\n\n".join(ctx_blocks)

    tbl_text = ""
    if is_table_query and table_blocks:
        tbl_lines = []
        for i, (md, src) in enumerate(table_blocks, 1):
            tbl_lines.append(f"### BẢNG {i} — {src}\n{md}")
        tbl_text = "\n\n".join(tbl_lines)

    user = (
        f"CÂU HỎI: {query}\n\n"
        + (f"BẢNG NGUYÊN VĂN (để bạn sử dụng, nếu phù hợp):\n{tbl_text}\n\n" if tbl_text else "")
        + f"NGỮ CẢNH (trích từ kho liệu):\n{ctx_text}\n\n"
        "YÊU CẦU TRẢ LỜI:\n"
        "- Nếu có bảng phù hợp: in bảng Markdown trước, rồi giải thích/đối chiếu căn cứ. Nếu nhiều bảng, chọn bảng phù hợp nhất.\n"
        "- Trả lời rõ ràng, từng ý; nêu điều/khoản/điểm, trang; chèn chỉ mục [n] cho câu/ý có căn cứ.\n"
        "- Nếu văn bản mới và cũ khác nhau, hãy ưu tiên văn bản mới và nêu rõ khác biệt.\n"
        "- Nếu không đủ căn cứ: trả đúng thông điệp chuẩn."
    )
    return [
        {"role":"system","content": make_system_prompt(is_table_query)},
        {"role":"user","content": user}
    ]

def answer_with_llm(query: str, ranked: List[Dict[str,Any]]) -> Dict[str,Any]:
    if not ranked:
        return {"answer": REFUSAL, "sources": []}

    contexts = [(x.get("text") or "").strip() for x in ranked]
    sources  = render_sources(ranked)

    is_table_query = (infer_query_modality(query) == "table") or any(((x.get("modality") or "")=="table") for x in ranked[:2])
    max_rows  = int(os.getenv("TABLE_RENDER_MAX_ROWS", "30"))
    max_tabs  = int(os.getenv("TABLE_RENDER_MAX_TABLES", "2"))
    max_chars = int(os.getenv("TABLE_RENDER_MAX_CHARS", "12000"))
    table_blocks = pick_table_blocks_from_ranked(ranked, max_tables=max_tabs, max_rows=max_rows, max_chars=max_chars) if is_table_query else []

    client = get_openai_client()
    msgs = make_user_prompt(query, contexts, sources, table_blocks, is_table_query)
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=msgs,
        temperature=0.2,
        max_tokens=900,
    )
    ans = resp.choices[0].message.content.strip()
    return {"answer": ans, "sources": sources}

# ================== Public API ==================
def answer_question(query: str, collection: str = COLLECTION, top_k_retrieve: Optional[int] = None,
                    prefer_modality: Optional[str] = None, only_modality: Optional[bool] = None) -> Dict[str,Any]:
    # 0) Prompt guard
    if os.getenv("PROMPT_GUARD", "1") == "1":
        g = guard_prompt(query)
        if g.get("status") == "empty":
            return {
                "answer": "Bạn muốn hỏi nội dung PCCC nào? Ví dụ: 'Nêu các biện pháp phòng cháy chữa cháy cho nhà xưởng'.",
                "sources": [],
                "suggestions": g.get("suggestions") or [],
            }
        if g.get("status") == "off_topic":
            sugs = g.get("suggestions") or []
            hint = ("\n- " + "\n- ".join(sugs)) if sugs else ""
            return {
                "answer": ("Câu hỏi hiện chưa thuộc lĩnh vực PCCC. Vui lòng đặt lại câu hỏi liên quan đến PCCC." + hint).strip(),
                "sources": [],
                "suggestions": sugs,
            }
        query = g.get("normalized_query") or query

    # Tham số
    topk_dense = int(os.getenv("TOPK_DENSE", os.getenv("TOPK_RETRIEVE", "20")))
    topk_lex   = int(os.getenv("TOPK_LEXICAL", "50"))
    final_k    = int(os.getenv("FINAL_K", "8"))
    hybrid_on  = (os.getenv("HYBRID_SEARCH", "1") == "1")

    # Tự động gợi ý modality
    if prefer_modality is None:
        pm = os.getenv("PREFER_MODALITY", "auto").lower()
        if pm in {"table","image","text"}:
            prefer_modality = pm
        elif pm == "auto":
            prefer_modality = infer_query_modality(query)
        else:
            prefer_modality = None

    # STRICT_TABLE_RETRIEVE / ONLY_MODALITY
    if only_modality is None:
        if (prefer_modality == "table") and (os.getenv("STRICT_TABLE_RETRIEVE","0") == "1"):
            only_modality = True
        else:
            only_modality = (os.getenv("ONLY_MODALITY","0") == "1")

    # 1) Retrieve (hybrid hoặc dense-only)
    if hybrid_on and BM25Okapi is not None:
        pool = retrieve_hybrid(
            query=query,
            collection_name=collection,
            topk_dense=topk_dense,
            topk_lex=topk_lex,
            prefer_modality=prefer_modality,
            only_modality=only_modality,
            final_limit=max(final_k*3, max(topk_dense, topk_lex)),
        )
    else:
        pool = retrieve_dense(query, collection, top_k=topk_dense, prefer_modality=prefer_modality, only_modality=only_modality)

    if not pool:
        return {"answer": REFUSAL, "sources": []}

    # 1b) Parent-scope expansion (giữ Điều nhất quán)
    if os.getenv("PARENT_EXPAND", "1") == "1":
        pool = expand_by_parent_scope(
            pool,
            collection_name=collection,
            top_groups=int(os.getenv("PARENT_TOP_GROUPS","3")),
            limit_per_group=int(os.getenv("PARENT_LIMIT_PER_GROUP","12")),
            max_extra=int(os.getenv("PARENT_MAX_EXTRA","36")),
        )

    # 2) Re-rank (cross-encoder + priors)
    weights = (
        float(os.getenv("W_CE","0.64")),
        float(os.getenv("W_VEC","0.18")),
        float(os.getenv("W_MOD","0.08")),
        float(os.getenv("W_REC","0.07")),
        float(os.getenv("W_COH","0.03")),
    )
    ranked = rerank_candidates(
        query=query,
        candidates=pool,
        backend=os.getenv("RERANK_BACKEND","cross_encoder"),
        model_name=os.getenv("RERANK_MODEL","BAAI/bge-reranker-v2-m3"),
        max_length=int(os.getenv("RERANK_MAX_LEN","512")),
        top_k_rerank=int(os.getenv("TOPK_RERANK","30")),
        final_k=int(os.getenv("FINAL_K","8")),
        prefer_modality=prefer_modality,
        weights=weights,
    )

    # 3) Generate
    return answer_with_llm(query, ranked)

# ================== CLI ==================
def main():
    p = argparse.ArgumentParser(description="QA Answerer PCCC — HYBRID (BM25 + Dense) → RRF → Parent Expand → Rerank → Answer")
    p.add_argument("--query", type=str, default=os.getenv("DEMO_QUERY", ""), help="Câu hỏi để demo")
    p.add_argument("--collection", type=str, default=COLLECTION, help="Tên collection Chroma")
    p.add_argument("--prefer_modality", type=str, default=None, choices=["table","image","text"], help="Ưu tiên modality")
    args = p.parse_args()

    q = args.query.strip()
    if not q:
        q = "Không xuất trình hồ sơ về phòng cháy, chữa cháy phục vụ kiểm tra bị phạt bao nhiêu tiền?"
        q = "Không cập nhật thông tin khi cơ sở có thay đổi so với thông tin đã khai báo trước đó vào hệ thống Cơ sở dữ liệu về phòng cháy bị phạt bao nhiêu"
    res = answer_question(q, collection=args.collection, prefer_modality=args.prefer_modality)
    print("\n--- ANSWER ---\n", res.get("answer",""))
    if isinstance(res.get("sources"), list) and res["sources"]:
        print("\n--- SOURCES ---")
        for i, s in enumerate(res["sources"], 1):
            print(f"[{i}] {s}")
    if isinstance(res.get("suggestions"), list) and res["suggestions"]:
        print("\n--- SUGGESTIONS ---")
        for s in res["suggestions"]:
            print("-", s)

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
QA Answerer cho PCCC (VN) — retrieve → (modality filter) → cross-encoder re-rank → generate
- Nếu câu hỏi về BẢNG: in chính nội dung bảng (Markdown) từ kho liệu trước, rồi mới giải thích & trích dẫn.
- Ưu tiên tốt các câu hỏi về BẢNG/HÌNH nhờ modality, table_html_excerpt, vv.

ENV:
  OPENAI_API_KEY
  CHROMA_DIR=./chroma_pccc_text_embedding_3_small
  CHROMA_COLLECTION=pccc_vi_text_embedding_3_small

  EMBED_MODEL=text-embedding-3-small
  CHAT_MODEL=gpt-4o-mini

  TOPK_RETRIEVE=20
  PREFER_MODALITY=auto       # auto|table|image|text|none
  ONLY_MODALITY=0            # 1 = lọc chặt theo modality
  STRICT_TABLE_RETRIEVE=0    # 1 = nếu query là 'table' thì ép only_modality=True

  # Rerank
  RERANK_BACKEND=cross_encoder
  RERANK_MODEL=BAAI/bge-reranker-v2-m3
  RERANK_MAX_LEN=512
  TOPK_RERANK=30
  FINAL_K=8
  W_CE=0.72 W_VEC=0.20 W_PRI=0.08

  # Render bảng
  TABLE_RENDER_MAX_ROWS=30
  TABLE_RENDER_MAX_TABLES=2
  TABLE_RENDER_MAX_CHARS=12000
"""

from __future__ import annotations
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import os, re, json, argparse

import chromadb
from openai import OpenAI

# ====== Embedding tokenizer (để cắt an toàn) ======
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

# ====== OpenAI ======
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
CHAT_MODEL  = os.getenv("CHAT_MODEL", "gpt-4o-mini")

def get_openai_client() -> OpenAI:
    key = os.getenv("OPENAI_API_KEY", "").strip()
    assert key, "Thiếu OPENAI_API_KEY"
    return OpenAI(api_key=key)

def embed_query(client: OpenAI, q: str) -> List[float]:
    q = truncate_tokens(q, int(os.getenv("MAX_EMBED_TOKENS", "8000")))
    return client.embeddings.create(model=EMBED_MODEL, input=[q]).data[0].embedding

# ====== Chroma ======
PERSIST_DIR   = Path(os.getenv("CHROMA_DIR", "./chroma_pccc_text_embedding_3_small"))
COLLECTION    = os.getenv("CHROMA_COLLECTION", "pccc_vi_text_embedding_3_small")
VALID_INCLUDE = {"documents","metadatas","distances","embeddings","uris","data"}

def get_chroma_collection(name: str):
    client = chromadb.PersistentClient(path=str(PERSIST_DIR))
    return client.get_or_create_collection(name=name)

def _safe_include(items: List[str]):
    return [x for x in items if x in VALID_INCLUDE] or None

# ====== Heuristics modality ======
_TABLE_PAT = re.compile(r"\b(bảng|bang|table|biểu|biểu)\b", re.I | re.U)
_IMAGE_PAT = re.compile(r"\b(hình|hinh|ảnh|anh|figure|sơ đồ|so do|biểu đồ|biểu đồ)\b", re.I | re.U)

def infer_query_modality(query: str) -> Optional[str]:
    if _TABLE_PAT.search(query or ""): return "table"
    if _IMAGE_PAT.search(query or ""): return "image"
    return None

# ====== Retrieve (ưu tiên modality cho bảng/hình; không dùng $or 1 phần tử) ======
def retrieve(
    query: str,
    collection_name: str,
    top_k: int = 20,
    prefer_modality: Optional[str] = None,   # None|"table"|"image"|"text"
    only_modality: bool = False,
) -> Dict[str, Any]:
    col = get_chroma_collection(collection_name)
    oai = get_openai_client()
    qvec = embed_query(oai, query)

    where = None
    where_document = None
    if prefer_modality in {"table", "image", "text"} and only_modality:
        where = {"modality": {"$eq": prefer_modality}}
        if prefer_modality == "table":
            where_document = {"$contains": "[BẢNG]"}
        elif prefer_modality == "image":
            where_document = {"$contains": "[HÌNH]"}

    res = col.query(
        query_embeddings=[qvec],
        n_results=top_k,
        where=where,
        where_document=where_document,
        include=_safe_include(["documents","metadatas","distances"]),
    )

    out = []
    ids_ = res.get("ids", [[]])[0]
    docs_ = res.get("documents", [[]])[0] if res.get("documents") else []
    metas_ = res.get("metadatas", [[]])[0] if res.get("metadatas") else []
    dists_ = res.get("distances", [[]])[0] if res.get("distances") else []

    for i in range(max(len(ids_), len(docs_), len(metas_), len(dists_))):
        m = metas_[i] if i < len(metas_) else {}
        item = {
            "id": ids_[i] if i < len(ids_) else None,
            "distance": dists_[i] if i < len(dists_) else None,
            "text": docs_[i] if i < len(docs_) else "",
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
            "table_html_excerpt": (m or {}).get("table_html_excerpt"),
            "has_image": (m or {}).get("has_image"),
        }
        out.append(item)
    return {"query": query, "top_k": top_k, "results": out}

# ====== Reranking (cross-encoder) ======
try:
    from sentence_transformers import CrossEncoder
except Exception:
    CrossEncoder = None

RERANK_BACKEND = os.getenv("RERANK_BACKEND", "cross_encoder")  # "cross_encoder"| "none"
RERANK_MODEL   = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
RERANK_MAX_LEN = int(os.getenv("RERANK_MAX_LEN", "512"))
TOPK_RERANK    = int(os.getenv("TOPK_RERANK", "30"))
FINAL_K        = int(os.getenv("FINAL_K", "8"))
W_CE = float(os.getenv("W_CE","0.72"))
W_VEC= float(os.getenv("W_VEC","0.20"))
W_PRI= float(os.getenv("W_PRI","0.08"))

_HTML_TAG = re.compile(r"<[^>]+>")

def strip_html_keep_space(html: str, max_chars: int = 1200) -> str:
    if not html: return ""
    txt = _HTML_TAG.sub(" ", html)
    txt = re.sub(r"\s+", " ", txt).strip()
    if max_chars and len(txt) > max_chars:
        txt = txt[:max_chars] + " ..."
    return txt

def normalize_distance_to_similarity(distances: List[Optional[float]]) -> List[float]:
    vals = [d for d in distances if d is not None]
    if not vals: return [0.0] * len(distances)
    mn, mx = min(vals), max(vals)
    if mx <= mn + 1e-12: return [1.0] * len(distances)
    return [1.0 - ((d - mn)/(mx - mn)) if d is not None else 0.0 for d in distances]

def make_rerank_text(cand: Dict[str,Any]) -> str:
    txt = cand.get("text") or ""
    if (cand.get("modality") == "table") or str(txt).lstrip().startswith("[BẢNG]"):
        html_ex = cand.get("table_html_excerpt")
        if isinstance(html_ex, list) and html_ex:
            html_ex = html_ex[0]
        if isinstance(html_ex, str) and html_ex.strip():
            txt += "\n[BẢNG_HTML_TRÍCH] " + strip_html_keep_space(html_ex)
    return txt

def compute_modality_prior(query: str, cands: List[Dict[str,Any]], prefer_modality: Optional[str]) -> List[float]:
    want = prefer_modality or infer_query_modality(query)
    out = []
    for c in cands:
        mod = (c.get("modality") or "").lower()
        if want and mod == want:
            out.append(1.0)
        elif want == "table" and str(c.get("text","")).lstrip().startswith("[BẢNG]"):
            out.append(0.9)
        elif want == "image" and str(c.get("text","")).lstrip().startswith("[HÌNH]"):
            out.append(0.9)
        else:
            out.append(0.0)
    return out

def fuse_scores(ce_scores: List[float], vec_sims: List[float], pri: List[float], w_ce: float, w_vec: float, w_prior: float) -> List[float]:
    def _minmax(x):
        if not x: return x
        mn, mx = min(x), max(x)
        if mx <= mn + 1e-12: return [1.0]*len(x)
        return [(v - mn)/(mx - mn) for v in x]
    ce_n  = _minmax(ce_scores)
    vec_n = _minmax(vec_sims)
    pri_n = _minmax(pri)
    return [w_ce*ce_n[i] + w_vec*vec_n[i] + w_prior*pri_n[i] for i in range(len(ce_scores))]

def rerank_candidates(
    query: str,
    candidates: List[Dict[str,Any]],
    *,
    backend: str = RERANK_BACKEND,
    model_name: str = RERANK_MODEL,
    max_length: int = RERANK_MAX_LEN,
    top_k_rerank: int = TOPK_RERANK,
    final_k: int = FINAL_K,
    prefer_modality: Optional[str] = None,
    weights: Tuple[float,float,float] = (W_CE, W_VEC, W_PRI)
) -> List[Dict[str,Any]]:
    if not candidates: return []
    pool = candidates[:min(top_k_rerank, len(candidates))]

    docs_for_ce = [ make_rerank_text(c) for c in pool ]
    vec_sims = normalize_distance_to_similarity([ c.get("distance") for c in pool ])
    pri = compute_modality_prior(query, pool, prefer_modality)

    if backend == "cross_encoder":
        assert CrossEncoder is not None, "Cần cài sentence-transformers để dùng cross-encoder."
        model = CrossEncoder(model_name, max_length=max_length)
        ce_scores = model.predict([[query, d] for d in docs_for_ce], convert_to_numpy=True).tolist()
    else:
        ce_scores = [0.0]*len(pool)

    w_ce, w_vec, w_prior = weights
    fused = fuse_scores(ce_scores, vec_sims, pri, w_ce, w_vec, w_prior)
    for i, c in enumerate(pool):
        c["score_ce"] = ce_scores[i]
        c["score_vec"] = vec_sims[i]
        c["score_prior"] = pri[i]
        c["score_rerank"] = fused[i]
    pool.sort(key=lambda x: x["score_rerank"], reverse=True)
    return pool[:final_k]

# ====== Utilities: render bảng ======

def is_markdown_table(text: str) -> bool:
    """Heuristic nhận diện bảng Markdown."""
    if not text: return False
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if len(lines) < 2: return False
    # cần có dòng có nhiều '|' và có dòng phân cách '---'
    has_bar = any(line.count("|") >= 2 for line in lines[:5])
    has_sep = any(re.match(r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$", ln) for ln in lines[:6])
    return has_bar and has_sep

def extract_markdown_table_from_text(text: str) -> Optional[str]:
    """Trích khối bảng Markdown đầu tiên từ text (bỏ tiền tố [BẢNG] nếu có)."""
    if not text: return None
    t = text.lstrip()
    if t.startswith("[BẢNG]"): t = t[len("[BẢNG]"):].lstrip()
    lines = t.splitlines()
    # tìm đoạn có bảng (từ header -> separator -> body)
    start = -1
    for i, ln in enumerate(lines):
        if "|" in ln:
            # kiểm tra dòng sau có separator
            if i + 1 < len(lines) and re.search(r"-{3,}", lines[i+1]):
                start = i; break
    if start < 0: return None
    # gom đến khi gặp dòng trống dài hoặc hết
    block = []
    for j in range(start, len(lines)):
        if not lines[j].strip():
            # cho phép 1-2 dòng trống trong bảng: dừng khi có 2 dòng trống liên tiếp
            nxt = j+1
            if nxt < len(lines) and not lines[nxt].strip():
                break
        block.append(lines[j])
    md = "\n".join(block).strip()
    return md if is_markdown_table(md) else None

def truncate_markdown_table(md: str, max_rows: int = 30) -> str:
    """Giữ header + separator + N dòng body."""
    lines = [ln for ln in md.splitlines()]
    if len(lines) <= 2: return md
    # tìm separator (dòng 2 hoặc dòng nào đó có ---)
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

def pick_table_blocks_from_ranked(
    ranked: List[Dict[str,Any]],
    max_tables: int,
    max_rows: int,
    max_chars: int
) -> List[Tuple[str, str]]:
    """
    Trả về list (table_markdown, source_label) lấy từ ranked (ưu tiên modality=table).
    """
    out: List[Tuple[str,str]] = []
    count = 0
    for item in ranked:
        if count >= max_tables: break
        txt = item.get("text") or ""
        mod = (item.get("modality") or "").lower()
        md = extract_markdown_table_from_text(txt)
        if not md and mod == "table":
            # thử từ HTML excerpt (cho LLM tự render): đặt như code block để không lẫn
            html_ex = item.get("table_html_excerpt")
            if isinstance(html_ex, list) and html_ex:
                html_ex = html_ex[0]
            if isinstance(html_ex, str) and html_ex.strip():
                md = "```html\n" + html_ex.strip() + "\n```"
        if not md:
            continue
        md = truncate_markdown_table(md, max_rows=max_rows)
        if max_chars and len(md) > max_chars:
            md = md[:max_chars] + "\n... (cắt bớt theo giới hạn ký tự)"
        out.append((md, build_citation_label(item)))
        count += 1
    return out

# ====== Compose answer ======
REFUSAL = "Không đủ thông tin từ nguồn đã lập chỉ mục."

def uniq_preserve(seq: List[str]) -> List[str]:
    seen = set(); out = []
    for x in seq:
        if not x or x in seen: continue
        seen.add(x); out.append(x)
    return out

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

def render_sources(ranked: List[Dict[str,Any]]) -> List[str]:
    labs = [ build_citation_label(x) for x in ranked ]
    return uniq_preserve(labs)

def make_system_prompt(is_table_query: bool) -> str:
    base = (
        "Bạn là trợ lý pháp lý PCCC. Trả lời bằng tiếng Việt, đầy đủ, có cấu trúc, "
        "ưu tiên căn cứ từ văn bản mới nhất. Nếu không đủ căn cứ, trả: "
        f"\"{REFUSAL}\".\n"
        "- Chèn trích dẫn dạng [n] trỏ tới danh sách SOURCES ở cuối.\n"
        "- Với câu hỏi về HÌNH: tóm lược nội dung caption/ngữ cảnh, nêu rõ điều/khoản/điểm và trang.\n"
    )
    if is_table_query:
        base += (
            "- Với câu hỏi về BẢNG: TRƯỚC HẾT hãy in BẢNG (nguyên văn) ở định dạng Markdown (giữ nguyên tiêu đề cột), "
            "sau đó tóm lược điểm chính và nêu căn cứ (điều/khoản/điểm, trang). Nếu context chứa HTML bảng trong "
            "khối ```html```, hãy chuyển hóa sang bảng Markdown trước khi trả lời.\n"
        )
    return base

def make_user_prompt(query: str, contexts: List[str], sources: List[str], table_blocks: List[Tuple[str,str]], is_table_query: bool) -> List[Dict[str,str]]:
    # Ghép context theo dạng đánh số để dễ dẫn [n]
    ctx_blocks = []
    for i, (ctx, src) in enumerate(zip(contexts, sources), 1):
        ctx_blocks.append(f"[{i}] SOURCE: {src}\nEXCERPT:\n{ctx}")
    ctx_text = "\n\n".join(ctx_blocks)

    # Nếu có bảng, truyền thêm khối BẢNG để LLM dùng trực tiếp
    tbl_text = ""
    if is_table_query and table_blocks:
        tbl_lines = []
        for i, (md, src) in enumerate(table_blocks, 1):
            tbl_lines.append(f"### BẢNG {i} — {src}\n{md}")
        tbl_text = "\n\n".join(tbl_lines)

    user = (
        f"CÂU HỎI: {query}\n\n"
        + (f"BẢNG NGUYÊN VĂN (để bạn sử dụng, nếu phù hợp):\n{tbl_text}\n\n" if tbl_text else "")
        + f"NGỮ CẢNH (trích từ kho liệu, có thể chứa [BẢNG]/[HÌNH] hoặc ```html``` cho bảng):\n{ctx_text}\n\n"
        "YÊU CẦU TRẢ LỜI:\n"
        "- Nếu có bảng phù hợp: in bảng (Markdown) trước, rồi giải thích/đối chiếu căn cứ. Nếu có nhiều bảng, chọn bảng phù hợp nhất.\n"
        "- Trả lời rõ ràng, từng ý; nêu điều/khoản/điểm, trang; chèn chỉ mục [n] cho mệnh đề có căn cứ.\n"
        "- Nếu văn bản mới và cũ khác nhau, hãy ưu tiên văn bản mới và nêu rõ sự khác biệt.\n"
        "- Nếu không đủ căn cứ: trả đúng thông điệp chuẩn."
    )
    return [
        {"role":"system","content": make_system_prompt(is_table_query)},
        {"role":"user","content": user}
    ]

def answer_with_llm(query: str, ranked: List[Dict[str,Any]]) -> Dict[str,Any]:
    if not ranked:
        return {"answer": REFUSAL, "sources": []}

    # Chuẩn bị ngữ cảnh
    contexts = []
    for x in ranked:
        txt = x.get("text") or ""
        contexts.append(txt.strip())
    sources = render_sources(ranked)

    # Chuẩn bị khối bảng để LLM in ra trực tiếp
    is_table_query = (infer_query_modality(query) == "table") or any((x.get("modality")=="table") for x in ranked[:2])
    max_rows  = int(os.getenv("TABLE_RENDER_MAX_ROWS", "30"))
    max_tabs  = int(os.getenv("TABLE_RENDER_MAX_TABLES", "2"))
    max_chars = int(os.getenv("TABLE_RENDER_MAX_CHARS", "12000"))
    table_blocks = pick_table_blocks_from_ranked(ranked, max_tables=max_tabs, max_rows=max_rows, max_chars=max_chars) if is_table_query else []

    msgs = make_user_prompt(query, contexts, sources, table_blocks, is_table_query)

    client = get_openai_client()
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=msgs,
        temperature=0.2,
        max_tokens=900,
    )
    ans = resp.choices[0].message.content.strip()
    return {"answer": ans, "sources": sources}

# ====== Public API ======
def answer_question(query: str, collection: str = COLLECTION, top_k_retrieve: int = None,
                    prefer_modality: Optional[str] = None, only_modality: Optional[bool] = None) -> Dict[str,Any]:
    if top_k_retrieve is None:
        top_k_retrieve = int(os.getenv("TOPK_RETRIEVE", "20"))

    # Tự động gợi ý modality
    if prefer_modality is None:
        pm = os.getenv("PREFER_MODALITY", "auto").lower()
        if pm in {"table","image","text"}:
            prefer_modality = pm
        elif pm == "auto":
            prefer_modality = infer_query_modality(query)
        else:
            prefer_modality = None

    # Nếu câu hỏi là bảng và STRICT_TABLE_RETRIEVE=1 → lọc chặt theo modality=table
    if only_modality is None:
        if (prefer_modality == "table") and (os.getenv("STRICT_TABLE_RETRIEVE","0") == "1"):
            only_modality = True
        else:
            only_modality = (os.getenv("ONLY_MODALITY","0") == "1")

    # 1) Retrieve
    ret = retrieve(query, collection, top_k=top_k_retrieve,
                   prefer_modality=prefer_modality, only_modality=only_modality)
    cands = ret["results"]
    if not cands:
        return {"answer": REFUSAL, "sources": []}

    # 2) Re-rank
    ranked = rerank_candidates(
        query=query,
        candidates=cands,
        backend=RERANK_BACKEND,
        model_name=RERANK_MODEL,
        max_length=RERANK_MAX_LEN,
        top_k_rerank=TOPK_RERANK,
        final_k=FINAL_K,
        prefer_modality=prefer_modality,
        weights=(W_CE, W_VEC, W_PRI),
    )

    # 3) Generate
    out = answer_with_llm(query, ranked)
    return out

# ====== CLI ======
def main():
    p = argparse.ArgumentParser(description="QA Answerer PCCC (retriever → reranker → answerer, in bảng nếu được hỏi)")
    p.add_argument("--query", type=str, default=os.getenv("DEMO_QUERY", ""), help="Câu hỏi để demo")
    p.add_argument("--collection", type=str, default=COLLECTION, help="Tên collection Chroma")
    p.add_argument("--top_k", type=int, default=int(os.getenv("TOPK_RETRIEVE","20")))
    p.add_argument("--prefer_modality", type=str, default=None, choices=["table","image","text"], help="Ưu tiên modality")
    p.add_argument("--only_modalility", action="store_true", help="(deprecated) Không dùng; dùng ONLY_MODALITY/STRICT_TABLE_RETRIEVE qua ENV")
    args = p.parse_args()

    q = args.query.strip()
    q = "Bảng Chiều dài của bãi đỗ xe chữa cháy đối với nhà nhóm F5"
    if not q:
        print("Vui lòng cung cấp --query hoặc đặt DEMO_QUERY.")
        return
    res = answer_question(q, collection=args.collection, top_k_retrieve=args.top_k)
    print("\n--- ANSWER ---\n", res["answer"])
    if res["sources"]:
        print("\n--- SOURCES ---")
        for i, s in enumerate(res["sources"], 1):
            print(f"[{i}] {s}")

if __name__ == "__main__":
    main()

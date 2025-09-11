#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
QA Answerer PCCC (VN) — retrieve → (lọc/ưu tiên modality) → rerank → generate
- Nếu câu hỏi về BẢNG: in nguyên văn bảng Markdown trước, rồi giải thích & trích dẫn.
- Ưu tiên văn bản mới (recency) nếu metadata có ngày; giảm trùng.
- Agent chuẩn hoá prompt & chống lệch chủ đề (prompt_guard).

ENV:
  OPENAI_API_KEY
  CHROMA_DIR=./chroma_pccc_text_embedding_3_small
  CHROMA_COLLECTION=pccc_vi_text_embedding_3_small

  EMBED_MODEL=text-embedding-3-small
  CHAT_MODEL=gpt-4o-mini

  TOPK_RETRIEVE=20
  PREFER_MODALITY=auto       # auto|table|image|text|none
  ONLY_MODALITY=0            # 1 = lọc chặt theo modality bằng where_document [BẢNG]/[HÌNH]
  STRICT_TABLE_RETRIEVE=0    # 1 = nếu query là 'table' thì ép only_modality=True

  # Rerank
  RERANK_BACKEND=cross_encoder
  RERANK_MODEL=BAAI/bge-reranker-v2-m3
  RERANK_MAX_LEN=512
  TOPK_RERANK=30
  FINAL_K=8

  # Trọng số rerank (w_ce,w_vec,w_mod,w_rec,w_coh)
  W_CE=0.64 W_VEC=0.18 W_MOD=0.08 W_REC=0.07 W_COH=0.03

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
import os, re, json, argparse

import chromadb
from openai import OpenAI
from prompt_guard import guard_prompt

# ---------- Embedding tokenizer ----------
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

# ---------- OpenAI ----------
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
CHAT_MODEL  = os.getenv("CHAT_MODEL", "gpt-4o-mini")

def get_openai_client() -> OpenAI:
    key = os.getenv("OPENAI_API_KEY", "").strip()
    assert key, "Thiếu OPENAI_API_KEY"
    return OpenAI(api_key=key)

def embed_query(client: OpenAI, q: str) -> List[float]:
    q = truncate_tokens(q, int(os.getenv("MAX_EMBED_TOKENS", "8000")))
    return client.embeddings.create(model=EMBED_MODEL, input=[q]).data[0].embedding

# ---------- Chroma ----------
PERSIST_DIR   = Path(os.getenv("CHROMA_DIR", "./chroma_pccc_text_embedding_3_small"))
COLLECTION    = os.getenv("CHROMA_COLLECTION", "pccc_vi_text_embedding_3_small")
VALID_INCLUDE = {"documents","metadatas","distances","embeddings","uris","data"}

def get_chroma_collection(name: str):
    client = chromadb.PersistentClient(path=str(PERSIST_DIR))
    return client.get_or_create_collection(name=name)

def _safe_include(items: List[str]):
    return [x for x in items if x in VALID_INCLUDE] or None

# ---------- Heuristics modality ----------
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

# ---------- Retrieve (dùng where_document để lọc theo nội dung văn bản) ----------
# Tài liệu: Chroma hỗ trợ where_document với toán tử $contains để lọc trên nội dung document. :contentReference[oaicite:3]{index=3}
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
    if prefer_modality in {"table", "image"} and only_modality:
        if prefer_modality == "table":
            where_document = {"$contains": "[BẢNG]"}
        elif prefer_modality == "image":
            where_document = {"$contains": "[HÌNH]"}
    # prefer_modality="text" thì không lọc cứng; để rerank xử lý.

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
        doc_text = docs_[i] if i < len(docs_) else ""
        m = metas_[i] if i < len(metas_) else {}
        modality = (m or {}).get("modality") or _infer_modality_from_text(doc_text)
        item = {
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
        }
        out.append(item)
    return {"query": query, "top_k": top_k, "results": out}

# ---------- Reranking ----------
from rerankers import rerank_candidates, infer_query_modality as rr_infer_modality

# ---------- Utilities: render bảng ----------
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

# ---------- Compose answer ----------
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

def make_user_prompt(query: str, contexts: List[str], sources: List[str], table_blocks: List[Tuple[str,str]], is_table_query: bool) -> List[Dict[str,str]]:
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

# ---------- Public API ----------
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

    # STRICT_TABLE_RETRIEVE
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
    weights = (
        float(os.getenv("W_CE","0.64")),
        float(os.getenv("W_VEC","0.18")),
        float(os.getenv("W_MOD","0.08")),
        float(os.getenv("W_REC","0.07")),
        float(os.getenv("W_COH","0.03")),
    )
    ranked = rerank_candidates(
        query=query,
        candidates=cands,
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

# ---------- CLI ----------
def main():
    p = argparse.ArgumentParser(description="QA Answerer PCCC (retriever → reranker → answerer, in bảng nếu được hỏi)")
    p.add_argument("--query", type=str, default=os.getenv("DEMO_QUERY", ""), help="Câu hỏi để demo")
    p.add_argument("--collection", type=str, default=COLLECTION, help="Tên collection Chroma")
    p.add_argument("--top_k", type=int, default=int(os.getenv("TOPK_RETRIEVE","20")))
    p.add_argument("--prefer_modality", type=str, default=None, choices=["table","image","text"], help="Ưu tiên modality")
    args = p.parse_args()

    q = args.query.strip() or "Cách ăn "
    res = answer_question(q, collection=args.collection, top_k_retrieve=args.top_k, prefer_modality=args.prefer_modality)
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

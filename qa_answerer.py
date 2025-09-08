#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RAG Answerer cho PCCC: Retriever → (tùy chọn) Reranker → Answerer (OpenAI)

Chức năng chính:
- embed_query(): nhúng câu hỏi bằng OpenAI Embeddings
- chroma_query(): truy hồi top-K từ Chroma (kèm metadata/citation)
- maybe_rerank(): rerank bằng Cross-Encoder (BAAI/bge-reranker-v2-m3) nếu khả dụng
- prefer_freshness(): ưu tiên mảnh mới hơn theo 'effective_date' (nếu đã enrich)
- build_context_blocks(): chọn & đóng gói ngữ cảnh với ngân sách token (tiktoken)
- generate_answer(): gọi OpenAI Chat Completions, ép trích dẫn [n] & câu từ chối chuẩn
- answer(): hàm 1 phát từ câu hỏi → (ans, sources)

Yêu cầu:
  pip install "chromadb>=0.5" "openai>=1.30.0"
  # (khuyến nghị) cho reranker:
  # pip install "sentence-transformers>=3.0" "transformers>=4.40" "torch>=2.1" -f https://download.pytorch.org/whl/cpu
  # (khuyến nghị) cho cắt token:
  # pip install "tiktoken>=0.6"

Biến môi trường quan trọng:
  OPENAI_API_KEY=...
  CHROMA_DIR=./chroma_pccc_text_embedding_3_small
  CHROMA_COLLECTION=pccc_vi_text_embedding_3_small
  EMBED_MODEL=text-embedding-3-small
  GEN_MODEL=gpt-4o-mini
  RETRIEVER_TOPK=8
  CONTEXT_TOPN=5
  USE_RERANK=1
"""

from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import os, json, re, math, sys
import chromadb
from openai import OpenAI

# =============== Cấu hình ===============

PERSIST_DIR = Path(os.getenv("CHROMA_DIR", "./chroma_pccc_text_embedding_3_small"))
COLLECTION  = os.getenv("CHROMA_COLLECTION", "pccc_vi_text_embedding_3_small")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")  # 1536d (OpenAI)
GEN_MODEL   = os.getenv("GEN_MODEL", "gpt-4o-mini")

TOPK        = int(os.getenv("RETRIEVER_TOPK", "8"))     # lấy từ vector DB
TOPN_CTX    = int(os.getenv("CONTEXT_TOPN", "5"))       # mảnh đưa vào prompt sau (re)rank
MIN_CTX     = int(os.getenv("MIN_CTX", "1"))            # tối thiểu để trả lời; nếu < MIN_CTX → từ chối

USE_RERANK  = os.getenv("USE_RERANK", "1") == "1"
RERANK_MODEL= os.getenv("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
RERANK_MAXLEN = int(os.getenv("RERANK_MAXLEN", "512"))  # token max mỗi pair query-passage

# Ngân sách token khi build prompt (để không vượt context của GEN_MODEL)
PROMPT_BUDGET = int(os.getenv("PROMPT_BUDGET", "6000"))  # tổng token (ước tính)
PROMPT_MARGIN = int(os.getenv("PROMPT_MARGIN", "512"))   # biên an toàn
CTX_MAX_TOKENS= max(512, PROMPT_BUDGET - PROMPT_MARGIN)  # token dành cho phần context

REFUSAL = 'Không đủ thông tin từ nguồn đã lập chỉ mục.'  # câu từ chối chuẩn (KPI)

# =============== OpenAI client ===============

def get_client() -> OpenAI:
    key = os.getenv("OPENAI_API_KEY", "").strip()
    assert key, "Thiếu OPENAI_API_KEY"
    return OpenAI(api_key=key)

# =============== Token utilities (tiktoken) ===============

try:
    import tiktoken
    ENC_CHAT = tiktoken.get_encoding("cl100k_base")  # phù hợp GPT-4o/3.5/emb v3
except Exception:
    ENC_CHAT = None

def count_tokens(text: str) -> int:
    if ENC_CHAT is None:
        # fallback ước lượng bảo thủ
        return max(1, len(text) // 2)
    return len(ENC_CHAT.encode(text or ""))

def truncate_by_tokens(text: str, max_tokens: int) -> str:
    """Cắt chuỗi theo số token; nếu không có tiktoken, cắt theo ký tự ước lượng."""
    if text is None:
        return ""
    if count_tokens(text) <= max_tokens:
        return text
    if ENC_CHAT is None:
        # fallback: ~2 chars/token
        return text[: max_tokens * 2]
    ids = ENC_CHAT.encode(text)
    cut = ids[:max_tokens]
    return ENC_CHAT.decode(cut)

# =============== Embedding câu hỏi ===============

def embed_query(client: OpenAI, query: str, model: str) -> List[float]:
    # câu hỏi thường ngắn, gọi trực tiếp
    resp = client.embeddings.create(model=model, input=[query])
    return resp.data[0].embedding

# =============== Chroma retrieval ===============

VALID_INCLUDE = {"documents", "embeddings", "metadatas", "distances", "uris", "data"}

def _safe_include(items):
    if not items: return None
    out = [x for x in items if x in VALID_INCLUDE]
    return out or None

def chroma_query(qvec: List[float], topk: int, where: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    client = chromadb.PersistentClient(path=str(PERSIST_DIR))
    col = client.get_collection(name=COLLECTION)
    include_fields = _safe_include(["documents", "metadatas", "distances"])
    res = col.query(
        query_embeddings=[qvec],
        n_results=topk,
        where=where,
        include=include_fields,  # 'ids' luôn trả về mặc định
    )
    ids   = res.get("ids", [[]])[0]
    docs  = res.get("documents", [[]])[0] if res.get("documents") else []
    metas = res.get("metadatas", [[]])[0] if res.get("metadatas") else []
    dists = res.get("distances", [[]])[0] if res.get("distances") else []

    out = []
    n = len(ids)
    for i in range(n):
        m = metas[i] if i < len(metas) else {}
        out.append({
            "id": ids[i],
            "text": docs[i] if i < len(docs) else "",
            "distance": dists[i] if i < len(dists) else None,
            "citation": m.get("citation") or "",
            "van_ban": m.get("van_ban"),
            "dieu": m.get("dieu"),
            "khoan": m.get("khoan"),
            "diem": m.get("diem"),
            "page_start": m.get("page_start"),
            "page_end": m.get("page_end"),
            "section_key": m.get("section_key"),
            "effective_date": m.get("effective_date"),  # nếu đã enrich
        })
    return out

# =============== (Tùy chọn) Reranker cross-encoder ===============

def maybe_rerank(query: str, hits: List[Dict[str, Any]], topn: int) -> List[Dict[str, Any]]:
    """Rerank bằng cross-encoder nếu khả dụng; nếu không, fallback theo distance tăng dần."""
    if not USE_RERANK or not hits:
        # fallback: theo khoảng cách cosine (nhỏ hơn tốt hơn)
        return sorted(hits, key=lambda x: (x.get("distance") is None, x.get("distance")))[:topn]
    try:
        from sentence_transformers import CrossEncoder
        ce = CrossEncoder(RERANK_MODEL, max_length=RERANK_MAXLEN)
        pairs = [(query, h["text"] or "") for h in hits]
        scores = ce.predict(pairs)  # càng cao càng liên quan
        for h, sc in zip(hits, scores):
            h["_rerank"] = float(sc)
        hits = sorted(hits, key=lambda x: x.get("_rerank", 0.0), reverse=True)
        return hits[:topn]
    except Exception as e:
        # Nếu thiếu thư viện/model → dùng xếp hạng theo distance
        return sorted(hits, key=lambda x: (x.get("distance") is None, x.get("distance")))[:topn]

# =============== Ưu tiên văn bản mới (nếu có) ===============

def prefer_freshness(hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Giữ tối đa một mảnh cho mỗi section_key; nếu trùng, ưu tiên effective_date mới hơn."""
    if not hits:
        return hits
    chosen: Dict[str, Dict[str, Any]] = {}
    for h in hits:
        k = h.get("section_key") or h["id"]
        cur = chosen.get(k)
        if cur is None:
            chosen[k] = h
        else:
            d_new = h.get("effective_date") or ""
            d_old = cur.get("effective_date") or ""
            if d_new > d_old:
                chosen[k] = h
    return list(chosen.values())

# =============== Build context with token budget ===============

def build_context_blocks(hits: List[Dict[str, Any]], max_tokens: int) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Trả về (context_text, used_hits). Mỗi block dạng:
    [i] <citation>
    <text>
    """
    lines = []
    used = []
    budget = max_tokens
    for i, h in enumerate(hits, 1):
        cite = h.get("citation") or ""
        txt  = (h.get("text") or "").strip()
        if not txt:
            continue
        # Ưu tiên giữ citation đầy đủ, cắt phần text nếu cần
        header = f"[{len(used)+1}] {cite}\n"
        est_header = count_tokens(header)
        if est_header >= budget:
            break
        budget -= est_header

        # chừa 1-2 câu đầu đủ nghĩa nếu cần cắt
        # cắt theo token ngân sách còn lại cho đoạn này (để dành 1 token newline)
        allow = max(16, min(800, budget - 1))
        txt_cut = truncate_by_tokens(txt, allow)
        block = header + txt_cut + "\n\n"
        tok = count_tokens(block)
        if tok > budget:
            # cắt thêm nếu vượt (phòng sai số)
            txt_cut = truncate_by_tokens(txt_cut, max(1, budget - est_header - 4))
            block = header + txt_cut + "\n\n"
            tok = count_tokens(block)
        if tok <= budget:
            lines.append(block)
            used.append(h)
            budget -= tok
        if budget <= 64:  # còn quá ít thì dừng
            break
    return ("".join(lines).strip(), used)

# =============== Gọi LLM sinh trả lời ===============

SYS = """Bạn là trợ lý pháp lý PCCC.
- Chỉ trả lời dựa trên NGUỒN được cung cấp (các block [n]).
- Nếu không đủ căn cứ để trả lời đúng trọng tâm, TRẢ LỜI CHÍNH XÁC: "Không đủ thông tin từ nguồn đã lập chỉ mục."
- Khi nêu kết luận/điều kiện/quy định, phải đính kèm chỉ số [n] tương ứng ngay sau mệnh đề liên quan.
- Giữ văn phong ngắn gọn, chính xác, tiếng Việt chuẩn; tránh suy đoán."""

def generate_answer(client: OpenAI, question: str, contexts: List[Dict[str, Any]]) -> str:
    if len(contexts) < MIN_CTX:
        return REFUSAL
    ctx_text, used = build_context_blocks(contexts, CTX_MAX_TOKENS)
    if not used:
        return REFUSAL

    user_prompt = (
        f"Câu hỏi: {question}\n\n"
        f"NGUỒN tham chiếu (chỉ dùng nội dung dưới đây để trả lời):\n{ctx_text}\n\n"
        f"YÊU CẦU:\n"
        f"- Trả lời ngắn gọn, chính xác và kèm [n] sau mỗi mệnh đề liên quan.\n"
        f"- Nếu không đủ căn cứ, trả về đúng chuỗi: \"{REFUSAL}\""
    )

    resp = client.chat.completions.create(
        model=GEN_MODEL,
        temperature=0.2,
        messages=[
            {"role": "system", "content": SYS},
            {"role": "user", "content": user_prompt},
        ],
    )
    return (resp.choices[0].message.content or "").strip()

# =============== API một phát ===============

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

def answer(question: str, where: Optional[Dict[str, Any]] = None) -> Tuple[str, List[Dict[str, Any]]]:
    client = get_client()
    qvec = embed_query(client, question, EMBED_MODEL)
    hits = chroma_query(qvec, TOPK, where=where)
    hits = prefer_freshness(hits)
    chosen = maybe_rerank(question, hits, topn=TOPN_CTX)
    ans = generate_answer(client, question, chosen)
    return ans, chosen

# =============== CLI ===============

def _print_sources(sources: List[Dict[str, Any]]):
    if not sources:
        print("\n--- SOURCES ---\n(None)")
        return
    print("\n--- SOURCES ---")
    for i, h in enumerate(sources, 1):
        cite = h.get("citation") or ""
        print(f"[{i}] {cite}")

if __name__ == "__main__":
    # q = "Yêu cầu về phòng cháy, chữa cháy khi lập, điều chỉnh dự án đầu tư xây dựng công trình, thiết kế công trình, cải tạo, thay đổi công năng sử dụng công trình, sản xuất, lắp ráp, đóng mới, hoán cải phương tiện giao thông" if len(sys.argv) == 1 else " ".join(sys.argv[1:])
    q = f"Trong các gara ô-tô dạng kín. Đưa ra Bảng Quy định về bố trí khoang đệm ngăn cháy trong gara ô-tô dạng kín"
    ans, ctx = answer(q)
    print("\n--- ANSWER ---\n", ans)
    _print_sources(ctx)
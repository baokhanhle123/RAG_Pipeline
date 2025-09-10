# /home/khanhle/RAG_Pipeline/rerankers.py
# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import List, Dict, Any, Optional, Tuple
import os, re, json
import numpy as np

try:
    from sentence_transformers import CrossEncoder
except Exception:
    CrossEncoder = None

# ===== Heuristics: nhận diện truy vấn về BẢNG / HÌNH =====
_TABLE_PAT = re.compile(r"\b(bảng|bang|table|biểu|biểu)\b", re.I | re.U)
_IMAGE_PAT = re.compile(r"\b(hình|hinh|ảnh|anh|figure|sơ đồ|so do|biểu đồ|biểu đồ)\b", re.I | re.U)

def infer_query_modality(query: str) -> Optional[str]:
    q = query or ""
    if _TABLE_PAT.search(q): return "table"
    if _IMAGE_PAT.search(q): return "image"
    return None

# ===== HTML utils cho bảng (khi có table_html_excerpt trong metadata) =====
_HTML_TAG = re.compile(r"<[^>]+>")
def strip_html_keep_space(html: str, max_chars: int = 1200) -> str:
    if not html: return ""
    txt = _HTML_TAG.sub(" ", html)
    txt = re.sub(r"\s+", " ", txt).strip()
    if max_chars and len(txt) > max_chars:
        txt = txt[:max_chars] + " ..."
    return txt

# ===== Chuẩn hoá điểm khoảng cách (Chroma trả distance: thấp = tốt) =====
def normalize_distance_to_similarity(distances: List[Optional[float]]) -> List[float]:
    vals = [d for d in distances if d is not None]
    if not vals:
        return [0.0] * len(distances)
    mn, mx = min(vals), max(vals)
    if mx <= mn + 1e-12:
        return [1.0] * len(distances)
    # map distance -> sim in [0,1], nhỏ (tốt) -> cao
    return [1.0 - ( (d - mn) / (mx - mn) ) if d is not None else 0.0 for d in distances]

# ===== Cross-Encoder Reranker =====
class CrossEncoderReranker:
    """
    Reranker dùng cross-encoder (HF), ví dụ:
      - BAAI/bge-reranker-v2-m3 (đa ngôn ngữ, nhanh) 
      - cross-encoder/ms-marco-MiniLM-L-6-v2 (phổ biến cho MS MARCO)
    """
    def __init__(self, model_name: Optional[str] = None, max_length: int = 512):
        self.model_name = model_name or os.getenv("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
        self.max_length = int(os.getenv("RERANK_MAX_LEN", str(max_length)))
        assert CrossEncoder is not None, "Bạn cần cài sentence-transformers để dùng CrossEncoder."
        self.model = CrossEncoder(self.model_name, max_length=self.max_length)

    def score(self, query: str, docs: List[str]) -> List[float]:
        pairs = [[query, d] for d in docs]
        # Một số cross-encoder (BGE) trả logits có thể dùng trực tiếp; nếu cần có thể sigmoid
        scores = self.model.predict(pairs, convert_to_numpy=True).tolist()
        return scores

# ===== Kết hợp điểm (fusion) =====
def fuse_scores(
    ce_scores: List[float],
    vec_sims: List[float],
    modality_prior: List[float],
    w_ce: float = 0.72,
    w_vec: float = 0.20,
    w_prior: float = 0.08,
) -> List[float]:
    # Chuẩn hoá về [0,1]
    def _minmax(x):
        if not x: return x
        mn, mx = min(x), max(x)
        if mx <= mn + 1e-12: return [1.0]*len(x)
        return [(v - mn)/(mx - mn) for v in x]
    ce_n = _minmax(ce_scores)
    vec_n = _minmax(vec_sims)
    pri_n = _minmax(modality_prior)
    return [w_ce*ce_n[i] + w_vec*vec_n[i] + w_prior*pri_n[i] for i in range(len(ce_scores))]

# ===== Chuẩn hoá văn bản đầu vào reranker (ưu ái bảng/hình) =====
def make_rerank_text(candidate: Dict[str, Any]) -> str:
    """
    candidate: 1 item từ bước retrieve() của bạn:
      {
        "text": "...",
        "distance": float | None,
        "citation": "...",
        "doc_id": "...",
        "van_ban": "...",
        "dieu": "...", "khoan": "...", "diem": "...",
        "page_start": ..., "page_end": ...,
        "modality": "table"|"image"|"text" (nếu có),
        "has_table_html": bool,
        ...
      }
    """
    txt = candidate.get("text") or ""
    md = candidate  # đã flatten
    # Nếu là BẢNG: thêm trích từ HTML (nếu có) để cross-encoder “thấy” tiêu đề cột/giá trị
    if (md.get("modality") == "table" or str(txt).lstrip().startswith("[BẢNG]")):
        html_ex = md.get("table_html_excerpt")
        if isinstance(html_ex, list) and html_ex:
            html_ex = html_ex[0]
        if isinstance(html_ex, str) and html_ex.strip():
            txt += "\n[BẢNG_HTML_TRÍCH] " + strip_html_keep_space(html_ex)
    # Nếu là HÌNH: ưu tiên caption (đã nằm trong text dạng [HÌNH] ...); không thêm ảnh nhúng
    # (Nếu meta có toạ độ/alt text bạn có thể nối thêm tuỳ ý.)
    return txt

# ===== Tính prior ưu tiên modality theo ý đồ truy vấn =====
def compute_modality_prior(
    query: str,
    cands: List[Dict[str, Any]],
    user_prefer_modality: Optional[str] = None
) -> List[float]:
    want = user_prefer_modality or infer_query_modality(query)
    out = []
    for c in cands:
        mod = (c.get("modality") or "").lower()
        if want and mod == want:
            out.append(1.0)
        elif want and want == "table" and str(c.get("text","")).lstrip().startswith("[BẢNG]"):
            out.append(0.9)
        elif want and want == "image" and str(c.get("text","")).lstrip().startswith("[HÌNH]"):
            out.append(0.9)
        else:
            out.append(0.0)
    return out

# ===== API chính để re-rank danh sách kết quả =====
def rerank_candidates(
    query: str,
    candidates: List[Dict[str, Any]],
    *,
    backend: str = "cross_encoder",
    model_name: Optional[str] = None,
    top_k_rerank: int = 20,
    final_k: int = 5,
    prefer_modality: Optional[str] = None,
    weights: Tuple[float, float, float] = (0.72, 0.20, 0.08),  # (w_ce, w_vec, w_prior)
) -> List[Dict[str, Any]]:
    """
    candidates: list kết quả từ bước retrieve (đã flatten).
    Trả về: top 'final_k' đã re-rank, kèm 'score_rerank', 'score_ce', 'score_vec', 'score_prior'.
    """
    if not candidates:
        return []

    # Lấy top_k_rerank đầu tiên từ retriever để tiết kiệm chi phí
    pool = candidates[:top_k_rerank]

    # Chuẩn bị văn bản cho cross-encoder
    docs_for_ce = [ make_rerank_text(c) for c in pool ]
    # Điểm vec-sim từ distance
    vec_sims = normalize_distance_to_similarity([ c.get("distance") for c in pool ])
    # Prior modality (bảng/hình)
    pri = compute_modality_prior(query, pool, user_prefer_modality=prefer_modality)

    if backend == "cross_encoder":
        rer = CrossEncoderReranker(model_name=model_name)
        ce_scores = rer.score(query, docs_for_ce)
    else:
        # No-op rerank (giữ nguyên) – fallback
        ce_scores = [0.0] * len(pool)

    w_ce, w_vec, w_prior = weights
    fused = fuse_scores(ce_scores, vec_sims, pri, w_ce, w_vec, w_prior)

    # Gắn điểm & sắp xếp
    for i, c in enumerate(pool):
        c["score_ce"] = ce_scores[i]
        c["score_vec"] = vec_sims[i]
        c["score_prior"] = pri[i]
        c["score_rerank"] = fused[i]

    pool.sort(key=lambda x: x["score_rerank"], reverse=True)
    return pool[:final_k]

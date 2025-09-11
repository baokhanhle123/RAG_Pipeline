# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import List, Dict, Any, Optional, Tuple
import os, re
from datetime import datetime, timezone

try:
    from sentence_transformers import CrossEncoder  # pip install sentence-transformers
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
    return [1.0 - ((d - mn) / (mx - mn)) if d is not None else 0.0 for d in distances]

# ===== Cross-Encoder (cache model toàn cục để tiết kiệm tải) =====
_CE_MODEL = None
def _get_ce_model(model_name: str, max_length: int):
    global _CE_MODEL
    if _CE_MODEL is None:
        if CrossEncoder is None:
            raise RuntimeError("Cần cài sentence-transformers để dùng CrossEncoder (pip install sentence-transformers).")
        _CE_MODEL = CrossEncoder(model_name, max_length=max_length)
    return _CE_MODEL

# ===== Chuẩn hoá văn bản đầu vào reranker (ưu ái bảng/hình) =====
def make_rerank_text(candidate: Dict[str, Any]) -> str:
    """
    candidate: 1 item từ bước retrieve() (đã flatten).
    """
    txt = candidate.get("text") or ""
    md = candidate
    # Nếu là BẢNG: thêm trích từ HTML (nếu có) để CE thấy tiêu đề cột/giá trị
    if (md.get("modality") == "table" or str(txt).lstrip().startswith("[BẢNG]")):
        html_ex = md.get("table_html_excerpt")
        if isinstance(html_ex, list) and html_ex:
            html_ex = html_ex[0]
        if isinstance(html_ex, str) and html_ex.strip():
            txt += "\n[BẢNG_HTML_TRÍCH] " + strip_html_keep_space(html_ex)
    return txt

# ===== Prior 1: ưu tiên modality theo ý đồ truy vấn =====
def compute_modality_prior(
    query: str,
    cands: List[Dict[str, Any]],
    user_prefer_modality: Optional[str] = None
) -> List[float]:
    want = (user_prefer_modality or infer_query_modality(query) or "").lower()
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

# ===== Prior 2: ưu tiên văn bản mới hơn (recency) nếu metadata có ngày =====
_DATE_KEYS = ["effective_date", "ngay_hieu_luc", "ban_hanh_date", "ngay_ban_hanh", "updated_at"]

def _parse_any_date(s: Any) -> Optional[datetime]:
    if not s: return None
    if isinstance(s, (int, float)):
        if s > 1e12: s = s / 1000.0
        try:
            return datetime.fromtimestamp(float(s), tz=timezone.utc)
        except Exception:
            return None
    ss = str(s).strip()
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%d/%m/%Y", "%d-%m-%Y"):
        try:
            dt = datetime.strptime(ss, fmt)
            return dt.replace(tzinfo=timezone.utc)
        except Exception:
            continue
    return None

def compute_recency_prior(cands: List[Dict[str, Any]]) -> List[float]:
    dates = []
    for c in cands:
        dt = None
        for k in _DATE_KEYS:
            dt = _parse_any_date(c.get(k))
            if dt: break
        dates.append(dt)
    vals = [int(d.timestamp()) if d else None for d in dates]
    nums = [v for v in vals if v is not None]
    if not nums:
        return [0.0]*len(cands)
    mn, mx = min(nums), max(nums)
    if mx <= mn:
        return [1.0 if v is not None else 0.0 for v in vals]
    return [ ((v - mn)/(mx - mn)) if v is not None else 0.0 for v in vals]

# ===== Prior 3: cohesion theo cùng 'van_ban' (khuyến khích nhóm nguồn nhất quán) =====
def compute_cohesion_prior(cands: List[Dict[str, Any]]) -> List[float]:
    counts = {}
    for c in cands:
        v = (c.get("van_ban") or "").strip()
        counts[v] = counts.get(v, 0) + 1
    return [ min(1.0, counts.get((c.get("van_ban") or "").strip(), 0) / 5.0) for c in cands ]

# ===== Giảm trùng: penalize khi trùng (doc_id, dieu, khoan, diem) =====
def apply_redundancy_penalty(cands: List[Dict[str, Any]], fused_scores: List[float]) -> List[float]:
    seen = set()
    out = list(fused_scores)
    for i, c in enumerate(cands):
        key = (c.get("doc_id"), c.get("dieu"), c.get("khoan"), c.get("diem"))
        if key in seen:
            out[i] *= 0.85
        else:
            seen.add(key)
    return out

# ===== Kết hợp điểm (fusion) =====
def _minmax(xs: List[float]) -> List[float]:
    if not xs: return xs
    mn, mx = min(xs), max(xs)
    if mx <= mn + 1e-12: return [1.0]*len(xs)
    return [(v - mn)/(mx - mn) for v in xs]

def fuse_scores(
    ce_scores: List[float],
    vec_sims: List[float],
    modality_prior: List[float],
    recency_prior: List[float],
    cohesion_prior: List[float],
    w_ce: float = 0.64,
    w_vec: float = 0.18,
    w_mod: float = 0.08,
    w_rec: float = 0.07,
    w_coh: float = 0.03,
) -> List[float]:
    ce_n  = _minmax(ce_scores)
    vec_n = _minmax(vec_sims)
    mod_n = _minmax(modality_prior)
    rec_n = _minmax(recency_prior)
    coh_n = _minmax(cohesion_prior)
    fused = [ w_ce*ce_n[i] + w_vec*vec_n[i] + w_mod*mod_n[i] + w_rec*rec_n[i] + w_coh*coh_n[i]
              for i in range(len(ce_scores)) ]
    return fused

# ===== API chính: rerank danh sách kết quả =====
def rerank_candidates(
    query: str,
    candidates: List[Dict[str, Any]],
    *,
    backend: str = "cross_encoder",  # hoặc "none"
    model_name: Optional[str] = None,
    max_length: int = 512,
    top_k_rerank: int = 30,
    final_k: int = 8,
    prefer_modality: Optional[str] = None,
    weights: Tuple[float, float, float, float, float] = (0.64, 0.18, 0.08, 0.07, 0.03),  # w_ce,w_vec,w_mod,w_rec,w_coh
) -> List[Dict[str, Any]]:
    if not candidates:
        return []

    pool = candidates[:min(top_k_rerank, len(candidates))]

    docs_for_ce = [ make_rerank_text(c) for c in pool ]
    vec_sims = normalize_distance_to_similarity([ c.get("distance") for c in pool ])
    pri_mod = compute_modality_prior(query, pool, user_prefer_modality=prefer_modality)
    pri_rec = compute_recency_prior(pool)
    pri_coh = compute_cohesion_prior(pool)

    if backend == "cross_encoder":
        model_name = model_name or os.getenv("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
        ce = _get_ce_model(model_name, max_length=max_length)
        ce_scores = ce.predict([[query, d] for d in docs_for_ce], convert_to_numpy=True).tolist()
    else:
        ce_scores = [0.0]*len(pool)

    w_ce, w_vec, w_mod, w_rec, w_coh = weights
    fused = fuse_scores(ce_scores, vec_sims, pri_mod, pri_rec, pri_coh, w_ce, w_vec, w_mod, w_rec, w_coh)
    fused = apply_redundancy_penalty(pool, fused)

    for i, c in enumerate(pool):
        c["score_ce"] = ce_scores[i]
        c["score_vec"] = vec_sims[i]
        c["score_prior_modality"] = pri_mod[i]
        c["score_prior_recency"] = pri_rec[i]
        c["score_prior_cohesion"] = pri_coh[i]
        c["score_rerank"] = fused[i]

    pool.sort(key=lambda x: x["score_rerank"], reverse=True)
    return pool[:final_k]

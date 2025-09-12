#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Prompt guard/normalizer for Vietnamese PCCC (Phòng cháy chữa cháy).

Chức năng:
- Sửa lỗi chính tả phổ biến tiếng Việt liên quan PCCC (ví dụ: "biện phòng" -> "biện pháp", "bom" -> "bơm", ...).
- Phân loại nhanh in-domain (PCCC) vs. off-topic (ăn chay, chứng khoán, ...).
- Gợi ý lại câu hỏi đúng domain nếu lệch; trả về: status, normalized_query, suggestions, reason.
- Tuỳ chọn kiểm tra/bồi dưỡng thêm bằng LLM: `PROMPT_GUARD_USE_LLM=1`.

ENV:
- OPENAI_API_KEY (khi bật PROMPT_GUARD_USE_LLM=1)
- CHAT_MODEL (mặc định: gpt-4o-mini)
"""

from __future__ import annotations
from typing import Dict, Any, Optional, List, Tuple
import os
import re

try:
    from openai import OpenAI
except Exception:
    OpenAI = None  # cho phép chạy guard không cần OpenAI

CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")

# ---------- Token utils ----------
try:
    import tiktoken
    _ENC = tiktoken.get_encoding("cl100k_base")
except Exception:
    _ENC = None

def _tok_len(s: str) -> int:
    if _ENC is None: return max(1, len(s)//2)
    return len(_ENC.encode(s or ""))

def truncate_tokens(s: str, max_tokens: int) -> str:
    if _tok_len(s) <= max_tokens: return s
    if _ENC is None: return s[: max_tokens*2]
    ids = _ENC.encode(s or "")
    return _ENC.decode(ids[:max_tokens])

def get_openai_client():
    key = os.getenv("OPENAI_API_KEY", "").strip()
    assert key, "Thiếu OPENAI_API_KEY"
    return OpenAI(api_key=key)

# ---------- Chuẩn hoá lỗi phổ biến ----------
# (pattern, replacement). Thứ tự quan trọng: các chuẩn hoá thô trước, cụ thể sau.
_COMMON_FIXES: List[Tuple[re.Pattern, str]] = [
    # khoảng trắng
    (re.compile(r"\s+", re.U), " "),
    # dấu tiếng Việt thường nhầm
    (re.compile(r"\bP\s*C\s*C\s*C\b", re.I | re.U), "PCCC"),
    (re.compile(r"\bC\s*N\s*C\s*H\b", re.I | re.U), "CNCH"),
    (re.compile(r"\bph[oòóỏõọ]ng\s*ch[aáàảãạ]y\b", re.I | re.U), "phòng cháy"),
    (re.compile(r"\bch[uư]a\s*ch[aáàảãạ]y\b", re.I | re.U), "chữa cháy"),
    (re.compile(r"\bph[oòóỏõọ]ng\s*ch[áa]y\s*ch[uư][aăâ]?\s*ch[aáàảãạ]y\b", re.I | re.U), "phòng cháy chữa cháy"),
    (re.compile(r"\bbi[eẹ]n\s*ph[oòóỏõọ]ng\b", re.I | re.U), "biện pháp phòng"),
    (re.compile(r"\bb[ií]e?n\b", re.I | re.U), "biện"),
    (re.compile(r"\bbom\b", re.I | re.U), "bơm"),
    (re.compile(r"\bchua\b", re.I | re.U), "chữa"),
    (re.compile(r"\bchu[aăâ]y\b", re.I | re.U), "chữa"),
    (re.compile(r"\bso\s*do\b", re.I | re.U), "sơ đồ"),
    (re.compile(r"\bth[uo]ng\s*t[uư]\b", re.I | re.U), "thông tư"),
    (re.compile(r"\bngh[iị]?\s*đ[iị]nh\b", re.I | re.U), "nghị định"),
    (re.compile(r"\bquy\s*ch[uâ]n\b", re.I | re.U), "quy chuẩn"),
    (re.compile(r"\bti[eê]u\s*ch[uâ]n\b", re.I | re.U), "tiêu chuẩn"),
    # chuẩn hoá dấu câu
    (re.compile(r"\s*,\s*"), ", "),
    (re.compile(r"\s*:\s*"), ": "),
    (re.compile(r"\s*;\s*"), "; "),
]

_DOMAIN_KEYWORDS = [
    "pccc", "phòng cháy", "chữa cháy", "cnch", "cứu nạn", "cứu hộ",
    "nghị định", "thông tư", "quy chuẩn", "tiêu chuẩn",
    "luật pccc", "điều", "khoản", "điểm", "phương án chữa cháy",
    "phương tiện chữa cháy", "máy bơm chữa cháy", "hệ thống chữa cháy",
]

_OFFTOPIC_HINTS = [
    "ăn chay", "chế độ ăn", "nấu ăn", "thời trang", "chứng khoán",
    "bóng đá", "du lịch", "tình yêu", "nhạc", "phim ảnh",
]

def normalize_typos(query: str) -> str:
    q = (query or "").strip()
    if not q:
        return q
    out = q
    for pat, rep in _COMMON_FIXES:
        out = pat.sub(rep, out)
    out = re.sub(r"\s+", " ", out).strip()
    # sửa các cụm hay gặp kiểu “biện phòng” -> “biện pháp phòng”
    out = re.sub(r"\bbiện\s*phòng\b", "biện pháp phòng", out, flags=re.I | re.U)
    # thống nhất chữ thường ngoại trừ từ viết tắt
    return out

def is_pccc_intent(query: str) -> bool:
    q = (query or "").lower()
    score = 0
    for kw in _DOMAIN_KEYWORDS:
        if kw in q: score += 1
    for hint in _OFFTOPIC_HINTS:
        if hint in q: score -= 1
    # tối thiểu 1 tín hiệu domain và không bị off-topic lấn át
    return score >= 1

def _llm_intent_and_rewrite(query: str) -> Optional[Dict[str, Any]]:
    if os.getenv("PROMPT_GUARD_USE_LLM", "0") != "1":
        return None
    if OpenAI is None:
        return None
    try:
        client = get_openai_client()
        prompt = (
            "Bạn là bộ lọc cho trợ lý pháp lý PCCC (Việt Nam).\n"
            "- Quyết định câu hỏi có thuộc PCCC không (is_pccc: true/false).\n"
            "- Sửa lỗi chính tả nhẹ, rút gọn, giữ đúng nghĩa về PCCC.\n"
            '- Trả JSON: {"is_pccc": bool, "normalized": str, "suggestions": [str]}.\n'
            f"Câu hỏi: {query}"
        )
        resp = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[{"role": "user", "content": truncate_tokens(prompt, 3000)}],
            temperature=0,
            max_tokens=256,
        )
        txt = resp.choices[0].message.content.strip()
        m = re.search(r"\{[\s\S]*\}$", txt)
        if not m:
            return None
        import json
        data = json.loads(m.group(0))
        return {
            "is_pccc": bool(data.get("is_pccc", False)),
            "normalized": (data.get("normalized") or query).strip(),
            "suggestions": [str(x) for x in (data.get("suggestions") or []) if str(x).strip()],
        }
    except Exception:
        return None

def guard_prompt(query: str) -> Dict[str, Any]:
    """
    status: "ok" | "off_topic" | "empty"
    """
    q0 = (query or "").strip()
    if not q0:
        return {
            "status": "empty",
            "normalized_query": "",
            "suggestions": [
                "Bạn muốn hỏi quy định PCCC nào? Ví dụ: 'Các biện pháp PCCC cho chung cư'.",
                "Hoặc: 'Yêu cầu kỹ thuật đối với máy bơm chữa cháy theo QCVN/TCVN?'.",
            ],
            "reason": "empty",
        }

    q1 = normalize_typos(q0)

    # LLM assist (tùy chọn)
    llm = _llm_intent_and_rewrite(q1)
    if llm is not None:
        q1 = llm.get("normalized") or q1
        if not llm.get("is_pccc", False):
            sug = llm.get("suggestions") or [
                "Vui lòng hỏi về lĩnh vực PCCC. Ví dụ: 'Nêu các biện pháp phòng cháy chữa cháy tại cơ sở'."
            ]
            return {
                "status": "off_topic",
                "normalized_query": q1,
                "suggestions": sug,
                "reason": "llm_off_topic",
            }

    # Heuristic intent
    if not is_pccc_intent(q1):
        return {
            "status": "off_topic",
            "normalized_query": q1,
            "suggestions": [
                "Câu hỏi chưa thuộc PCCC. Ví dụ đúng: 'Nêu các biện pháp phòng cháy chữa cháy cho nhà xưởng'.",
                "Hoặc: 'Quy định về phương án chữa cháy theo Nghị định/Thông tư nào?'.",
            ],
            "reason": "heuristic_off_topic",
        }

    return {
        "status": "ok",
        "normalized_query": q1,
        "suggestions": [],
        "reason": "",
    }

__all__ = ["guard_prompt", "normalize_typos", "is_pccc_intent"]
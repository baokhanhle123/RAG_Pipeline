#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Prompt guard/normalizer for Vietnamese PCCC (Phòng cháy chữa cháy) assistant.

Mục tiêu:
- Sửa chính tả nhẹ & chuẩn hoá query.
- Phát hiện và chặn prompt injection / topic hijack.
- Phân tách multi-intent, chỉ giữ lại phần thuộc miền PCCC.
- Sinh cảnh báo (user_warning) để thông báo người dùng khi có nội dung bị loại bỏ.

Trả về:
{
  "status": "ok" | "off_topic" | "empty",
  "normalized_query": str,
  "suggestions": [str],
  "reason": str,
  "removed_segments": [{"text": "...", "label": "injection|off_topic|sensitive_hijack"}],
  "user_warning": str | ""        # cảnh báo ngắn gọn cho người dùng
}

ENV:
- PROMPT_GUARD_USE_LLM=1 (tuỳ chọn) để gọi LLM hỗ trợ sửa câu.
- PROMPT_GUARD_NOTIFY=1 (mặc định 1): có sinh user_warning cho người dùng.
"""

from __future__ import annotations
from typing import Dict, Any, Optional, List, Tuple
import os
import re
from openai import OpenAI

CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")

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

def get_openai_client() -> OpenAI:
    key = os.getenv("OPENAI_API_KEY", "").strip()
    assert key, "Thiếu OPENAI_API_KEY"
    return OpenAI(api_key=key)

# -------------------------------
# 1) Chuẩn hoá chính tả đơn giản
# -------------------------------
_COMMON_FIXES: List[tuple[re.Pattern, str]] = [
    (re.compile(r"\s+", re.U), " "),  # collapse spaces
    # common PCCC terms
    (re.compile(r"\bbị?ệ?n\b", re.I | re.U), "biện"),
    (re.compile(r"\bphòng\s*cháy\s*chữa\s*cháy\b", re.I | re.U), "phòng cháy chữa cháy"),
    (re.compile(r"\bphòng\s*chấy\b", re.I | re.U), "phòng cháy"),
    (re.compile(r"\bchửa\s*cháy\b", re.I | re.U), "chữa cháy"),
    (re.compile(r"\bP\s*C\s*C\s*C\b", re.I | re.U), "PCCC"),
    (re.compile(r"\bCNCH\b", re.I | re.U), "CNCH"),
    # punctuation spacing
    (re.compile(r"\s*,\s*"), ", "),
    (re.compile(r"\s*:\s*"), ": "),
]

_DOMAIN_KEYWORDS = [
    "pccc", "phòng cháy", "chữa cháy", "cnch", "cứu nạn", "cứu hộ",
    "nghị định", "thông tư", "quy chuẩn", "tiêu chuẩn", "điều", "khoản", "điểm",
    "phương án chữa cháy", "phương tiện chữa cháy", "phương án cứu nạn",
]

_OFFTOPIC_HINTS = [
    "ăn chay", "nấu ăn", "thời trang", "chứng khoán", "bóng đá", "du lịch",
    "chu vi", "diện tích", "1 + 1", "1+1",
]

# -------------------------------
# 2) Mẫu tấn công injection / hijack
# -------------------------------
_PAT_OVERRIDE = re.compile(
    r"(bỏ qua|ignore|quên hết|vô hiệu hóa|disable|override).{0,40}(hướng dẫn|ràng buộc|luật|quy tắc|guard|policy)",
    re.I | re.U
)
_PAT_ROLEPLAY = re.compile(
    r"(đóng vai|hóa thân|pretend|act as).{0,40}(thầy|cô|giáo viên|kỹ sư|luật sư|bác sĩ|chuyên gia)",
    re.I | re.U
)
_PAT_OWNERSHIP_Q = re.compile(r"\b(của\s*nước\s*nào|thuộc\s*nước\s*nào)\b", re.I | re.U)
_PAT_SENSITIVE_TERMS = re.compile(
    r"\b(hoàng\s*sa|trường\s*sa|paracel|spratly|biển\s*đông|đường\s*lưỡi\s*bò)\b",
    re.I | re.U
)

def normalize_typos(query: str) -> str:
    q = (query or "").strip()
    if not q: return q
    out = q
    for pat, rep in _COMMON_FIXES:
        out = pat.sub(rep, out)
    return re.sub(r"\s+", " ", out).strip()

# -------------------------------
# 3) Phân tách multi-intent
# -------------------------------
_SENT_SPLIT = re.compile(r"[?\n]+|(?<=[\.\!\;])\s+")
def _SPLIT_SENTENCES(q: str) -> List[str]:
    parts = _SENT_SPLIT.split(q.strip())
    return [p for p in parts if p.strip()]

def is_pccc_intent(text: str) -> bool:
    t = (text or "").lower()
    score = 0
    for kw in _DOMAIN_KEYWORDS:
        if kw in t:
            score += 1
    for hint in _OFFTOPIC_HINTS:
        if hint in t:
            score -= 1
    return score >= 1

# -------------------------------
# 4) Gắn nhãn từng segment
# -------------------------------
def label_segment(seg: str) -> Tuple[str, str]:
    """
    Return (label, reason)
    label in {"in_domain","off_topic","injection","sensitive_hijack"}
    """
    s = seg.strip()

    # Injection: override/roleplay
    if _PAT_OVERRIDE.search(s) or _PAT_ROLEPLAY.search(s):
        return "injection", "override_or_roleplay"

    # Sensitive hijack: sở hữu/lãnh thổ + từ nhạy cảm
    if _PAT_OWNERSHIP_Q.search(s) and _PAT_SENSITIVE_TERMS.search(s):
        return "sensitive_hijack", "national_ownership_sensitive"

    # Off-topic nếu không phải PCCC
    if not is_pccc_intent(s):
        return "off_topic", "not_pccc"

    return "in_domain", ""

# -------------------------------
# 5) LLM phụ trợ (tuỳ chọn)
# -------------------------------
def _llm_intent_and_rewrite(query: str) -> Optional[Dict[str, Any]]:
    if os.getenv("PROMPT_GUARD_USE_LLM", "0") != "1":
        return None
    try:
        client = get_openai_client()
        prompt = (
            "Bạn là bộ lọc câu hỏi cho trợ lý pháp lý PCCC (Việt Nam).\n"
            "- Xác định các câu PCCC và sửa lỗi chính tả nhẹ.\n"
            '- Trả về JSON {"normalized": str}.\n'
            f"Truy vấn: {query}"
        )
        resp = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[{"role":"user","content": truncate_tokens(prompt, 3000)}],
            temperature=0,
            max_tokens=200,
        )
        txt = (resp.choices[0].message.content or "").strip()
        m = re.search(r"\{[\s\S]*\}$", txt)
        if not m: return None
        import json
        data = json.loads(m.group(0))
        return {"normalized": str(data.get("normalized") or query).strip()}
    except Exception:
        return None

# -------------------------------
# 6) Tạo cảnh báo cho người dùng
# -------------------------------
def _summarize_removed(removed: List[Dict[str,str]], max_items: int = 2, max_chars: int = 90) -> str:
    if not removed: return ""
    # Ưu tiên hiển thị injection/hijack trước
    priority = {"sensitive_hijack": 0, "injection": 1, "off_topic": 2}
    removed_sorted = sorted(removed, key=lambda r: priority.get(r.get("label","off_topic"), 9))
    picks = removed_sorted[:max_items]
    parts = []
    for r in picks:
        txt = r.get("text","").strip().replace("\n"," ")
        if len(txt) > max_chars: txt = txt[:max_chars].rstrip() + " ..."
        lab = r.get("label","off_topic")
        parts.append(f"- ({lab}) {txt}")
    more = ""
    if len(removed) > len(picks):
        more = f"\n- ... và {len(removed) - len(picks)} phần khác"
    return ("Đã loại bỏ một số nội dung ngoài phạm vi/tiềm ẩn 'prompt injection' để đảm bảo trả lời đúng chủ đề PCCC:\n"
            + "\n".join(parts) + more)

# -------------------------------
# 7) API chính
# -------------------------------
def guard_prompt(query: str) -> Dict[str, Any]:
    q0 = (query or "").strip()
    if not q0:
        return {
            "status": "empty",
            "normalized_query": "",
            "suggestions": [
                "Bạn muốn hỏi quy định PCCC nào? Ví dụ: 'Các biện pháp PCCC cho chung cư'",
            ],
            "reason": "empty",
            "removed_segments": [],
            "user_warning": "",
        }

    q1 = normalize_typos(q0)

    segs = _SPLIT_SENTENCES(q1)
    kept: List[str] = []
    removed: List[Dict[str,str]] = []

    for seg in segs:
        label, why = label_segment(seg)
        if label == "in_domain":
            kept.append(seg.strip())
        else:
            removed.append({"text": seg.strip(), "label": label})

    # LLM phụ trợ (tuỳ chọn)
    if kept:
        draft = " ".join(kept).strip()
        aux = _llm_intent_and_rewrite(draft)
        if aux and aux.get("normalized"):
            draft = aux["normalized"]
        normalized = draft
    else:
        normalized = ""

    notify = (os.getenv("PROMPT_GUARD_NOTIFY", "1") == "1")
    user_warning = _summarize_removed(removed) if (removed and notify) else ""

    if not normalized:
        hints = [
            "Vui lòng đặt câu hỏi liên quan PCCC. Ví dụ: 'Không xuất trình hồ sơ PCCC khi kiểm tra bị phạt bao nhiêu tiền?'",
            "Hoặc: 'Nêu các biện pháp PCCC cho nhà xưởng'.",
        ]
        if user_warning:
            hints.insert(0, "Một số nội dung đã bị loại bỏ do ngoài phạm vi/tiềm ẩn 'prompt injection'.")
        return {
            "status": "off_topic",
            "normalized_query": "",
            "suggestions": hints,
            "reason": "no_in_domain_after_filter",
            "removed_segments": removed,
            "user_warning": user_warning,
        }

    return {
        "status": "ok",
        "normalized_query": normalized,
        "suggestions": (["Một số nội dung ngoài chủ đề đã được loại bỏ."] if user_warning else []),
        "reason": "ok",
        "removed_segments": removed,
        "user_warning": user_warning,
    }

__all__ = [
    "guard_prompt", "normalize_typos", "is_pccc_intent",
]

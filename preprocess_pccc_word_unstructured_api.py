#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Tiền xử lý Word (.doc/.docx) văn bản luật PCCC bằng Unstructured API.
- Gửi file qua multipart/form-data
- Nhận list element (Title/NarrativeText/ListItem/Table/... + metadata)
- Gắn thêm metadata pháp lý (chuong/muc/dieu/khoan/diem/phu_luc) để phục vụ trích dẫn
- Xuất JSONL (1 dòng / element) cho pipeline RAG

Yêu cầu:
  pip install requests python-dateutil
Biến môi trường:
  UNSTRUCTURED_API_KEY: API key
    : mặc định dùng free endpoint https://api.unstructured.io/general/v0/general
Tham khảo API & tham số: docs.unstructured.io (partition endpoint, api parameters)

export UNSTRUCTURED_API_URL="https://api.unstructuredapp.io/general/v0/general"
export UNSTRUCTURED_API_KEY=""
"""

from __future__ import annotations
import os, re, json, time, uuid, hashlib, mimetypes
from pathlib import Path
from typing import Dict, List, Any, Iterable, Optional, Tuple
import requests

# =========================
# Cấu hình & tiện ích chung
# =========================

API_URL = os.getenv("UNSTRUCTURED_API_URL", "").strip()
API_KEY = os.getenv("UNSTRUCTURED_API_KEY", "").strip()

# MIME cho Word
MIME_BY_EXT = {
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".doc":  "application/msword",
}

HEADERS = {
    "accept": "application/json",
    # CHUẨN ĐÚNG: Không dùng Authorization Bearer
    "unstructured-api-key": API_KEY if API_KEY else "",
}

def sha1_of_file(p: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha1()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()

def guess_mime(path: Path) -> str:
    return MIME_BY_EXT.get(path.suffix.lower(), mimetypes.guess_type(str(path))[0] or "application/octet-stream")

# ================
# Gọi Unstructured
# ================

def partition_word_via_unstructured_api(
    file_path: Path,
    languages: List[str] = ["vie", "eng"],
    include_page_breaks: bool = True,
    coordinates: bool = True,
    encoding: str = "utf-8",
    timeout: int = 120,
    max_retries: int = 3,
    backoff_sec: float = 2.0,
) -> List[Dict[str, Any]]:
    assert API_KEY, (
        "Thiếu UNSTRUCTURED_API_KEY. Hãy `export UNSTRUCTURED_API_KEY=...` "
        "và đảm bảo header 'unstructured-api-key' được set."
    )
    assert API_URL.startswith("http"), (
        "Thiếu/không hợp lệ UNSTRUCTURED_API_URL. Ví dụ Free: "
        "https://api.unstructured.io/general/v0/general; "
        "Starter/Team: URL hiển thị trong tài khoản."
    )
    assert file_path.exists(), f"Không thấy file: {file_path}"

    data = {
        "languages": json.dumps(languages),          # multipart: truyền list dưới dạng JSON string
        "include_page_breaks": str(include_page_breaks).lower(),
        "coordinates": str(coordinates).lower(),
        "encoding": encoding,
        "output_format": "application/json",
    }

    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            mime = guess_mime(file_path)
            # QUAN TRỌNG: Mở lại file mỗi lần retry để không bị EOF
            with file_path.open("rb") as f:
                files = {"files": (file_path.name, f, mime)}
                resp = requests.post(API_URL, headers=HEADERS, data=data, files=files, timeout=timeout)

            if resp.status_code == 200:
                out = resp.json()
                if isinstance(out, dict) and "elements" in out:
                    return out["elements"] or []
                if isinstance(out, list):
                    return out
                return out if isinstance(out, list) else []

            # Gợi ý chẩn đoán thông minh cho 401/403
            if resp.status_code in (401, 403):
                msg = resp.text
                hint = []
                if "API key is missing" in msg or "missing" in msg.lower():
                    hint.append("Header phải là 'unstructured-api-key', không phải 'Authorization'.")
                hint.append("Kiểm tra cặp URL–Key: Free dùng api.unstructured.io; Starter/Team dùng URL .app.io (xem account).")
                raise RuntimeError(f"Auth lỗi {resp.status_code}: {msg[:300]} | Gợi ý: " + " ".join(hint))

            if resp.status_code in (429, 500, 502, 503, 504):
                last_err = RuntimeError(f"{resp.status_code} {resp.text[:300]}")
                time.sleep(backoff_sec * attempt)
                continue

            raise RuntimeError(f"API lỗi {resp.status_code}: {resp.text[:1000]}")

        except requests.RequestException as e:
            last_err = e
            time.sleep(backoff_sec * attempt)
            continue

    raise last_err or RuntimeError("Partition failed without explicit error.")

# ===========================
# Chuẩn hoá & trích xuất luật
# ===========================

_re_chuong  = re.compile(r"^\s*Chương\s+([IVXLCDM]+)\b", re.IGNORECASE)
_re_muc     = re.compile(r"^\s*Mục\s+([IVXLCDM]+)\b", re.IGNORECASE)
_re_dieu    = re.compile(r"^\s*Điều\s+(\d+[A-Za-z]*)\b", re.IGNORECASE)
_re_khoan   = re.compile(r"^\s*Khoản\s+(\d+)\b", re.IGNORECASE)
_re_diem    = re.compile(r"^\s*([a-zA-Z])\)", re.IGNORECASE)  # "a) b) c)"
_re_phuluc  = re.compile(r"^\s*Phụ\s*lục\s*([A-Z0-9\-]+)?\b", re.IGNORECASE)

# Nhận diện loại văn bản & số hiệu (cơ bản, có thể mở rộng)
_re_loai_vb = re.compile(
    r"\b(?:Luật|Nghị\s*định|Thông\s*tư|QCVN|TCVN)\b[^\n]*", re.IGNORECASE
)

def enrich_hierarchy(elements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Duyệt tuần tự element để gắn trạng thái Chương/Mục/Điều/Khoản/Điểm/Phụ lục.
    Đồng thời suy đoán 'van_ban' (tên văn bản/số hiệu) từ các Title/Heading/NarrativeText đầu tài liệu.
    """
    chuong = muc = dieu = khoan = diem = phu_luc = None
    van_ban: Optional[str] = None

    # Thử suy đoán văn bản ở 50 element đầu
    head_text = []
    for el in elements[:50]:
        t = (el.get("text") or "").strip()
        if t:
            head_text.append(t)
    joined = "\n".join(head_text)
    m = _re_loai_vb.search(joined)
    if m:
        van_ban = m.group(0).strip()

    out: List[Dict[str, Any]] = []
    for el in elements:
        text = (el.get("text") or "").strip()
        if not text:
            # vẫn giữ lại (ví dụ PageBreak) để bảo toàn số trang
            pass

        # cập nhật mức tiêu đề nếu khớp
        if _re_chuong.match(text):
            chuong, muc, dieu, khoan, diem = _re_chuong.match(text).group(1), None, None, None, None
        elif _re_muc.match(text):
            muc, dieu, khoan, diem = _re_muc.match(text).group(1), None, None, None
        elif _re_dieu.match(text):
            dieu, khoan, diem = _re_dieu.match(text).group(1), None, None
        elif _re_khoan.match(text):
            khoan, diem = _re_khoan.match(text).group(1), None
        elif _re_diem.match(text):
            diem = _re_diem.match(text).group(1)
        elif _re_phuluc.match(text):
            phu_luc = _re_phuluc.match(text).group(1) or ""

        meta = el.get("metadata") or {}
        # gắn hierarchy vào metadata mới
        meta_enriched = {
            **meta,
            "chuong": chuong,
            "muc": muc,
            "dieu": dieu,
            "khoan": khoan,
            "diem": diem,
            "phu_luc": phu_luc,
            "van_ban": van_ban,
        }

        # tạo bản ghi chuẩn hoá
        normalized = {
            "doc_id": meta.get("filename") or "",
            "source_sha1": meta.get("sha256") or "",  # API đôi khi cung cấp sha256; nếu không có sẽ gán ở ngoài
            "element_id": el.get("id") or str(uuid.uuid4()),
            "type": el.get("type"),
            "text": text,
            "metadata": meta_enriched,
        }
        out.append(normalized)
    return out

def attach_file_level_metadata(
    normalized: List[Dict[str, Any]],
    file_path: Path,
    source_sha1: str,
) -> List[Dict[str, Any]]:
    for row in normalized:
        row["doc_id"] = file_path.name
        if not row.get("source_sha1"):
            row["source_sha1"] = source_sha1
        # đảm bảo có page_number nếu API đã chèn PageBreak
        # Unstructured sẽ tạo element type="PageBreak" với metadata.page_number tăng dần nếu bật include_page_breaks
        # (khả dụng cho một số định dạng; với .docx có thể không luôn có trang)
    return normalized

def write_jsonl(records: Iterable[Dict[str, Any]], out_path: Path) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    return n

# =====================
# Pipeline xử lý thư mục
# =====================

def preprocess_word_folder(
    inputs: List[Path],
    out_jsonl: Path,
    languages: List[str] = ["vie","eng"],
) -> int:
    """
    Xử lý danh sách file .doc/.docx → out_jsonl (append).
    """
    total = 0
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    for p in inputs:
        if not p.exists():
            print(f"⚠️  Bỏ qua (không tồn tại): {p}")
            continue
        if p.suffix.lower() not in (".doc", ".docx"):
            print(f"⚠️  Bỏ qua (không phải Word): {p.name}")
            continue

        print(f"🔹 Partition: {p.name}")
        elements = partition_word_via_unstructured_api(
            p,
            languages=languages,
            include_page_breaks=True,
            coordinates=True,
            encoding="utf-8",
        )
        source_sha1 = sha1_of_file(p)
        normalized = enrich_hierarchy(elements)
        normalized = attach_file_level_metadata(normalized, p, source_sha1)

        with out_jsonl.open("a", encoding="utf-8") as f:
            for rec in normalized:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                total += 1

    print(f"✅ Đã ghi {total} element → {out_jsonl}")
    return total

# ============
# Ví dụ chạy thử
# ============

if __name__ == "__main__":
    # Thay đổi đường dẫn theo môi trường của bạn
    # (Các file mẫu user đã upload ở /mnt/data)
    sample_paths = [
        Path("dataset/Luật PCCC 2024 55_2024_QH15_621347.doc"),
        Path("dataset/Nghị định 105-2025-NĐ-CP ngày 15-05-2025 hướng dẫn Luật Phòng cháy, chữa cháy và cứu nạn, cứu hộ.doc"),
        Path("dataset/QCVN 03 2023 BCA về Phương tiện phòng cháy và chữa cháy_VN.doc"),
        Path("dataset/B.1. Bảng đối chiếu quy hoạch.docx"),
    ]
    out = Path("./pccc_word_elements.jsonl")
    preprocess_word_folder(sample_paths, out)

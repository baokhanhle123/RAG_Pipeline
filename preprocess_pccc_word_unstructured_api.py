#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Tiền xử lý Word (DOC/DOCX) bằng Unstructured API.
- Tự động thêm file mới trong thư mục vào pccc_word_elements.jsonl
- Phát hiện trùng file theo nội dung (SHA-1), in log và bỏ qua nếu đã có

Yêu cầu:
  pip install requests python-dateutil

Biến môi trường:
  UNSTRUCTURED_API_URL  (mặc định: https://api.unstructuredapp.io/general/v0/general)
  UNSTRUCTURED_API_KEY  (bắt buộc)

Tham khảo Unstructured Partition Endpoint:
- URL & header 'unstructured-api-key'  ➜ docs Overview & Quickstart
- Tham số 'files' (bắt buộc), 'include_page_breaks', 'languages', 'encoding' ➜ API parameters
"""

from __future__ import annotations

import argparse
import hashlib
import json
import mimetypes
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import requests
from requests.adapters import HTTPAdapter, Retry


# ============== Cấu hình mặc định ==============

DEFAULT_API_URL = os.getenv("UNSTRUCTURED_API_URL", "https://api.unstructuredapp.io/general/v0/general").strip()
API_KEY_ENV = "UNSTRUCTURED_API_KEY"

# Unstructured chấp nhận form-data:
#  - files=<file>
#  - include_page_breaks=true/false
#  - languages=["vie"] (chuỗi JSON)
#  - encoding="utf-8"
# Tham chiếu: Partition Endpoint overview & parameters
# https://docs.unstructured.io/api-reference/partition/overview
# https://docs.unstructured.io/api-reference/partition/api-parameters

ALLOWED_EXTS = {".doc", ".docx"}

# ============== Kiểu dữ liệu tiện ích ==============

@dataclass
class ApiConfig:
    url: str
    api_key: str
    include_page_breaks: bool = True
    encoding: str = "utf-8"
    languages: List[str] = None  # ví dụ ["vie"]

    def to_form(self) -> Dict[str, str]:
        data = {
            "include_page_breaks": "true" if self.include_page_breaks else "false",
            "encoding": self.encoding,
        }
        if self.languages:
            data["languages"] = json.dumps(self.languages, ensure_ascii=False)
        return data


# ============== Tiện ích I/O JSONL ==============

def append_jsonl(path: Path, rows: Iterable[dict]) -> int:
    """Append các dòng JSON vào file JSONL (tạo nếu chưa có)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def iter_jsonl(path: Path) -> Iterable[dict]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                # bỏ dòng hỏng (log cảnh báo nhẹ)
                sys.stderr.write(f"[WARN] Bỏ qua dòng JSONL lỗi tại {path}\n")
                continue


# ============== Tính hash & lọc trùng ==============

def sha1_file(path: Path, bufsize: int = 1 << 20) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        while True:
            b = f.read(bufsize)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def scan_word_files(in_dir: Path) -> List[Path]:
    files = []
    for p in sorted(in_dir.rglob("*")):
        if p.is_file() and p.suffix.lower() in ALLOWED_EXTS:
            files.append(p)
    return files


def get_mime_for_word(path: Path) -> str:
    # .docx: application/vnd.openxmlformats-officedocument.wordprocessingml.document
    # .doc : application/msword
    if path.suffix.lower() == ".docx":
        return "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    if path.suffix.lower() == ".doc":
        return "application/msword"
    # fallback
    mt, _ = mimetypes.guess_type(str(path))
    return mt or "application/octet-stream"


def collect_existing_hashes(out_jsonl: Path) -> Dict[str, int]:
    """Lấy set SHA-1 đã có trong JSONL (tính theo 'source_sha1')."""
    seen = {}
    if not out_jsonl.exists():
        return seen
    for i, row in enumerate(iter_jsonl(out_jsonl), 1):
        src_sha1 = (row.get("source_sha1") or
                    (row.get("metadata") or {}).get("source_sha1"))
        if src_sha1:
            seen[src_sha1] = seen.get(src_sha1, 0) + 1
    return seen


def find_duplicates_in_batch(paths: List[Path]) -> Dict[str, List[Path]]:
    """Nhóm các file trong thư mục có SHA-1 trùng nhau."""
    groups: Dict[str, List[Path]] = {}
    for p in paths:
        try:
            h = sha1_file(p)
        except Exception as e:
            sys.stderr.write(f"[WARN] Không tính được SHA-1: {p} ({e})\n")
            continue
        groups.setdefault(h, []).append(p)
    return {h: lst for h, lst in groups.items() if len(lst) > 1}


# ============== Unstructured API ==============

def make_session() -> requests.Session:
    sess = requests.Session()
    retries = Retry(
        total=5,
        backoff_factor=1.2,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["POST"]),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retries)
    sess.mount("http://", adapter)
    sess.mount("https://", adapter)
    return sess


def partition_via_unstructured_api(
    path: Path,
    cfg: ApiConfig,
    timeout: int = 120,
) -> List[dict]:
    """
    Gọi Partition Endpoint cho 1 file.
    - Header: 'unstructured-api-key' (theo Quickstart/Overview)
    - Form data chính: files, languages, encoding, include_page_breaks
    Trả về: list element (JSON)
    """
    assert cfg.api_key, "Thiếu UNSTRUCTURED_API_KEY"

    mime = get_mime_for_word(path)
    files = {
        "files": (path.name, path.open("rb"), mime),
    }
    data = cfg.to_form()

    headers = {
        "accept": "application/json",
        "unstructured-api-key": cfg.api_key,  # theo tài liệu Quickstart
    }

    sess = make_session()
    resp = sess.post(cfg.url, headers=headers, files=files, data=data, timeout=timeout)

    if resp.status_code != 200:
        # In lỗi chi tiết để dễ debug (401: thiếu key; 4xx khác: tham số/định dạng)
        snippet = resp.text[:1000]
        raise RuntimeError(f"Unstructured API lỗi {resp.status_code}: {snippet}")

    try:
        data = resp.json()
    except Exception as e:
        raise RuntimeError(f"Phản hồi API không phải JSON: {e}")

    if not isinstance(data, list):
        raise RuntimeError("Phản hồi API không phải danh sách element.")

    return data


# ============== Chuẩn hoá kết quả để ghi JSONL ==============

def elements_to_jsonl_rows(
    elements: List[dict],
    src_path: Path,
    src_sha1: str,
    doc_id: Optional[str] = None,
    extra_meta: Optional[Dict] = None,
) -> List[dict]:
    """
    Mỗi element sẽ ghi 1 dòng JSONL, giữ nguyên 'element' gốc + metadata bổ sung.
    """
    ts = datetime.utcnow().isoformat() + "Z"
    doc_id = doc_id or src_path.stem
    rows = []
    for el in elements:
        # el: {"type": "...", "text": "...", "metadata": {...}, "id": "...", ...}
        text = (el.get("text") or "").strip()
        etype = el.get("type") or ""
        meta = el.get("metadata") or {}
        row = {
            "source_path": str(src_path),
            "source_name": src_path.name,
            "source_sha1": src_sha1,
            "doc_id": doc_id,
            "element_id": el.get("id"),
            "element_type": etype,
            "text": text,
            "metadata": meta,
            "element": el,  # giữ nguyên element gốc để downstream có thể dùng thêm
            "ingested_at": ts,
        }
        if extra_meta:
            row["metadata_extra"] = extra_meta
        rows.append(row)
    return rows


# ============== Pipeline chính ==============

def preprocess_word_folder(
    in_dir: Path,
    out_jsonl: Path,
    api_url: str,
    api_key: str,
    languages: Optional[List[str]] = None,
    include_page_breaks: bool = True,
    encoding: str = "utf-8",
) -> Tuple[int, int, int]:
    """
    Quét thư mục, phát hiện file Word mới, gọi API, và append vào JSONL.
    Trả về: (n_files_tổng, n_files_mới, n_elements_ghi)
    """
    assert in_dir.exists(), f"Không thấy thư mục: {in_dir}"
    cfg = ApiConfig(
        url=api_url.strip() or DEFAULT_API_URL,
        api_key=api_key.strip(),
        include_page_breaks=include_page_breaks,
        encoding=encoding,
        languages=languages or ["vie"],
    )

    files = scan_word_files(in_dir)
    if not files:
        print(f"[INFO] Không có file .doc/.docx trong {in_dir}")
        return (0, 0, 0)

    # 1) Phát hiện trùng trong batch (cùng thư mục)
    dups = find_duplicates_in_batch(files)
    if dups:
        print("[WARN] Phát hiện file trùng nhau (cùng nội dung) trong thư mục:")
        for h, lst in dups.items():
            for i, p in enumerate(lst, 1):
                print(f"    - [{i}] {p}")
            print(f"      SHA1={h}\n")

    # 2) Đã xử lý trước đó? (đọc JSONL để lấy SHA-1 đã có)
    existing = collect_existing_hashes(out_jsonl)
    if existing:
        print(f"[INFO] Đã có {len(existing)} SHA-1 trong {out_jsonl} (sẽ bỏ qua nếu trùng).")

    n_total, n_new_files, n_elements_written = 0, 0, 0

    for path in files:
        n_total += 1
        try:
            h = sha1_file(path)
        except Exception as e:
            print(f"[WARN] Bỏ qua (không tính được SHA-1): {path} ({e})")
            continue

        if h in existing:
            print(f"[SKIP] Trùng file đã xử lý (SHA1={h[:12]}...) — {path}")
            continue

        print(f"[PROCESS] {path} (SHA1={h[:12]}...)")
        try:
            els = partition_via_unstructured_api(path, cfg)
        except Exception as e:
            print(f"[ERROR] API lỗi cho {path}: {e}")
            continue

        rows = elements_to_jsonl_rows(
            elements=els,
            src_path=path,
            src_sha1=h,
            doc_id=path.stem,  # bạn có thể thay logic xác định doc_id theo convention nội bộ
            extra_meta={
                "api_url": cfg.url,
                "include_page_breaks": cfg.include_page_breaks,
                "encoding": cfg.encoding,
                "languages": cfg.languages,
            },
        )
        n_written = append_jsonl(out_jsonl, rows)
        n_elements_written += n_written
        n_new_files += 1
        # cập nhật bộ nhớ 'existing' để nếu batch có file khác cùng SHA-1 sẽ skip tiếp
        existing[h] = existing.get(h, 0) + len(rows)

    print(f"[DONE] Files: total={n_total}, new={n_new_files} → wrote {n_elements_written} elements to {out_jsonl}")
    return (n_total, n_new_files, n_elements_written)


# ============== CLI ==============

def parse_args():
    p = argparse.ArgumentParser(description="Preprocess PCCC Word files via Unstructured API.")
    p.add_argument("--in_dir", type=str, required=True, help="Thư mục chứa các file .doc/.docx")
    p.add_argument("--out_jsonl", type=str, default="./pccc_word_elements.jsonl", help="Đường dẫn file JSONL đầu ra (append)")
    p.add_argument("--api_url", type=str, default=DEFAULT_API_URL, help="Partition Endpoint URL (mặc định theo ENV hoặc default)")
    p.add_argument("--languages", type=str, default="vie", help="Danh sách ngôn ngữ, ví dụ: 'vie' hoặc 'vie,eng'")
    p.add_argument("--no_page_breaks", action="store_true", help="Tắt include_page_breaks (mặc định bật)")
    p.add_argument("--encoding", type=str, default="utf-8", help="Encoding văn bản (mặc định utf-8)")
    return p.parse_args()


def main():
    args = parse_args()

    api_key = os.getenv(API_KEY_ENV, "").strip()
    if not api_key:
        print(f"[FATAL] Thiếu biến môi trường {API_KEY_ENV}.")
        print("  Xem tài liệu Unstructured Partition Endpoint để lấy API key & URL mặc định.")
        # default URL & header: https://docs.unstructured.io/api-reference/partition/overview
        sys.exit(1)

    in_dir = Path(args.in_dir)
    out_jsonl = Path(args.out_jsonl)
    api_url = args.api_url.strip() or DEFAULT_API_URL
    languages = [x.strip() for x in (args.languages or "").split(",") if x.strip()]

    preprocess_word_folder(
        in_dir=in_dir,
        out_jsonl=out_jsonl,
        api_url=api_url,
        api_key=api_key,
        languages=languages or ["vie"],
        include_page_breaks=not args.no_page_breaks,
        encoding=args.encoding,
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Preprocess PCCC legal docs via Unstructured API (DOC/DOCX/PDF)
- Incremental append to JSONL (skip duplicates by SHA1)
- Robust retries (re-open file each attempt to avoid empty uploads)
- Large-PDF friendly: split into page ranges before upload
- Table-friendly: hi_res + pdf_infer_table_structure (HTML table), keep coordinates
- Preserve absolute page numbers for accurate citations later

Env:
  UNSTRUCTURED_API_URL   (default: https://api.unstructuredapp.io/general/v0/general)
  UNSTRUCTURED_API_KEY   (required)
  UNSTRUCTURED_TIMEOUT_CONNECT_S (default: 15)
  UNSTRUCTURED_TIMEOUT_READ_S    (default: 120)
  UNSTRUCTURED_LANGUAGES         (default: vie)  # comma-separated

  # PDF controls
  UNSTRUCTURED_DEFAULT_STRATEGY_PDF (default: hi_res)
  UNSTRUCTURED_PDF_COORDINATES      (0/1, default: 1)
  UNSTRUCTURED_PDF_INFER_TABLE      (0/1, default: 1)   # request text_as_html for Table
  UNSTRUCTURED_HI_RES_MODEL_NAME    (optional, e.g., yolox)

  # Splitting thresholds
  PDF_SPLIT_ENABLE   (0/1, default: 1)
  PDF_MAX_MB         (default: 20)
  PDF_MAX_PAGES      (default: 150)
  PDF_PAGES_PER_CALL (default: 25)

Usage:
  pip install requests pypdf
  python preprocess_pccc_word_unstructured_api.py --data-dir /path/to/dir --out-jsonl pccc_word_elements.jsonl
"""

import argparse
import hashlib
import json
import mimetypes
import os
import sys
import time
import uuid
import tempfile
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Iterable, List, Tuple, Optional

import requests

try:
    from pypdf import PdfReader, PdfWriter
except Exception:
    PdfReader = None
    PdfWriter = None

SUPPORTED_EXTS = {".doc", ".docx", ".pdf"}

DEFAULT_API_URL = os.environ.get("UNSTRUCTURED_API_URL", "https://api.unstructuredapp.io/general/v0/general")
API_KEY = os.environ.get("UNSTRUCTURED_API_KEY", "").strip()
TIMEOUT_CONNECT = float(os.environ.get("UNSTRUCTURED_TIMEOUT_CONNECT_S", "15"))
TIMEOUT_READ = float(os.environ.get("UNSTRUCTURED_TIMEOUT_READ_S", "120"))

# PDF options
PDF_STRATEGY = os.environ.get("UNSTRUCTURED_DEFAULT_STRATEGY_PDF", "hi_res")
PDF_COORDINATES = os.environ.get("UNSTRUCTURED_PDF_COORDINATES", "1") == "1"
PDF_INFER_TABLE = os.environ.get("UNSTRUCTURED_PDF_INFER_TABLE", "1") == "1"
HI_RES_MODEL_NAME = os.environ.get("UNSTRUCTURED_HI_RES_MODEL_NAME", None)
LANGS = [s.strip() for s in os.environ.get("UNSTRUCTURED_LANGUAGES", "vie").split(",") if s.strip()]

# Splitting thresholds
PDF_SPLIT_ENABLE = os.environ.get("PDF_SPLIT_ENABLE", "1") == "1"
PDF_MAX_MB = float(os.environ.get("PDF_MAX_MB", "20"))
PDF_MAX_PAGES = int(os.environ.get("PDF_MAX_PAGES", "150"))
PDF_PAGES_PER_CALL = int(os.environ.get("PDF_PAGES_PER_CALL", "25"))

RETRY_STATUS = {408, 429, 500, 502, 503, 504}


@dataclass
class SrcFile:
    path: str
    ext: str
    mime: str
    sha1: str
    size_mb: float
    n_pages: Optional[int]  # only for PDF if available


def sha1_of_file(path: str, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sniff_mime(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return "application/pdf"
    if ext == ".docx":
        return "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    if ext == ".doc":
        return "application/msword"
    mt, _ = mimetypes.guess_type(path)
    return mt or "application/octet-stream"


def pdf_num_pages(path: str) -> Optional[int]:
    if PdfReader is None:
        return None
    try:
        return len(PdfReader(path).pages)
    except Exception:
        return None


def iter_source_files(data_dir: str) -> Iterable[SrcFile]:
    for root, _, files in os.walk(data_dir):
        for name in files:
            ext = os.path.splitext(name)[1].lower()
            if ext in SUPPORTED_EXTS:
                path = os.path.join(root, name)
                size_mb = os.path.getsize(path) / (1024 * 1024)
                n_pages = pdf_num_pages(path) if ext == ".pdf" else None
                yield SrcFile(
                    path=path,
                    ext=ext,
                    mime=sniff_mime(path),
                    sha1=sha1_of_file(path),
                    size_mb=size_mb,
                    n_pages=n_pages,
                )


def load_existing_hashes(out_jsonl: str) -> Dict[str, str]:
    hashes: Dict[str, str] = {}
    if not os.path.exists(out_jsonl):
        return hashes
    with open(out_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
            except Exception:
                continue
            h = obj.get("source_sha1")
            p = obj.get("source_path")
            if h and p and h not in hashes:
                hashes[h] = p
    return hashes


def _api_headers() -> Dict[str, str]:
    return {
        "accept": "application/json",
        "unstructured-api-key": API_KEY,
    }


def _form_common(include_page_breaks: bool = True) -> Dict[str, str]:
    return {
        "output_format": "application/json",
        "include_page_breaks": "true" if include_page_breaks else "false",
    }


def _form_for_pdf() -> Dict[str, str]:
    form = _form_common(include_page_breaks=True)
    form.update(
        {
            "strategy": PDF_STRATEGY,        # 'hi_res' (quality), 'fast' (speed), 'ocr_only'
            "coordinates": "true" if PDF_COORDINATES else "false",
        }
    )
    # Prefer enabling table structure HTML for table Q&A
    # NOTE: docs note pdf_infer_table_structure yields text_as_html for Table elements (hi_res) :contentReference[oaicite:5]{index=5}
    if PDF_INFER_TABLE:
        form["pdf_infer_table_structure"] = "true"
    if HI_RES_MODEL_NAME:
        form["hi_res_model_name"] = HI_RES_MODEL_NAME
    return form


def _form_for_word() -> Dict[str, str]:
    return _form_common(include_page_breaks=True)


def _build_multipart_data(form_dict: Dict[str, str], languages: List[str]) -> List[tuple]:
    items = list(form_dict.items())
    for lang in languages:
        items.append(("languages", lang))
    return items


def _post_partition(file_path: str, file_mime: str, form: Dict[str, str], langs: List[str]) -> requests.Response:
    """Open the file fresh for EVERY call, so the upload body is never empty on retries."""
    with open(file_path, "rb") as fh:
        files = {"files": (os.path.basename(file_path), fh, file_mime)}
        data = _build_multipart_data(form, langs)
        return requests.post(
            DEFAULT_API_URL,
            headers=_api_headers(),
            files=files,
            data=data,
            timeout=(TIMEOUT_CONNECT, TIMEOUT_READ),
        )


def _partition_with_retries(file_path: str, file_mime: str, forms_to_try: List[Dict[str, str]], languages_to_try: List[List[str]]) -> List[dict]:
    last_err = None
    for form, langs in zip(forms_to_try, languages_to_try):
        for attempt in range(5):
            try:
                resp = _post_partition(file_path, file_mime, form, langs)
                if resp.status_code == 200:
                    return resp.json()
                # Retry on transient statuses
                if resp.status_code in RETRY_STATUS:
                    wait = min(2 ** attempt, 16)
                    print(f"[WARN] API {resp.status_code} retry in {wait}s: {resp.text[:200]}")
                    time.sleep(wait)
                    continue
                # Non-transient: move to next fallback form
                last_err = RuntimeError(f"API error {resp.status_code}: {resp.text[:500]}")
                print(f"[WARN] Try fallback for {os.path.basename(file_path)}: {last_err}")
                break
            except requests.RequestException as e:
                last_err = e
                wait = min(2 ** attempt, 16)
                print(f"[WARN] Network error, retry in {wait}s: {e}")
                time.sleep(wait)
                continue
    raise RuntimeError(f"Failed to partition {file_path}: {last_err}")


def _partition_pdf(src: SrcFile) -> List[dict]:
    primary_form = _form_for_pdf()
    fallbacks: List[Dict[str, str]] = []
    if "pdf_infer_table_structure" in primary_form:
        fb1 = dict(primary_form)
        fb1["pdf_infer_table_structure"] = "false"  # disable table inference if it crashes
        fallbacks.append(fb1)
    fallbacks.append({"strategy": "fast", "output_format": "application/json"})
    fallbacks.append({"strategy": "ocr_only", "output_format": "application/json"})
    forms_to_try = [primary_form] + fallbacks
    languages_to_try = [LANGS] + [LANGS, [], LANGS]
    return _partition_with_retries(src.path, src.mime, forms_to_try, languages_to_try)


def _partition_word(src: SrcFile) -> List[dict]:
    form = _form_for_word()
    return _partition_with_retries(src.path, src.mime, [form], [[]])


def _normalize_element(raw: dict,
                       src: SrcFile,
                       page_offset: int = 0,
                       doc_page_range: Optional[Tuple[int,int]] = None) -> dict:
    """
    Normalize Unstructured element:
    - keep text/type/metadata (metadata may include page_number, text_as_html for tables)
    - add absolute page number adjusted by offset (for PDF splits)
    """
    etype = raw.get("type") or raw.get("category")
    text = raw.get("text") or ""
    meta = raw.get("metadata") or {}
    rel_page = meta.get("page_number") or meta.get("page")
    try:
        rel_page = int(rel_page) if rel_page is not None else None
    except Exception:
        rel_page = None

    abs_page = (rel_page + page_offset) if rel_page is not None else None

    # inject absolute page number + doc page range for downstream citation
    meta_out = dict(meta)
    if abs_page is not None:
        meta_out["page_number_abs"] = abs_page  # absolute page (1-based) within original file
    if doc_page_range:
        meta_out["doc_page_range"] = {"start": doc_page_range[0], "end": doc_page_range[1]}

    return {
        "id": str(uuid.uuid4()),
        "text": text,
        "type": etype,
        "page_number": abs_page,     # promote absolute page
        "metadata": meta_out,        # keep all + enrich
        # provenance
        "source_path": src.path,
        "source_name": os.path.basename(src.path),
        "source_ext": src.ext,
        "source_mime": src.mime,
        "source_sha1": src.sha1,
        "ingested_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }


# ---------- PDF splitting utilities ----------

def _should_split_pdf(src: SrcFile) -> bool:
    if not PDF_SPLIT_ENABLE or src.ext != ".pdf":
        return False
    if src.size_mb >= PDF_MAX_MB:
        return True
    if src.n_pages is not None and src.n_pages >= PDF_MAX_PAGES:
        return True
    return False


def _split_pdf_to_temp_ranges(src: SrcFile, pages_per_call: int = PDF_PAGES_PER_CALL) -> List[Tuple[str, Tuple[int,int]]]:
    """
    Return list of (temp_pdf_path, (start_abs, end_abs)) with 1-based inclusive page nums.
    """
    assert PdfReader is not None and PdfWriter is not None, "pypdf not installed"
    reader = PdfReader(src.path)
    total = len(reader.pages)
    out: List[Tuple[str, Tuple[int,int]]] = []
    for start in range(1, total + 1, pages_per_call):
        end = min(start + pages_per_call - 1, total)
        writer = PdfWriter()
        for p in range(start-1, end):
            writer.add_page(reader.pages[p])
        tmp = tempfile.NamedTemporaryFile(prefix="pccc_pdf_part_", suffix=".pdf", delete=False)
        with open(tmp.name, "wb") as fh:
            writer.write(fh)
        out.append((tmp.name, (start, end)))
    return out


def _partition_pdf_splitted(src: SrcFile) -> List[dict]:
    """
    Split large PDF to parts and partition each; adjust page numbers back.
    """
    parts = _split_pdf_to_temp_ranges(src, PDF_PAGES_PER_CALL)
    all_elems: List[dict] = []
    for tmp_path, (start_abs, end_abs) in parts:
        try:
            sub = SrcFile(
                path=tmp_path,
                ext=".pdf",
                mime="application/pdf",
                sha1=src.sha1,         # keep original SHA1 to dedup by source
                size_mb=os.path.getsize(tmp_path)/(1024*1024),
                n_pages=end_abs - start_abs + 1,
            )
            # build forms to try
            primary_form = _form_for_pdf()
            fallbacks: List[Dict[str, str]] = []
            if "pdf_infer_table_structure" in primary_form:
                fb1 = dict(primary_form)
                fb1["pdf_infer_table_structure"] = "false"
                fallbacks.append(fb1)
            fallbacks.append({"strategy": "fast", "output_format": "application/json"})
            fallbacks.append({"strategy": "ocr_only", "output_format": "application/json"})
            forms_to_try = [primary_form] + fallbacks
            languages_to_try = [LANGS] + [LANGS, [], LANGS]

            last_err = None
            for form, langs in zip(forms_to_try, languages_to_try):
                for attempt in range(5):
                    try:
                        resp = _post_partition(sub.path, sub.mime, form, langs)
                        if resp.status_code == 200:
                            data = resp.json() or []
                            # remap page numbers
                            for raw in data:
                                yield_elem = _normalize_element(
                                    raw, src,
                                    page_offset=(start_abs - 1),
                                    doc_page_range=(start_abs, end_abs),
                                )
                                all_elems.append(yield_elem)
                            raise StopIteration  # break both loops
                        if resp.status_code in RETRY_STATUS:
                            wait = min(2 ** attempt, 16)
                            print(f"[WARN] API {resp.status_code} retry in {wait}s (part {start_abs}-{end_abs})")
                            time.sleep(wait); continue
                        last_err = RuntimeError(f"API error {resp.status_code}: {resp.text[:400]}")
                        print(f"[WARN] Fallback next (part {start_abs}-{end_abs}): {last_err}")
                        break
                    except requests.RequestException as e:
                        last_err = e
                        wait = min(2 ** attempt, 16)
                        print(f"[WARN] Network error retry in {wait}s (part {start_abs}-{end_abs}): {e}")
                        time.sleep(wait); continue
            if isinstance(last_err, StopIteration):  # type: ignore
                pass
        finally:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
    return all_elems


def preprocess_folder(data_dir: str, out_jsonl: str) -> Tuple[int, int]:
    """
    Iterate folder, partition files, and append normalized elements.
    Skip duplicates by source SHA1.
    For PDFs, optionally split by page ranges.
    """
    if not API_KEY:
        print("ERROR: UNSTRUCTURED_API_KEY is missing.", file=sys.stderr)
        sys.exit(1)

    os.makedirs(os.path.dirname(os.path.abspath(out_jsonl)), exist_ok=True)
    seen_hashes = load_existing_hashes(out_jsonl)

    n_seen, n_new = 0, 0
    with open(out_jsonl, "a", encoding="utf-8") as fout:
        for src in iter_source_files(data_dir):
            n_seen += 1
            if src.sha1 in seen_hashes:
                print(f"[SKIP] Duplicate file (SHA1={src.sha1[:12]}...): {src.path} == {seen_hashes[src.sha1]}")
                continue

            try:
                if src.ext == ".pdf" and _should_split_pdf(src) and PdfReader is not None:
                    print(f"[INFO] Splitting large PDF: {src.path} | size={src.size_mb:.1f}MB, pages={src.n_pages}")
                    elements = _partition_pdf_splitted(src)
                else:
                    # single-shot partition
                    if src.ext == ".pdf":
                        raw_elems = _partition_pdf(src)
                    else:
                        raw_elems = _partition_word(src)
                    # normalize; for non-split, no page offset
                    elements = [_normalize_element(raw, src, page_offset=0) for raw in (raw_elems or [])]
            except Exception as e:
                print(f"[ERROR] Partition failed for {src.path}: {e}")
                continue

            count = 0
            for obj in elements or []:
                fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
                count += 1

            print(f"[OK] {src.path} → {count} elements")
            seen_hashes[src.sha1] = src.path
            n_new += 1

    return n_seen, n_new


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, help="Folder with DOC/DOCX/PDF")
    parser.add_argument("--out-jsonl", required=True, help="Output JSONL to append elements")
    args = parser.parse_args()

    n_seen, n_new = preprocess_folder(args.data_dir, args.out_jsonl)
    print(f"✅ Scanned {n_seen} files, appended {n_new} new files to {args.out_jsonl}")


if __name__ == "__main__":
    main()

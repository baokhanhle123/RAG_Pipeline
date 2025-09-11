#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Preprocess PCCC docs:
- DOC/DOCX/PDF: Unstructured API (PDF có thể chunk).
- PPT/PPTX: xử lý CỤC BỘ bằng python-pptx (KHÔNG dùng Unstructured API).
- Ghi JSONL: mỗi element có text + metadata (không chứa None), giữ số trang/slide tuyệt đối.
"""

from __future__ import annotations
import argparse, hashlib, io, json, mimetypes, os, sys, time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# -------- Optional deps --------
try:
    from requests_toolbelt.multipart.encoder import MultipartEncoder
except Exception:
    MultipartEncoder = None

try:
    from pypdf import PdfReader, PdfWriter
except Exception:
    PdfReader = None  # type: ignore
    PdfWriter = None  # type: ignore

# python-pptx (local PPTX parsing)
try:
    from pptx import Presentation
    from pptx.enum.shapes import MSO_SHAPE_TYPE  # Hình ảnh, v.v. :contentReference[oaicite:6]{index=6}
    PPTX_AVAILABLE = True
except Exception:
    PPTX_AVAILABLE = False

# ----------------- Config -----------------
SUPPORTED_EXTS = {".doc", ".docx", ".pdf", ".ppt", ".pptx", ".pptm", ".pot", ".potx"}
PPT_EXTS = {".ppt", ".pptx", ".pptm", ".pot", ".potx"}

DEFAULT_API_URL = os.environ.get("UNSTRUCTURED_API_URL", "https://api.unstructuredapp.io/general/v0/general")
API_KEY = os.environ.get("UNSTRUCTURED_API_KEY", "").strip()

TIMEOUT_CONNECT = float(os.environ.get("UNSTRUCTURED_TIMEOUT_CONNECT_S", "15"))
TIMEOUT_READ = float(os.environ.get("UNSTRUCTURED_TIMEOUT_READ_S", "120"))

MIN_UPLOAD_KBPS = float(os.environ.get("UNSTRUCTURED_MIN_UPLOAD_KBPS", "256"))
MAX_READ_TIMEOUT_S = float(os.environ.get("UNSTRUCTURED_MAX_READ_TIMEOUT_S", "1800"))
UPLOAD_SAFETY_PAD_S = float(os.environ.get("UNSTRUCTURED_UPLOAD_TIMEOUT_SAFETY", "45"))

DEFAULT_MAX_PDF_PAGES_PER_CHUNK = 60

PDF_PARAMS = {
    "strategy": "hi_res",
    "coordinates": "true",
    "include_page_breaks": "true",
    "pdf_infer_table_structure": "true",  # HTML bảng cho PDF khi hi_res  :contentReference[oaicite:7]{index=7}
    "languages": "vie",
}
GENERIC_PARAMS = {"strategy": "auto", "include_page_breaks": "true", "languages": "vie"}

# ----------------- Helpers -----------------
def sha1_of_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b: break
            h.update(b)
    return h.hexdigest()

def load_existing_hashes(jsonl_path: Path) -> set:
    seen = set()
    if not jsonl_path.exists(): return seen
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try: obj = json.loads(line)
            except Exception: continue
            fs = (obj.get("metadata") or {}).get("file_sha1")
            if fs: seen.add(fs)
    return seen

def detect_mime(path: Path) -> str:
    mime, _ = mimetypes.guess_type(str(path))
    return mime or "application/octet-stream"

def build_partition_params_for(path: Path) -> Dict[str, str]:
    return dict(PDF_PARAMS if path.suffix.lower() == ".pdf" else GENERIC_PARAMS)

def chunk_pdf_bytes(source_pdf: Path, max_pages: int) -> Iterable[Tuple[bytes, int, int]]:
    if PdfReader is None:
        yield (source_pdf.read_bytes(), 1, -1); return
    reader = PdfReader(str(source_pdf))
    total = len(reader.pages)
    if total <= max_pages:
        yield (source_pdf.read_bytes(), 1, total); return
    start = 1
    while start <= total:
        end = min(start + max_pages - 1, total)
        writer = PdfWriter()
        for p in range(start - 1, end):
            writer.add_page(reader.pages[p])
        buf = io.BytesIO(); writer.write(buf)
        yield (buf.getvalue(), start, end)
        start = end + 1

def _compute_upload_timeouts(file_size_bytes: int) -> Tuple[float, float]:
    connect_t = max(3.05, TIMEOUT_CONNECT)
    base_read = TIMEOUT_READ
    if file_size_bytes <= 0 or MIN_UPLOAD_KBPS <= 0:
        return (connect_t, max(base_read, 300.0))
    est_seconds = (file_size_bytes / 1024.0) / MIN_UPLOAD_KBPS
    read_t = max(base_read, est_seconds * 3.0 + UPLOAD_SAFETY_PAD_S)
    return (connect_t, min(read_t, MAX_READ_TIMEOUT_S))

def _make_session() -> requests.Session:
    s = requests.Session()
    retries = Retry(total=0, connect=0, read=0, backoff_factor=0, status_forcelist=[],
                    allowed_methods=frozenset(["POST"]), raise_on_status=False)
    adapter = HTTPAdapter(max_retries=retries, pool_connections=4, pool_maxsize=4)
    s.mount("https://", adapter); s.mount("http://", adapter)
    return s

def _post_to_unstructured_bytes(file_bytes: bytes, filename: str, extra_params: Dict[str, str],
                                timeout_pair: Tuple[float, float], max_retries: int = 4) -> List[dict]:
    if not API_KEY:
        raise RuntimeError("UNSTRUCTURED_API_KEY is not set.")
    url = DEFAULT_API_URL
    headers = {"accept": "application/json", "unstructured-api-key": API_KEY}
    last_err = None
    sess = _make_session()
    for attempt in range(1, max_retries + 1):
        try:
            files = {"files": (filename, io.BytesIO(file_bytes), detect_mime(Path(filename)))}
            resp = sess.post(url, headers=headers, files=files, data=extra_params, timeout=timeout_pair)
            if resp.status_code in (200, 201): return resp.json()
            if resp.status_code >= 500 or resp.status_code in (408, 429):
                last_err = RuntimeError(f"HTTP {resp.status_code}: {resp.text[:300]}"); time.sleep(min(2**attempt, 10)); continue
            resp.raise_for_status()
        except Exception as e:
            last_err = e; time.sleep(min(2**attempt, 10)); continue
    raise RuntimeError(f"Failed to call Unstructured API after {max_retries} attempts: {last_err}")

# ----------------- PPTX local parsing -----------------
# Tài liệu: has_text_frame / paragraphs / runs, has_table, MSO_SHAPE_TYPE.PICTURE :contentReference[oaicite:8]{index=8}
EMU_PER_INCH = 914400
def _emu_to_px(emu: int, dpi: int = 96) -> int:
    return int(round((emu / EMU_PER_INCH) * dpi))

def _clean(s: Optional[str]) -> str:
    return " ".join(str(s or "").split())

def _para_to_text(paragraph) -> str:
    runs = [r.text for r in paragraph.runs if r.text]
    return _clean("".join(runs)) or _clean(paragraph.text)

def _shape_coordinates(shape) -> Optional[dict]:
    try:
        return {
            "left_px": _emu_to_px(int(shape.left)),
            "top_px": _emu_to_px(int(shape.top)),
            "width_px": _emu_to_px(int(shape.width)),
            "height_px": _emu_to_px(int(shape.height)),
            "units": "px@96dpi",
        }
    except Exception:
        return None

def _table_to_markdown(table) -> str:
    rows = table.rows
    if len(rows) == 0: return ""
    def cell_text(cell): return _clean(getattr(cell, "text", "") or "")
    header = [cell_text(c) for c in rows[0].cells]
    md = []
    md.append("| " + " | ".join(header) + " |")
    md.append("| " + " | ".join(["---" for _ in header]) + " |")
    for r in rows[1:]:
        md.append("| " + " | ".join([cell_text(c) for c in r.cells]) + " |")
    return "\n".join(md)

def _table_to_html(table) -> str:
    rows = table.rows
    if len(rows) == 0: return ""
    def cell_text(cell): return _clean(getattr(cell, "text", "") or "")
    html = ["<table>"]
    for ir, r in enumerate(rows):
        html.append("<tr>")
        tag = "th" if ir == 0 else "td"
        for c in r.cells: html.append(f"<{tag}>{cell_text(c)}</{tag}>")
        html.append("</tr>")
    html.append("</table>")
    return "".join(html)

def parse_pptx_locally(path: Path) -> List[dict]:
    if not PPTX_AVAILABLE:
        raise RuntimeError("python-pptx is not installed. pip install python-pptx")
    prs = Presentation(str(path))
    slide_w, slide_h = int(prs.slide_width), int(prs.slide_height)

    elements: List[dict] = []
    for idx, slide in enumerate(prs.slides, start=1):
        for shape in slide.shapes:
            try:
                # TABLE
                if getattr(shape, "has_table", False):
                    table = shape.table
                    md = _table_to_markdown(table)
                    html = _table_to_html(table)
                    text = ("[BẢNG]\n" + md) if md else "[BẢNG]"
                    meta = {
                        "file_name": path.name, "file_path": str(path), "mime_type": detect_mime(path),
                        "page_number": idx, "abs_page_number": idx, "slide_number": idx,
                        "has_table_html": True, "table_html_excerpt": html,
                        "coordinates": _shape_coordinates(shape),
                        "slide_size_px": {"width_px": _emu_to_px(slide_w), "height_px": _emu_to_px(slide_h)},
                    }
                    elements.append({
                        "element_id": f"pptx-{idx}-table-{len(elements)+1}",
                        "type": "Table",
                        "text": text,
                        "metadata": {k: v for k, v in meta.items() if v is not None},
                    })
                    continue

                # TEXT (title / bullets / narrative)
                if getattr(shape, "has_text_frame", False):
                    for para in shape.text_frame.paragraphs:
                        txt = _para_to_text(para)
                        if not txt: continue
                        lvl = getattr(para, "level", 0) or 0
                        is_bullet = getattr(para, "bullet", None)
                        if is_bullet:
                            md_text = ("  " * lvl) + "- " + txt
                            el_type = "ListItem"
                        else:
                            el_type = "NarrativeText"
                            md_text = txt
                        meta = {
                            "file_name": path.name, "file_path": str(path), "mime_type": detect_mime(path),
                            "page_number": idx, "abs_page_number": idx, "slide_number": idx,
                            "coordinates": _shape_coordinates(shape),
                            "slide_size_px": {"width_px": _emu_to_px(slide_w), "height_px": _emu_to_px(slide_h)},
                        }
                        elements.append({
                            "element_id": f"pptx-{idx}-text-{len(elements)+1}",
                            "type": el_type,
                            "text": md_text,
                            "metadata": {k: v for k, v in meta.items() if v is not None},
                        })
                    continue

                # IMAGE
                if getattr(shape, "shape_type", None) == MSO_SHAPE_TYPE.PICTURE:
                    meta = {
                        "file_name": path.name, "file_path": str(path), "mime_type": detect_mime(path),
                        "page_number": idx, "abs_page_number": idx, "slide_number": idx,
                        "has_image": True, "coordinates": _shape_coordinates(shape),
                        "slide_size_px": {"width_px": _emu_to_px(slide_w), "height_px": _emu_to_px(slide_h)},
                    }
                    elements.append({
                        "element_id": f"pptx-{idx}-image-{len(elements)+1}",
                        "type": "Figure",
                        "text": "[HÌNH]",
                        "metadata": {k: v for k, v in meta.items() if v is not None},
                    })
                    continue

            except Exception as e:
                sys.stderr.write(f"[WARN] slide {idx}: skip shape due to {e}\n")

        # Page break marker (tuỳ chọn)
        elements.append({
            "element_id": f"pptx-{idx}-pagebreak",
            "type": "PageBreak",
            "text": "",
            "metadata": {"page_number": idx, "abs_page_number": idx, "slide_number": idx,
                         "file_name": path.name, "file_path": str(path)},
        })

    return elements

# ----------------- Normalize (dùng chung) -----------------
def _normalize_element(e: dict, *, file_sha1: str, file_path: Path, abs_page_offset: int = 0) -> dict:
    meta = e.get("metadata") or {}
    page_no = meta.get("page_number")
    abs_page = (page_no + max(abs_page_offset, 0)) if isinstance(page_no, int) else meta.get("abs_page_number")
    normalized = {
        "element_id": e.get("element_id"),
        "type": e.get("type"),
        "text": e.get("text"),
        "metadata": {
            "file_name": meta.get("file_name", file_path.name),
            "file_path": meta.get("file_path", str(file_path)),
            "file_sha1": file_sha1,
            "source": meta.get("source", "pptx_local" if file_path.suffix.lower() in PPT_EXTS else "unstructured_api"),
            "mime_type": meta.get("mime_type", detect_mime(file_path)),
            "page_number": meta.get("page_number"),
            "abs_page_number": abs_page,
            "slide_number": meta.get("slide_number"),
            "languages": meta.get("languages"),
            "coordinates": meta.get("coordinates"),
            "text_as_html": meta.get("text_as_html"),
            "has_table_html": meta.get("has_table_html"),
            "table_html_excerpt": meta.get("table_html_excerpt"),
            "has_image": meta.get("has_image"),
            "slide_size_px": meta.get("slide_size_px"),
            "parent_id": meta.get("parent_id"),
            "section": meta.get("section"),
            "category_depth": meta.get("category_depth"),
        },
    }
    normalized["metadata"] = {k: v for k, v in normalized["metadata"].items() if v is not None}
    return normalized

# ----------------- Core -----------------
def _post_doc_or_pdf(path: Path, out_jsonl: Path, max_pdf_pages_per_chunk: int) -> int:
    """DOC/DOCX/PDF → Unstructured API; PDF có thể chunk."""
    file_sha1 = sha1_of_file(path)
    file_size = path.stat().st_size
    timeout_pair = _compute_upload_timeouts(file_size)
    params = build_partition_params_for(path)
    written = 0

    if path.suffix.lower() == ".pdf":
        for chunk_bytes, start_idx, end_idx in chunk_pdf_bytes(path, max_pages=max_pdf_pages_per_chunk):
            resp_json = _post_to_unstructured_bytes(chunk_bytes, path.name, params, timeout_pair=timeout_pair)
            offset = max(start_idx - 1, 0) if end_idx != -1 else 0
            with out_jsonl.open("a", encoding="utf-8") as out:
                for el in resp_json:
                    norm = _normalize_element(el, file_sha1=file_sha1, file_path=path, abs_page_offset=offset)
                    out.write(json.dumps(norm, ensure_ascii=False) + "\n")
                    written += 1
    else:
        file_bytes = path.read_bytes()
        resp_json = _post_to_unstructured_bytes(file_bytes, path.name, params, timeout_pair=timeout_pair)
        with out_jsonl.open("a", encoding="utf-8") as out:
            for el in resp_json:
                norm = _normalize_element(el, file_sha1=file_sha1, file_path=path, abs_page_offset=0)
                out.write(json.dumps(norm, ensure_ascii=False) + "\n")
                written += 1
    return written

def _post_ppt_or_pptx_locally(path: Path, out_jsonl: Path) -> int:
    """PPT/PPTX → PARSE LOCAL (python-pptx), KHÔNG dùng Unstructured API."""
    file_sha1 = sha1_of_file(path)
    elements = parse_pptx_locally(path)
    written = 0
    with out_jsonl.open("a", encoding="utf-8") as out:
        for el in elements:
            norm = _normalize_element(el, file_sha1=file_sha1, file_path=path, abs_page_offset=0)
            out.write(json.dumps(norm, ensure_ascii=False) + "\n")
            written += 1
    return written

def preprocess_file(path: Path, out_jsonl: Path, max_pdf_pages_per_chunk: int) -> int:
    if path.suffix.lower() not in SUPPORTED_EXTS: return 0
    if path.suffix.lower() in PPT_EXTS:
        # KHÔNG dùng Unstructured API cho PPT/PPTX
        return _post_ppt_or_pptx_locally(path, out_jsonl)
    else:
        # DOC/DOCX/PDF vẫn đi qua API
        return _post_doc_or_pdf(path, out_jsonl, max_pdf_pages_per_chunk)

def preprocess_folder(data_dir: str, out_jsonl: str, max_pdf_pages_per_chunk: int) -> Tuple[int, int]:
    data_path = Path(data_dir); out_path = Path(out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    existing_hashes = load_existing_hashes(out_path)
    seen_files = new_files = 0
    for p in sorted(data_path.rglob("*")):
        if not p.is_file(): continue
        if p.suffix.lower() not in SUPPORTED_EXTS: continue
        seen_files += 1
        fsha1 = sha1_of_file(p)
        if fsha1 in existing_hashes: continue
        try:
            written = preprocess_file(p, out_path, max_pdf_pages_per_chunk)
            if written > 0:
                new_files += 1; existing_hashes.add(fsha1)
                print(f"Appended {written} elements from: {p.name}")
        except Exception as e:
            print(f"⚠️  Failed {p.name}: {e}", file=sys.stderr)
    return seen_files, new_files

# ----------------- CLI -----------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, help="Folder with DOC/DOCX/PDF/PPT/PPTX")
    parser.add_argument("--out-jsonl", required=True, help="Output JSONL to append elements")
    parser.add_argument("--max-pdf-pages-per-chunk", type=int,
                        default=int(os.environ.get("MAX_PDF_PAGES_PER_CHUNK", DEFAULT_MAX_PDF_PAGES_PER_CHUNK)),
                        help=f"Max pages per PDF chunk (default {DEFAULT_MAX_PDF_PAGES_PER_CHUNK})")
    args = parser.parse_args()

    if not PPTX_AVAILABLE:
        print("⚠️  python-pptx chưa được cài — cài: pip install python-pptx", file=sys.stderr)

    n_seen, n_new = preprocess_folder(args.data_dir, args.out_jsonl, max_pdf_pages_per_chunk=args.max_pdf_pages_per_chunk)
    print(f"✅ Scanned {n_seen} files, appended {n_new} new files to {args.out_jsonl}")

if __name__ == "__main__":
    main()

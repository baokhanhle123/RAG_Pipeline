#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Chunking 'smart' cho PCCC — tối ưu bảng & hình ảnh (DOC/DOCX/PDF/PPTX), incremental.

Đầu vào:  JSONL elements từ bước preprocess (đã có PPTX parse cục bộ)
Đầu ra:   JSONL chunks (append)

Điểm chính:
- BẢNG: ưu tiên HTML→Markdown nếu có (PDF: text_as_html; PPTX: table_html_excerpt); lưu excerpt HTML vào metadata.
- HÌNH: tạo chunk [HÌNH] (caption + ngữ cảnh lân cận), lưu tọa độ/page phục vụ trích dẫn.
- Ba chế độ xử lý bảng: metadata_only | append_md | separate_chunk (mặc định: separate_chunk).
- Nhận diện Điều/Khoản/Điểm để ghép header pháp lý cho mọi chunk.
- Đọc đúng trường nguồn từ metadata (file_sha1/file_name/file_path) và ưu tiên abs_page_number nếu có.
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Any
import argparse
import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass

# ============= JSONL I/O =============

def read_jsonl_iter(path: Path) -> Iterable[dict]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                sys.stderr.write(f"[WARN] Bỏ dòng hỏng {path}:{ln}\n")

def append_jsonl(path: Path, rows: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("a", encoding="utf-8") as f:
        for r in rows:
            # Lọc None trong metadata để tránh lỗi index (Chroma yêu cầu primitive types)
            meta = r.get("metadata") or {}
            meta = {k: v for k, v in meta.items() if v is not None}
            r["metadata"] = meta
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n

# ============= Theo dõi nguồn đã chunk =============

def collect_processed_sha1(out_jsonl: Path) -> Dict[str, int]:
    seen: Dict[str, int] = {}
    if not out_jsonl.exists():
        return seen
    for row in read_jsonl_iter(out_jsonl):
        sha1 = row.get("source_sha1") or (row.get("metadata") or {}).get("source_sha1")
        if sha1:
            seen[sha1] = seen.get(sha1, 0) + 1
    return seen

# ============= Gom element theo nguồn =============

@dataclass
class RawElement:
    element_id: Optional[str]
    element_type: str
    text: str
    metadata: Dict[str, Any]
    page_number: Optional[int]
    source_name: str
    source_path: str
    source_ext: str
    source_sha1: str

def _extract_element_fields(row: dict) -> Optional[RawElement]:
    """
    Chuẩn hoá 1 dòng element từ preprocess (Unstructured/PPTX local):
    - đọc 'type', 'text', 'metadata'
    - suy ra page_number ưu tiên abs_page_number
    - rút nguồn từ metadata: file_sha1 / file_name / file_path
    """
    # Dạng chuẩn (preprocess hiện tại)
    text = row.get("text")
    etype = row.get("type")
    meta = row.get("metadata") or {}
    if text is not None and etype is not None:
        # Trang ưu tiên tuyệt đối
        page = meta.get("abs_page_number", meta.get("page_number"))
        try:
            page = int(page) if page is not None else None
        except Exception:
            page = None

        # Nguồn từ metadata
        m_name = meta.get("file_name") or ""
        m_path = meta.get("file_path") or ""
        m_sha1 = meta.get("file_sha1") or meta.get("source_sha1") or row.get("source_sha1") or ""
        src_name = row.get("source_name") or m_name or Path(m_path).name
        src_path = row.get("source_path") or m_path
        src_ext = (row.get("source_ext") or Path(src_name).suffix).lower()

        return RawElement(
            element_id=row.get("element_id") or row.get("id"),
            element_type=str(etype),
            text=str(text or ""),
            metadata=meta,
            page_number=page,
            source_name=src_name or "",
            source_path=src_path or "",
            source_ext=src_ext or "",
            source_sha1=str(m_sha1 or ""),
        )

    # Fallback schema cũ
    el = row.get("element") or {}
    text = el.get("text") or row.get("text") or ""
    etype = el.get("type") or row.get("element_type") or ""
    meta = el.get("metadata") or row.get("metadata") or {}
    if not etype:
        return None
    page = meta.get("abs_page_number", meta.get("page_number"))
    try:
        page = int(page) if page is not None else None
    except Exception:
        page = None

    m_name = meta.get("file_name") or ""
    m_path = meta.get("file_path") or ""
    m_sha1 = meta.get("file_sha1") or meta.get("source_sha1") or row.get("source_sha1") or ""
    src_name = row.get("source_name") or m_name or Path(m_path).name
    src_path = row.get("source_path") or m_path
    src_ext = (row.get("source_ext") or Path(src_name).suffix).lower()

    return RawElement(
        element_id=row.get("element_id") or el.get("id"),
        element_type=str(etype),
        text=str(text or ""),
        metadata=meta,
        page_number=page,
        source_name=src_name or "",
        source_path=src_path or "",
        source_ext=src_ext or "",
        source_sha1=str(m_sha1 or ""),
    )

def group_elements_by_source(in_jsonl: Path) -> Dict[Tuple[str, str], List[RawElement]]:
    groups: Dict[Tuple[str, str], List[RawElement]] = {}
    for row in read_jsonl_iter(in_jsonl):
        relem = _extract_element_fields(row)
        if not relem or not relem.source_sha1:
            continue
        # doc_id: ưu tiên stem(file_name); fallback SHA1 rút gọn
        default_id = Path(relem.source_name).stem if relem.source_name else relem.source_sha1[:12]
        doc_id = row.get("doc_id") or default_id
        key = (relem.source_sha1, doc_id)
        groups.setdefault(key, []).append(relem)
    return groups

# ============= Tokenizer (tiktoken) =============

try:
    import tiktoken
    ENC = tiktoken.get_encoding("cl100k_base")
except Exception:
    ENC = None

def tok_len(s: str) -> int:
    if ENC is None:
        return max(1, len(s)//2)
    return len(ENC.encode(s or ""))

def truncate_tokens(s: str, max_tokens: int) -> str:
    if tok_len(s) <= max_tokens:
        return s
    if ENC is None:
        return s[: max_tokens*2]
    ids = ENC.encode(s or "")
    return ENC.decode(ids[:max_tokens])

# ============= Tách câu (TV) =============

_SENT_SPLIT = re.compile(r"(?<=[\.!?…])\s+|\n+(?=[^\s])")

def split_sentences_vi(text: str) -> List[str]:
    return [s.strip() for s in _SENT_SPLIT.split(text.strip()) if s.strip()]

# ============= Heading pháp lý + liệt kê =============

HDR_PATTERNS = {
    "chuong": re.compile(r"^\s*Chương\s+([IVXLCDM]+|\d+)\b", re.IGNORECASE),
    "muc":    re.compile(r"^\s*Mục\s+([A-Z0-9]+)\b", re.IGNORECASE),
    "dieu":   re.compile(r"^\s*Điều\s+(\d+)\b", re.IGNORECASE),
    "khoan":  re.compile(r"^\s*Khoản\s+(\d+)\b", re.IGNORECASE),
    "diem":   re.compile(r"^\s*Điểm\s+([a-z])\b", re.IGNORECASE),
    "phuluc": re.compile(r"^\s*Phụ\s*lục\s+([A-Z0-9]+)\b", re.IGNORECASE),
}
ENUM_KHOAN = [re.compile(r"^\s*(\d+)[\.\)]\s+"), re.compile(r"^\s*\((\d+)\)\s+")]
ENUM_DIEM  = [re.compile(r"^\s*([a-z])[\.\)]\s+"), re.compile(r"^\s*\(([a-z])\)\s+")]

def detect_heading_info(text: str) -> Dict[str, Optional[str]]:
    info = {"chuong":None,"muc":None,"dieu":None,"khoan":None,"diem":None,"phu_luc":None}
    t = (text or "").strip()
    if not t: return info
    for k, pat in HDR_PATTERNS.items():
        m = pat.search(t)
        if m:
            v = m.group(1)
            if k == "phuluc": k = "phu_luc"
            info[k] = v
    if info["khoan"] is None:
        for p in ENUM_KHOAN:
            m = p.match(t)
            if m: info["khoan"] = m.group(1); break
    if info["diem"] is None:
        for p in ENUM_DIEM:
            m = p.match(t)
            if m: info["diem"] = m.group(1); break
    return info

def update_state_with(info: Dict[str, Optional[str]], state: Dict[str, Optional[str]]) -> Dict[str, Optional[str]]:
    st = dict(state); changed = False
    if info.get("chuong"): st["chuong"] = info["chuong"]; changed = True
    if info.get("muc"):    st["muc"]    = info["muc"];    changed = True
    if info.get("phu_luc"):st["phu_luc"]= info["phu_luc"];changed = True
    if info.get("dieu"):   st["dieu"]   = info["dieu"]; st["khoan"]=None; st["diem"]=None; changed=True
    if info.get("khoan"):  st["khoan"]  = info["khoan"]; st["diem"]=None; changed=True
    if info.get("diem"):   st["diem"]   = info["diem"];  changed=True
    st["_changed"] = changed
    return st

def make_section_key(state: Dict[str, Optional[str]], fallback: str="") -> str:
    parts = []
    if state.get("chuong"): parts.append(f"CHUONG_{state['chuong']}")
    if state.get("muc"):    parts.append(f"MUC_{state['muc']}")
    if state.get("dieu"):   parts.append(f"DIEU_{state['dieu']}")
    if state.get("khoan"):  parts.append(f"KHOAN_{state['khoan']}")
    if state.get("diem"):   parts.append(f"DIEM_{state['diem']}")
    if state.get("phu_luc"):parts.append(f"PHULUC_{state['phu_luc']}")
    if not parts and fallback:
        parts.append(re.sub(r"\s+", "_", fallback.strip())[:60])
    return ".".join(parts) or "SECTION"

def build_citation(van_ban: str, st: Dict[str, Optional[str]], p1: Optional[int], p2: Optional[int]) -> str:
    segs = []
    if st.get("chuong"): segs.append(f"Chương {st['chuong']}")
    if st.get("muc"):    segs.append(f"Mục {st['muc']}")
    if st.get("dieu"):   segs.append(f"Điều {st['dieu']}")
    if st.get("khoan"):  segs.append(f"Khoản {st['khoan']}")
    if st.get("diem"):   segs.append(f"Điểm {st['diem']}")
    if st.get("phu_luc"):segs.append(f"Phụ lục {st['phu_luc']}")
    right = ", ".join(segs) if segs else ""
    cite = f"{van_ban}"
    if right: cite += f" — {right}"
    if p1 is not None or p2 is not None:
        a = p1 if p1 is not None else p2
        b = p2 if p2 is not None else p1
        if a is not None and b is not None:
            cite += f" | trang {a}–{b}" if a != b else f" | trang {a}"
    return cite

def build_article_header_line(st: Dict[str, Optional[str]], title: str) -> str:
    bits = []
    if st.get("dieu"): bits.append(f"Điều {st['dieu']}")
    if st.get("khoan"): bits.append(f"Khoản {st['khoan']}")
    if st.get("diem"): bits.append(f"Điểm {st['diem']}")
    head = " — ".join([bits[0], ", ".join(bits[1:])]) if len(bits) > 1 else (bits[0] if bits else "")
    if head and title: return f"{head}: {title}"
    return head or title or ""

# ============= HTML→Markdown cho bảng =============

def html_table_to_markdown(html: str) -> str:
    """
    Chuyển bảng HTML -> Markdown đơn giản.
    """
    if not html:
        return ""
    try:
        import htmltabletomd  # pip install htmltabletomd
        md = htmltabletomd.convert_table(html)
        return md.strip()
    except Exception:
        pass
    try:
        from markdownify import markdownify as mdify  # pip install markdownify
        md = mdify(html, strip=["style", "script"])
        return md.strip()
    except Exception:
        # Fallback: bóc tag
        txt = re.sub(r"<\s*br\s*/?>", "\n", html, flags=re.I)
        txt = re.sub(r"<[^>]+>", " ", txt)
        txt = re.sub(r"\s+", " ", txt).strip()
        return txt

def _normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()

# ============= Section buffer =============

@dataclass
class SectionBuf:
    title: str
    state: Dict[str, Optional[str]]
    texts: List[str]
    elem_ids: List[str]
    pages: List[int]
    table_items: List[dict]   # {"html":..., "md":..., "text":..., "page":..., "meta":...}
    image_items: List[dict]   # {"caption":..., "page":..., "meta":...}

def find_nearby_caption(idx: int, arr: List[RawElement], window: int = 2) -> str:
    cand: List[str] = []
    for j in range(max(0, idx-window), idx):
        t = _normalize_ws(arr[j].text)
        if t: 
            cand.append(t)
    for j in range(idx+1, min(len(arr), idx+1+window)):
        t = _normalize_ws(arr[j].text)
        if t: 
            cand.append(t)
    picks = []
    for s in cand:
        if re.search(r"\b(Hình|Figure|Ảnh)\b", s, flags=re.I) or len(s) <= 240:
            picks.append(s)
    cap = " ".join(picks[:2]).strip()
    return cap

# ============= Chuyển element → section → chunk =============

def smart_chunk_one_doc(
    elems: List[RawElement],
    doc_id: str,
    source_sha1: str,
    chunk_tokens: int = 520,
    overlap_tokens: int = 80,
    prepend_article_header: bool = True,
    flush_on_khoan: bool = True,
    flush_on_diem: bool = True,
    table_mode: str = "separate_chunk",
    table_md_max_chars: int = 4000,
    image_caption_window: int = 2,
) -> List[dict]:

    current = SectionBuf(
        title="",
        state={"chuong":None,"muc":None,"dieu":None,"khoan":None,"diem":None,"phu_luc":None},
        texts=[], elem_ids=[], pages=[], table_items=[], image_items=[]
    )
    sections: List[SectionBuf] = []
    title_buffer: List[str] = []

    def flush_section():
        nonlocal current
        if not current.title and title_buffer:
            current.title = " — ".join([t.strip() for t in title_buffer if t.strip()])
        if current.texts or current.table_items or current.image_items:
            sections.append(current)
        current = SectionBuf(
            title="", state={"chuong":None,"muc":None,"dieu":None,"khoan":None,"diem":None,"phu_luc":None},
            texts=[], elem_ids=[], pages=[], table_items=[], image_items=[]
        )

    def ensure_title_from_buffer():
        if not current.title and title_buffer:
            current.title = " — ".join([t.strip() for t in title_buffer if t.strip()])

    for i, el in enumerate(elems):
        t = (el.text or "").strip()
        if not t:
            continue
        etype = (el.element_type or "").lower()

        if etype in {"header", "footer", "pageheader", "pagefooter", "pagebreak", "pagenumber"}:
            continue

        if etype == "title":
            title_buffer.append(t)
            info = detect_heading_info(t)
            next_state = update_state_with(info, current.state)
            if next_state["_changed"]:
                if current.texts or current.table_items or current.image_items:
                    flush_section()
                current.state = {k:v for k,v in next_state.items() if k != "_changed"}
            continue

        # Non-title
        ensure_title_from_buffer()

        # Heading inline
        info_inline = detect_heading_info(t)
        if any(info_inline.values()):
            if info_inline.get("dieu"):
                if current.texts or current.table_items or current.image_items:
                    flush_section()
                current.state = {k:v for k,v in update_state_with(info_inline, current.state).items() if k != "_changed"}
            else:
                changed = update_state_with(info_inline, current.state)
                if changed["_changed"]:
                    if (info_inline.get("khoan") and flush_on_khoan and (current.texts or current.table_items or current.image_items)) or \
                       (info_inline.get("diem")  and flush_on_diem  and (current.texts or current.table_items or current.image_items)):
                        flush_section()
                    current.state = {k:v for k,v in changed.items() if k != "_changed"}

        # Branch theo loại element
        if etype == "table":
            meta = el.metadata or {}
            # HTML từ PDF (text_as_html) hoặc PPTX (table_html_excerpt)
            html = meta.get("text_as_html") or meta.get("table_html_excerpt")
            table_text = ""
            table_md = ""
            if html:
                table_md = html_table_to_markdown(html)
                if table_md_max_chars and len(table_md) > table_md_max_chars:
                    table_md = table_md[:table_md_max_chars] + " ..."
            else:
                # fallback: dùng text nếu không có HTML
                table_md = ""
                table_text = t

            current.table_items.append({
                "html": html,
                "md": table_md,
                "text": table_text,
                "page": el.page_number,
                "meta": meta,
            })

            if table_mode == "append_md":
                block = "\n[BẢNG]\n"
                if table_md:
                    block += table_md
                elif table_text:
                    block += table_text
                current.texts.append(block)
            continue

        if etype in {"image", "figure", "picture"}:
            meta = el.metadata or {}
            caption = find_nearby_caption(i, elems, window=image_caption_window)
            current.image_items.append({
                "caption": caption or t,  # text có thể là alt/desc
                "page": el.page_number,
                "meta": meta,
            })
            continue

        # Các loại khác → đưa vào văn bản section
        current.texts.append(t)
        if el.element_id: current.elem_ids.append(el.element_id)
        if el.page_number is not None: current.pages.append(el.page_number)

    if current.texts or current.table_items or current.image_items or title_buffer:
        flush_section()

    # Section → Chunk
    chunks: List[dict] = []
    # van_ban: tên văn bản để hiển thị trích dẫn
    van_ban = doc_id
    for sec in sections:
        page_start = min(sec.pages) if sec.pages else None
        page_end   = max(sec.pages) if sec.pages else None
        section_key = make_section_key(sec.state, fallback=sec.title)
        citation = build_citation(van_ban, sec.state, page_start, page_end)

        # 1) Chunk văn bản thường
        body = "\n".join(sec.texts).strip()
        if body:
            pre_header = build_article_header_line(sec.state, sec.title).strip() if prepend_article_header else ""
            header = (pre_header + "\n") if pre_header else ""
            text_full = (header + body).strip()

            sents = split_sentences_vi(text_full)
            parts = chunk_sentences(sents, max_tokens=chunk_tokens, overlap_tokens=overlap_tokens)
            for j, part in enumerate(parts):
                base = f"{source_sha1}|{section_key}|TXT|{j}"
                cid = hashlib.sha1(base.encode("utf-8")).hexdigest()[:16]
                meta = {
                    "section_key": section_key,
                    "van_ban": van_ban,
                    "chuong": sec.state.get("chuong"),
                    "muc": sec.state.get("muc"),
                    "dieu": sec.state.get("dieu"),
                    "khoan": sec.state.get("khoan"),
                    "diem": sec.state.get("diem"),
                    "phu_luc": sec.state.get("phu_luc"),
                    "page_start": page_start,
                    "page_end": page_end,
                    "citation": citation,
                    "element_ids": sec.elem_ids,
                }
                chunks.append({
                    "chunk_id": cid,
                    "text": part,
                    "doc_id": doc_id,
                    "source_sha1": source_sha1,
                    "metadata": meta,
                })

        # 2) Chunk BẢNG (separate_chunk)
        if sec.table_items and table_mode == "separate_chunk":
            for k, it in enumerate(sec.table_items):
                t_header = build_article_header_line(sec.state, sec.title).strip() if prepend_article_header else ""
                head_line = (t_header + "\n") if t_header else ""
                payload = ""
                if it.get("md"):
                    payload = "[BẢNG]\n" + it["md"].strip()
                elif it.get("text"):
                    payload = "[BẢNG]\n" + it["text"].strip()
                if not payload:
                    continue

                # Chia nhỏ nếu bảng dài
                sents = split_sentences_vi(payload)
                parts = chunk_sentences(sents, max_tokens=chunk_tokens, overlap_tokens=overlap_tokens) if tok_len(payload) > chunk_tokens else [payload]

                for j, part in enumerate(parts):
                    base = f"{source_sha1}|{section_key}|TBL|{k}|{j}"
                    cid = hashlib.sha1(base.encode("utf-8")).hexdigest()[:16]
                    meta = {
                        "section_key": section_key,
                        "van_ban": van_ban,
                        "chuong": sec.state.get("chuong"),
                        "muc": sec.state.get("muc"),
                        "dieu": sec.state.get("dieu"),
                        "khoan": sec.state.get("khoan"),
                        "diem": sec.state.get("diem"),
                        "phu_luc": sec.state.get("phu_luc"),
                        "page_start": it.get("page") or page_start,
                        "page_end": it.get("page") or page_end,
                        "citation": citation,
                        "has_table_html": bool(it.get("html")),
                        "table_html_excerpt": (it["html"][:1200] + " ...") if it.get("html") else None,
                    }
                    chunks.append({
                        "chunk_id": cid,
                        "text": (head_line + part).strip(),
                        "doc_id": doc_id,
                        "source_sha1": source_sha1,
                        "metadata": {k: v for k, v in meta.items() if v is not None},
                    })

        # 3) Chunk HÌNH
        if sec.image_items:
            for k, im in enumerate(sec.image_items):
                cap = (im.get("caption") or "").strip()
                if not cap:
                    continue
                payload = "[HÌNH] " + cap
                parts = [payload] if tok_len(payload) <= chunk_tokens else \
                        chunk_sentences(split_sentences_vi(payload), max_tokens=chunk_tokens, overlap_tokens=overlap_tokens)

                for j, part in enumerate(parts):
                    base = f"{source_sha1}|{section_key}|IMG|{k}|{j}"
                    cid = hashlib.sha1(base.encode("utf-8")).hexdigest()[:16]
                    meta = {
                        "section_key": section_key,
                        "van_ban": van_ban,
                        "chuong": sec.state.get("chuong"),
                        "muc": sec.state.get("muc"),
                        "dieu": sec.state.get("dieu"),
                        "khoan": sec.state.get("khoan"),
                        "diem": sec.state.get("diem"),
                        "phu_luc": sec.state.get("phu_luc"),
                        "page_start": im.get("page") or page_start,
                        "page_end": im.get("page") or page_end,
                        "citation": citation,
                        "has_image": True,
                        "image_meta_excerpt": (json.dumps(im.get("meta"))[:1200] + " ...") if im.get("meta") else None,
                    }
                    chunks.append({
                        "chunk_id": cid,
                        "text": part,
                        "doc_id": doc_id,
                        "source_sha1": source_sha1,
                        "metadata": {k: v for k, v in meta.items() if v is not None},
                    })

    return chunks

# ============= Chunk theo token =============

def chunk_sentences(sentences: List[str], max_tokens: int, overlap_tokens: int) -> List[str]:
    chunks, cur = [], []
    cur_tok = 0
    def flush():
        nonlocal cur, cur_tok
        if cur:
            chunks.append(" ".join(cur).strip()); cur = []; cur_tok = 0
    for s in sentences:
        t = tok_len(s)
        if t > max_tokens:
            txt = s
            while tok_len(txt) > max_tokens:
                if ENC:
                    ids = ENC.encode(txt); seg = ENC.decode(ids[:max_tokens]); rest = ENC.decode(ids[max_tokens:])
                else:
                    seg, rest = txt[:max_tokens*2], txt[max_tokens*2:]
                if seg.strip():
                    if cur_tok + tok_len(seg) > max_tokens: flush()
                    cur.append(seg); cur_tok += tok_len(seg); flush()
                txt = rest
            if txt.strip():
                if cur_tok + tok_len(txt) > max_tokens: flush()
                cur.append(txt); cur_tok += tok_len(txt); flush()
            continue
        if cur_tok + t <= max_tokens:
            cur.append(s); cur_tok += t
        else:
            flush()
            if chunks and overlap_tokens > 0:
                prev = chunks[-1]
                if tok_len(prev) > overlap_tokens:
                    if ENC:
                        ids = ENC.encode(prev); ov = ENC.decode(ids[-overlap_tokens:])
                    else:
                        ov = prev[-overlap_tokens*2:]
                    cur = [ov]; cur_tok = tok_len(ov)
            if cur_tok + t > max_tokens: flush()
            cur.append(s); cur_tok += t
    flush()
    return [c for c in chunks if c]

# ============= Pipeline incremental =============

def run_incremental_chunking(
    in_jsonl: Path,
    out_jsonl: Path,
    chunk_tokens: int = 520,
    overlap_tokens: int = 80,
    prepend_article_header: bool = True,
    flush_on_khoan: bool = True,
    flush_on_diem: bool = True,
    table_mode: str = "separate_chunk",
    table_md_max_chars: int = 4000,
    image_caption_window: int = 2,
) -> Tuple[int, int, int]:
    groups = group_elements_by_source(in_jsonl)
    processed = collect_processed_sha1(out_jsonl)
    if processed:
        print(f"[INFO] Đã có {len(processed)} nguồn trong {out_jsonl} — sẽ bỏ qua nếu trùng.")

    n_total_docs = len(groups)
    n_new_docs = 0
    n_chunks = 0

    for (sha1, doc_id), elems in groups.items():
        if sha1 in processed:
            print(f"[SKIP] Đã chunk trước đó: {doc_id} (SHA1={sha1[:12]}...)")
            continue

        elems = sorted(elems, key=lambda e: (e.page_number is None, e.page_number))
        print(f"[PROCESS] {doc_id} (SHA1={sha1[:12]}...) — elements={len(elems)}")

        chunks = smart_chunk_one_doc(
            elems=elems,
            doc_id=doc_id,
            source_sha1=sha1,
            chunk_tokens=chunk_tokens,
            overlap_tokens=overlap_tokens,
            prepend_article_header=prepend_article_header,
            flush_on_khoan=flush_on_khoan,
            flush_on_diem=flush_on_diem,
            table_mode=table_mode,
            table_md_max_chars=table_md_max_chars,
            image_caption_window=image_caption_window,
        )
        if not chunks:
            print(f"[WARN] Không tạo được chunk: {doc_id} (SHA1={sha1[:12]}...)")
            continue

        wrote = append_jsonl(out_jsonl, chunks)
        n_chunks += wrote
        n_new_docs += 1
        processed[sha1] = processed.get(sha1, 0) + wrote

    print(f"[DONE] docs total={n_total_docs}, new={n_new_docs} → wrote {n_chunks} chunks → {out_jsonl}")
    return n_total_docs, n_new_docs, n_chunks

# ============= CLI =============

def parse_args():
    p = argparse.ArgumentParser(description="Chunking 'smart' incremental (DOC/DOCX/PDF/PPTX) tối ưu bảng & hình ảnh")
    p.add_argument("--in_jsonl", type=str, required=True, help="Đường dẫn elements.jsonl (từ preprocess)")
    p.add_argument("--out_jsonl", type=str, default="./pccc_chunks.jsonl", help="File JSONL đầu ra (append)")
    p.add_argument("--chunk_tokens", type=int, default=520)
    p.add_argument("--overlap_tokens", type=int, default=80)
    p.add_argument("--no_article_header", action="store_true", help="Không chèn tiêu đề Điều/Khoản/Điểm vào đầu chunk")
    p.add_argument("--no_flush_khoan", action="store_true")
    p.add_argument("--no_flush_diem", action="store_true")
    p.add_argument("--table_mode", type=str, choices=["metadata_only","append_md","separate_chunk"], default="separate_chunk",
                   help="Cách xử lý Table cho embedding & metadata")
    p.add_argument("--table_md_max_chars", type=int, default=4000, help="Giới hạn ký tự Markdown bảng (nếu có)")
    p.add_argument("--image_caption_window", type=int, default=2, help="Số phần tử lân cận để gom caption ảnh")
    return p.parse_args()

def main():
    args = parse_args()
    in_jsonl = Path(args.in_jsonl)
    out_jsonl = Path(args.out_jsonl)
    assert in_jsonl.exists(), f"Không thấy file: {in_jsonl}"

    run_incremental_chunking(
        in_jsonl=in_jsonl,
        out_jsonl=out_jsonl,
        chunk_tokens=args.chunk_tokens,
        overlap_tokens=args.overlap_tokens,
        prepend_article_header=not args.no_article_header,
        flush_on_khoan=not args.no_flush_khoan,
        flush_on_diem=not args.no_flush_diem,
        table_mode=args.table_mode,
        table_md_max_chars=args.table_md_max_chars,
        image_caption_window=args.image_caption_window,
    )

if __name__ == "__main__":
    main()

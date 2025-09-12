#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Smart chunking PCCC — bảng & hình (DOC/DOCX/PDF/PPTX) + Parent Document support.

Đầu vào : elements.jsonl (từ preprocess, có source_* + doc_id)
Đầu ra   : chunks.jsonl (append)
- Sinh mẩu CON (child) như trước (TXT/TBL/IMG).
- Gắn parent cho từng child:
    * parent_level: 'khoan' nếu có, ngược lại 'dieu'
    * parent_key   : DIEU_x.KHOAN_y hoặc DIEU_x
    * parent_id    : sha1(source_sha1|parent_key|level)
    * ancestor_dieu_id: id của parent ở cấp Điều (ổn định cho mọi child trong cùng Điều)
- Sinh mẩu CHA (doc_type='parent') ứng với mỗi parent_key, gom văn bản con vào 1 đoạn lớn (để embed).
- Khử lặp tiêu đề và vệ sinh metadata (đảm bảo Chroma nhận dạng kiểu hợp lệ).

CLI:
  python chunk_pccc_smart.py --in_jsonl elements.jsonl --out_jsonl pccc_chunks.jsonl
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
    doc_id: str

def _get_src(row: dict, meta: dict) -> Tuple[str,str,str,str,str]:
    src_sha1 = row.get("source_sha1") or meta.get("file_sha1") or ""
    src_name = row.get("source_name") or meta.get("file_name") or Path(row.get("source_path","") or meta.get("file_path","")).name
    src_path = row.get("source_path") or meta.get("file_path") or ""
    src_ext  = (row.get("source_ext") or Path(src_name).suffix or "").lower()
    doc_id   = row.get("doc_id") or Path(src_name).stem
    return src_sha1, src_name, src_path, src_ext, doc_id

def _extract_element_fields(row: dict) -> Optional[RawElement]:
    text = row.get("text")
    etype = row.get("type")
    meta  = row.get("metadata") or {}
    if text is not None and etype is not None:
        page = row.get("page_number", meta.get("page_number"))
        try: page = int(page) if page is not None else None
        except Exception: page = None
        ssha1, sname, spath, sext, doc_id = _get_src(row, meta)
        return RawElement(
            element_id=row.get("element_id") or row.get("id"),
            element_type=str(etype),
            text=str(text or ""),
            metadata=meta,
            page_number=page,
            source_name=sname,
            source_path=spath,
            source_ext=sext,
            source_sha1=ssha1,
            doc_id=doc_id,
        )
    # Fallback schema cũ
    el = row.get("element") or {}
    text = el.get("text") or row.get("text") or ""
    etype = el.get("type") or row.get("element_type") or ""
    meta = el.get("metadata") or row.get("metadata") or {}
    if not etype: return None
    page = meta.get("page_number")
    try: page = int(page) if page is not None else None
    except Exception: page = None
    ssha1, sname, spath, sext, doc_id = _get_src(row, meta)
    return RawElement(
        element_id=row.get("element_id") or el.get("id"),
        element_type=str(etype),
        text=str(text or ""),
        metadata=meta,
        page_number=page,
        source_name=sname,
        source_path=spath,
        source_ext=sext,
        source_sha1=ssha1,
        doc_id=doc_id,
    )

def group_elements_by_source(in_jsonl: Path) -> Dict[Tuple[str, str], List[RawElement]]:
    groups: Dict[Tuple[str, str], List[RawElement]] = {}
    for row in read_jsonl_iter(in_jsonl):
        relem = _extract_element_fields(row)
        if not relem or not relem.source_sha1:
            continue
        key = (relem.source_sha1, relem.doc_id)
        groups.setdefault(key, []).append(relem)
    return groups

# ============= Tokenizer =============
try:
    import tiktoken
    ENC = tiktoken.get_encoding("cl100k_base")
except Exception:
    ENC = None

def tok_len(s: str) -> int:
    if ENC is None: return max(1, len(s)//2)
    return len(ENC.encode(s or ""))

def truncate_tokens(s: str, max_tokens: int) -> str:
    if tok_len(s) <= max_tokens: return s
    if ENC is None: return s[: max_tokens*2]
    ids = ENC.encode(s or ""); return ENC.decode(ids[:max_tokens])

# ============= Split câu (TV) =============
_SENT_SPLIT = re.compile(r"(?<=[\.!?…])\s+|\n+(?=[^\s])")
def split_sentences_vi(text: str) -> List[str]:
    t = (text or "").strip()
    if not t: return []
    return [s.strip() for s in _SENT_SPLIT.split(t) if s.strip()]

# ============= Heading pháp lý + enum =============
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
    cite = f"{van_ban}" + (f" — {', '.join(segs)}" if segs else "")
    if p1 is not None or p2 is not None:
        a = p1 if p1 is not None else p2
        b = p2 if p2 is not None else p1
        cite += f" | trang {a}–{b}" if (a is not None and b is not None and a != b) else (f" | trang {a}" if a is not None else "")
    return cite

def build_article_header_line(st: Dict[str, Optional[str]], title: str) -> str:
    bits = []
    if st.get("dieu"): bits.append(f"Điều {st['dieu']}")
    if st.get("khoan"): bits.append(f"Khoản {st['khoan']}")
    if st.get("diem"): bits.append(f"Điểm {st['diem']}")
    head = " — ".join([bits[0], ", ".join(bits[1:])]) if len(bits) > 1 else (bits[0] if bits else "")
    if head and title: return f"{head}: {title}"
    return head or title or ""

# ============= HTML→Markdown fallback cho bảng =============
def html_table_to_markdown(html: str) -> str:
    if not html: return ""
    try:
        import htmltabletomd
        md = htmltabletomd.convert_table(html)
        return md.strip()
    except Exception:
        pass
    try:
        from markdownify import markdownify as mdify
        md = mdify(html, strip=["style", "script"])
        return md.strip()
    except Exception:
        txt = re.sub(r"<\s*br\s*/?>", "\n", html, flags=re.I)
        txt = re.sub(r"<[^>]+>", " ", txt)
        txt = re.sub(r"\s+", " ", txt).strip()
        return txt

def _normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()

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

# ============= Section buffer =============
@dataclass
class SectionBuf:
    title: str
    state: Dict[str, Optional[str]]
    texts: List[str]
    elem_ids: List[str]
    pages: List[int]
    table_items: List[dict]
    image_items: List[dict]

# Heuristic caption ảnh
def find_nearby_caption(idx: int, arr: List[RawElement], window: int = 2) -> str:
    cand: List[str] = []
    for j in range(max(0, idx-window), idx):
        t = _normalize_ws(arr[j].text)
        if t: cand.append(t)
    for j in range(idx+1, min(len(arr), idx+1+window)):
        t = _normalize_ws(arr[j].text)
        if t: cand.append(t)
    picks = []
    for s in cand:
        if re.search(r"\b(Hình|Figure|Ảnh)\b", s, flags=re.I) or len(s) <= 240:
            picks.append(s)
    return " ".join(picks[:2]).strip()

# ===== Parent helpers =====
def parent_level_for_state(st: Dict[str, Optional[str]]) -> str:
    return "khoan" if st.get("khoan") else ("dieu" if st.get("dieu") else "doc")

def parent_key_for_state(st: Dict[str, Optional[str]]) -> str:
    if st.get("khoan") and st.get("dieu"):
        return f"DIEU_{st['dieu']}.KHOAN_{st['khoan']}"
    if st.get("dieu"):
        return f"DIEU_{st['dieu']}"
    return "DOC"

def dieu_key_for_state(st: Dict[str, Optional[str]]) -> Optional[str]:
    return f"DIEU_{st['dieu']}" if st.get("dieu") else None

def make_parent_id(source_sha1: str, parent_key: str, level: str) -> str:
    base = f"{source_sha1}|{level}|{parent_key}"
    return hashlib.sha1(base.encode("utf-8")).hexdigest()[:16]

# ============= Tiện ích metadata an toàn cho Chroma =============
def _to_scalar(v: Any) -> Optional[Any]:
    """Chroma metadata chỉ nhận str/int/float/bool. Trả None để loại bỏ."""
    if v is None:
        return None
    if isinstance(v, (str, int, float, bool)):
        return v
    try:
        s = json.dumps(v, ensure_ascii=False)
    except Exception:
        s = str(v)
    return s[:4000]  # cắt gọn

def clean_metadata(md: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    for k, v in (md or {}).items():
        sv = _to_scalar(v)
        if sv is not None:
            out[k] = sv
    return out

# ============= Khử lặp header khi thân đã chứa tiêu đề ============
_HDR_INLINE_PAT = re.compile(r"^\s*(Điều\s+\d+)(?:\s*[,\.—–-]\s*|:?\s+)", re.I)
def _starts_with_header(text: str, st: Dict[str, Optional[str]]) -> bool:
    if not text or not st.get("dieu"):
        return False
    m = _HDR_INLINE_PAT.match(text.strip())
    if not m:
        return False
    # Nếu inline “Điều X …” đã có, không cần chèn header nữa
    return True

# ============= Chuyển element → section → chunk (kèm parent) =============
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

        ensure_title_from_buffer()

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

        # Bảng
        if etype == "table":
            meta = el.metadata or {}
            html = meta.get("text_as_html")
            table_md = ""
            if html:
                table_md = html_table_to_markdown(html)
                if table_md_max_chars and len(table_md) > table_md_max_chars:
                    table_md = table_md[:table_md_max_chars] + " ..."
            else:
                table_md = t.replace("[BẢNG]", "").strip()
            current.table_items.append({
                "html": html,
                "md": table_md,
                "text": t if not table_md else "",
                "page": el.page_number,
                "meta": meta,
            })
            if table_mode == "append_md":
                block = "\n[BẢNG]\n" + (table_md or t)
                current.texts.append(block)
            continue

        # Hình
        if etype in {"image", "figure"}:
            meta = el.metadata or {}
            caption = find_nearby_caption(i, elems, window=image_caption_window)
            current.image_items.append({
                "caption": caption or t,
                "page": el.page_number,
                "meta": meta,
            })
            continue

        # Văn bản thường
        current.texts.append(t)
        if el.element_id: current.elem_ids.append(el.element_id)
        if el.page_number is not None: current.pages.append(el.page_number)

    if current.texts or current.table_items or current.image_items or title_buffer:
        flush_section()

    # Build chunks + parent aggregates
    chunks: List[dict] = []
    parent_agg: Dict[str, dict] = {}  # parent_id -> {text_parts, pages, state_level, key, level}
    dieu_parent_ids: Dict[str, str] = {}  # "DIEU_X" -> parent_id

    def ensure_parent_entry(st: Dict[str,Optional[str]], page_start: Optional[int], page_end: Optional[int]) -> Tuple[str,str,str]:
        level = parent_level_for_state(st)
        key   = parent_key_for_state(st)
        pid   = make_parent_id(source_sha1, key, level)
        # tạo (hoặc lấy) parent cấp Điều (ancestor)
        dkey = dieu_key_for_state(st)
        ancestor_id = None
        if dkey:
            ancestor_id = dieu_parent_ids.get(dkey)
            if not ancestor_id:
                ancestor_id = make_parent_id(source_sha1, dkey, "dieu")
                dieu_parent_ids[dkey] = ancestor_id
                # nếu parent hiện tại là điều, cũng cần có entry aggregate
                if "dieu" == level and ancestor_id != pid:
                    pass  # nothing
        # aggregate current parent
        if pid not in parent_agg:
            snap = {"dieu": st.get("dieu"), "khoan": st.get("khoan")}
            parent_agg[pid] = {
                "level": level,
                "key": key,
                "state": snap,
                "text_parts": [],
                "page_start": page_start,
                "page_end": page_end,
                "ancestor_dieu_id": ancestor_id if ancestor_id else (pid if level == "dieu" else None),
                "section_title": "",  # sẽ đổ ở dưới
            }
        else:
            if page_start is not None:
                ps = parent_agg[pid]["page_start"]
                parent_agg[pid]["page_start"] = min(ps, page_start) if ps is not None else page_start
            if page_end is not None:
                pe = parent_agg[pid]["page_end"]
                parent_agg[pid]["page_end"] = max(pe, page_end) if pe is not None else page_end
        return pid, key, level

    van_ban = doc_id

    def section_label(st: Dict[str,Optional[str]], title: str) -> str:
        bits = []
        if st.get("dieu"): bits.append(f"Điều {st['dieu']}")
        if st.get("khoan"): bits.append(f"Khoản {st['khoan']}")
        if st.get("diem"): bits.append(f"Điểm {st['diem']}")
        t = " — ".join([bits[0], ", ".join(bits[1:])]) if len(bits) > 1 else (bits[0] if bits else "")
        if t and title: return f"{t}: {title}"
        return t or title or ""

    for sec in sections:
        page_start = min(sec.pages) if sec.pages else None
        page_end   = max(sec.pages) if sec.pages else None
        section_key = make_section_key(sec.state, fallback=sec.title)
        citation = build_citation(van_ban, sec.state, page_start, page_end)
        sec_title = section_label(sec.state, sec.title)

        # 1) Normal text child
        body = "\n".join(sec.texts).strip()
        if body:
            pre_header = ""
            if prepend_article_header and not _starts_with_header(body, sec.state):
                hline = build_article_header_line(sec.state, sec.title).strip()
                pre_header = (hline + "\n") if hline else ""
            text_full = (pre_header + body).strip()

            # Register parent aggregate
            pid, pkey, plevel = ensure_parent_entry(sec.state, page_start, page_end)
            parent_agg[pid]["text_parts"].append(text_full)
            if sec_title and not parent_agg[pid]["section_title"]:
                parent_agg[pid]["section_title"] = sec_title

            sents = split_sentences_vi(text_full)
            parts = chunk_sentences(sents, max_tokens=chunk_tokens, overlap_tokens=overlap_tokens)
            for j, part in enumerate(parts):
                base = f"{source_sha1}|{section_key}|TXT|{j}"
                cid = hashlib.sha1(base.encode("utf-8")).hexdigest()[:16]
                # legal hierarchy
                hier = {
                    "dieu": sec.state.get("dieu"),
                    "khoan": sec.state.get("khoan"),
                    "diem": sec.state.get("diem"),
                }
                meta = clean_metadata({
                    "doc_type": "child",
                    "modality": "text",
                    "section_key": section_key,
                    "section_title": sec_title,
                    "section_label": sec_title,
                    "van_ban": van_ban,
                    **hier,
                    "page_start": page_start,
                    "page_end": page_end,
                    "citation": citation,
                    "legal_hierarchy": hier,
                    "parent_id": pid,
                    "parent_key": pkey,
                    "parent_level": plevel,
                    "ancestor_dieu_id": parent_agg[pid]["ancestor_dieu_id"] or "",
                })
                chunks.append({
                    "chunk_id": cid,
                    "text": part,
                    "doc_id": doc_id,
                    "source_sha1": source_sha1,
                    "metadata": meta,
                })

        # 2) Table children
        if sec.table_items and table_mode == "separate_chunk":
            compact_tbl_texts = []
            for it in sec.table_items:
                if it.get("md"):   compact_tbl_texts.append("[BẢNG]\n" + it["md"].strip())
                elif it.get("text"): compact_tbl_texts.append("[BẢNG]\n" + it["text"].strip())
            if compact_tbl_texts:
                pid, pkey, plevel = ensure_parent_entry(sec.state, page_start, page_end)
                parent_agg[pid]["text_parts"].append("\n\n".join(compact_tbl_texts))
                if sec_title and not parent_agg[pid]["section_title"]:
                    parent_agg[pid]["section_title"] = sec_title

            for k, it in enumerate(sec.table_items):
                t_header = build_article_header_line(sec.state, sec.title).strip() if prepend_article_header else ""
                head_line = (t_header + "\n") if (t_header and not _starts_with_header(t_header, sec.state)) else ""
                payload = ""
                if it.get("md"):
                    payload = "[BẢNG]\n" + it["md"].strip()
                elif it.get("text"):
                    payload = "[BẢNG]\n" + it["text"].strip()
                if not payload:
                    continue

                sents = split_sentences_vi(payload)
                parts = chunk_sentences(sents, max_tokens=chunk_tokens, overlap_tokens=overlap_tokens) if tok_len(payload) > chunk_tokens else [payload]

                for j, part in enumerate(parts):
                    base = f"{source_sha1}|{section_key}|TBL|{k}|{j}"
                    cid = hashlib.sha1(base.encode("utf-8")).hexdigest()[:16]
                    pid, pkey, plevel = ensure_parent_entry(sec.state, it.get("page") or page_start, it.get("page") or page_end)
                    hier = {
                        "dieu": sec.state.get("dieu"),
                        "khoan": sec.state.get("khoan"),
                        "diem": sec.state.get("diem"),
                    }
                    meta = clean_metadata({
                        "doc_type": "child",
                        "modality": "table",
                        "section_key": section_key,
                        "section_title": sec_title,
                        "section_label": sec_title,
                        "van_ban": van_ban,
                        **hier,
                        "page_start": it.get("page") or page_start,
                        "page_end": it.get("page") or page_end,
                        "citation": citation,
                        "has_table_html": bool(it.get("html")),
                        "table_html_excerpt": (it.get("html")[:1200] + " ...") if it.get("html") else "",
                        "legal_hierarchy": hier,
                        "parent_id": pid,
                        "parent_key": pkey,
                        "parent_level": plevel,
                        "ancestor_dieu_id": parent_agg[pid]["ancestor_dieu_id"] or "",
                    })
                    chunks.append({
                        "chunk_id": cid,
                        "text": (head_line + part).strip(),
                        "doc_id": doc_id,
                        "source_sha1": source_sha1,
                        "metadata": meta,
                    })

        # 3) Image children
        if sec.image_items:
            cap_join = " ".join([im.get("caption","").strip() for im in sec.image_items if im.get("caption")])
            if cap_join:
                pid, _, _ = ensure_parent_entry(sec.state, page_start, page_end)
                parent_agg[pid]["text_parts"].append("[HÌNH] " + cap_join)
                if sec_title and not parent_agg[pid]["section_title"]:
                    parent_agg[pid]["section_title"] = sec_title

            for k, im in enumerate(sec.image_items):
                cap = im.get("caption") or ""
                if not cap.strip():
                    continue
                payload = "[HÌNH] " + cap.strip()
                parts = [payload] if tok_len(payload) <= chunk_tokens else chunk_sentences(split_sentences_vi(payload), chunk_tokens, overlap_tokens)
                for j, part in enumerate(parts):
                    base = f"{source_sha1}|{section_key}|IMG|{k}|{j}"
                    cid = hashlib.sha1(base.encode("utf-8")).hexdigest()[:16]
                    pid, pkey, plevel = ensure_parent_entry(sec.state, im.get("page") or page_start, im.get("page") or page_end)
                    hier = {
                        "dieu": sec.state.get("dieu"),
                        "khoan": sec.state.get("khoan"),
                        "diem": sec.state.get("diem"),
                    }
                    meta = clean_metadata({
                        "doc_type": "child",
                        "modality": "image",
                        "section_key": section_key,
                        "section_title": sec_title,
                        "section_label": sec_title,
                        "van_ban": van_ban,
                        **hier,
                        "page_start": im.get("page") or page_start,
                        "page_end": im.get("page") or page_end,
                        "citation": citation,
                        "has_image": True,
                        "image_meta_excerpt": (json.dumps(im.get("meta"))[:1200] + " ...") if im.get("meta") else "",
                        "legal_hierarchy": hier,
                        "parent_id": pid,
                        "parent_key": pkey,
                        "parent_level": plevel,
                        "ancestor_dieu_id": parent_agg[pid]["ancestor_dieu_id"] or "",
                    })
                    chunks.append({
                        "chunk_id": cid,
                        "text": part,
                        "doc_id": doc_id,
                        "source_sha1": source_sha1,
                        "metadata": meta,
                    })

    # 4) Emit parent documents (one per parent_key)
    for pid, item in parent_agg.items():
        level = item["level"]
        key   = item["key"]
        pst, ped = item["page_start"], item["page_end"]
        st = {"dieu": item["state"].get("dieu"), "khoan": item["state"].get("khoan")}
        cite = build_citation(van_ban, st, pst, ped)
        text = "\n\n".join([t for t in item["text_parts"] if t]).strip()
        if tok_len(text) > 8000:
            text = truncate_tokens(text, 8000)
        sec_title = item.get("section_title","")
        dkey = dieu_key_for_state(st)
        ancestor_dieu_id = item.get("ancestor_dieu_id") or (make_parent_id(source_sha1, dkey, "dieu") if dkey else "")

        # liên kết cha-con hai cấp: Khoản -> Điều
        super_parent_id = ""
        if level == "khoan" and dkey:
            super_parent_id = make_parent_id(source_sha1, dkey, "dieu")

        meta = clean_metadata({
            "doc_type": "parent",
            "modality": "text",
            "parent_level": level,
            "parent_key": key,
            "van_ban": van_ban,
            "dieu": st.get("dieu"),
            "khoan": st.get("khoan"),
            "page_start": pst,
            "page_end": ped,
            "citation": cite,
            "section_title": sec_title,
            "section_label": sec_title,
            "ancestor_dieu_id": ancestor_dieu_id,
            "super_parent_id": super_parent_id,
            "legal_hierarchy": {"dieu": st.get("dieu"), "khoan": st.get("khoan")},
        })
        chunks.append({
            "chunk_id": pid,  # dùng parent_id làm id
            "text": text,
            "doc_id": doc_id,
            "source_sha1": source_sha1,
            "metadata": meta,
        })

    return chunks

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
    p = argparse.ArgumentParser(description="Chunking smart + ParentDoc (DOC/DOCX/PDF/PPTX) với metadata an toàn")
    p.add_argument("--in_jsonl", type=str, required=True, help="Đường dẫn elements.jsonl (từ preprocess)")
    p.add_argument("--out_jsonl", type=str, default="./pccc_chunks.jsonl", help="File JSONL đầu ra (append)")
    p.add_argument("--chunk_tokens", type=int, default=520)
    p.add_argument("--overlap_tokens", type=int, default=80)
    p.add_argument("--no_article_header", action="store_true")
    p.add_argument("--no_flush_khoan", action="store_true")
    p.add_argument("--no_flush_diem", action="store_true")
    p.add_argument("--table_mode", type=str, choices=["metadata_only","append_md","separate_chunk"], default="separate_chunk")
    p.add_argument("--table_md_max_chars", type=int, default=4000)
    p.add_argument("--image_caption_window", type=int, default=2)
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

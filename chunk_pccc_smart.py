#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Chunking 'smart' (incremental) cho PCCC — bản nâng cấp:
- Hợp nhất nhiều Title liên tiếp thành 1 tiêu đề section.
- Phát hiện Khoản/Điểm dạng liệt kê: "1.", "1)", "(1)", "a.", "a)", "(a)", và "Khoản x", "Điểm y".
- Chèn tiêu đề Điều/Khoản/Điểm vào ĐẦU MỖI CHUNK để tăng recall.
- Incremental: chỉ xử lý các (source_sha1, doc_id) mới, log trùng file.

Tài liệu tham chiếu:
- Cấu trúc Chương–Mục–Điều–Khoản–Điểm trong VBQPPL VN.  (see: Law on promulgation …) 
- Element types của Unstructured: Title, NarrativeText, ListItem, ... (Partitioning docs)
- Đếm token bằng tiktoken; chunk theo token & overlap (OpenAI Cookbook; RAG chunking best practices)

CLI:
  python chunk_pccc_smart_incremental.py \
      --in_jsonl ./pccc_word_elements.jsonl \
      --out_jsonl ./pccc_chunks.jsonl \
      --chunk_tokens 520 \
      --overlap_tokens 80 \
      --no_article_header   # (tùy chọn) tắt chèn tiêu đề Điều vào đầu chunk
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

# =========================
# I/O JSONL helpers
# =========================

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
                sys.stderr.write(f"[WARN] Bỏ dòng hỏng ở {path}:{ln}\n")

def append_jsonl(path: Path, rows: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n

# =========================
# Thu thập SHA-1 đã xử lý
# =========================

def collect_processed_sha1(out_jsonl: Path) -> Dict[str, int]:
    """Tập SHA-1 đã có trong file chunks (để skip)."""
    seen: Dict[str, int] = {}
    if not out_jsonl.exists():
        return seen
    for row in read_jsonl_iter(out_jsonl):
        sha1 = row.get("source_sha1") or (row.get("metadata") or {}).get("source_sha1")
        if sha1:
            seen[sha1] = seen.get(sha1, 0) + 1
    return seen

# =========================
# Gom element theo nguồn
# =========================

@dataclass
class RawElement:
    element_id: Optional[str]
    element_type: str
    text: str
    metadata: Dict[str, Any]
    source_name: str
    source_path: str
    page_number: Optional[int]

def _extract_element_fields(row: dict) -> RawElement:
    el = row.get("element") or {}
    element_id = row.get("element_id") or el.get("id")
    etype = row.get("element_type") or el.get("type") or ""
    text = row.get("text") or el.get("text") or ""
    metadata = row.get("metadata") or el.get("metadata") or {}
    src_name = row.get("source_name") or Path(row.get("source_path","")).name
    src_path = row.get("source_path") or ""
    page = metadata.get("page_number")
    try:
        page = int(page) if page is not None else None
    except Exception:
        page = None
    return RawElement(
        element_id=element_id,
        element_type=str(etype),
        text=str(text or ""),
        metadata=metadata or {},
        source_name=str(src_name or ""),
        source_path=str(src_path or ""),
        page_number=page,
    )

def group_elements_by_source(in_jsonl: Path) -> Tuple[Dict[Tuple[str,str], List[RawElement]], Dict[str, List[str]]]:
    """
    Trả về:
      - groups[(source_sha1, doc_id)] = [RawElement, ...]
      - dup_paths[source_sha1] = [source_path1, source_path2, ...] nếu có nhiều path cho cùng sha1
    """
    groups: Dict[Tuple[str,str], List[RawElement]] = {}
    sha1_to_paths: Dict[str, List[str]] = {}

    for row in read_jsonl_iter(in_jsonl):
        sha1 = row.get("source_sha1")
        if not sha1:
            continue
        doc_id = row.get("doc_id") or (Path(row.get("source_path","")).stem or sha1)
        relem = _extract_element_fields(row)

        key = (sha1, doc_id)
        groups.setdefault(key, []).append(relem)

        p = relem.source_path or row.get("source_path") or ""
        if p:
            lst = sha1_to_paths.setdefault(sha1, [])
            if p not in lst:
                lst.append(p)

    dup_paths = {h: lst for h, lst in sha1_to_paths.items() if len(lst) > 1}
    return groups, dup_paths

# =========================
# Tiktoken để hạn mức token
# =========================

try:
    import tiktoken  # OpenAI tokenizer
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
        return s[: max_tokens * 2]
    ids = ENC.encode(s)
    return ENC.decode(ids[:max_tokens])

# =========================
# Heuristic tách câu (TV)
# =========================

_SENT_SPLIT = re.compile(r"(?<=[\.!?…])\s+|\n+(?=[^\s])")

def split_sentences_vi(text: str) -> List[str]:
    sents = [s.strip() for s in _SENT_SPLIT.split(text.strip()) if s.strip()]
    return sents

# =========================
# Nhận diện heading pháp lý (mở rộng)
# =========================

# Các pattern chính thức
HDR_PATTERNS = {
    "chuong": re.compile(r"^\s*Chương\s+([IVXLCDM]+|\d+)\b", re.IGNORECASE),
    "muc":    re.compile(r"^\s*Mục\s+([A-Z0-9]+)\b", re.IGNORECASE),
    "dieu":   re.compile(r"^\s*Điều\s+(\d+)\b", re.IGNORECASE),
    "khoan":  re.compile(r"^\s*Khoản\s+(\d+)\b", re.IGNORECASE),
    "diem":   re.compile(r"^\s*Điểm\s+([a-z])\b", re.IGNORECASE),
    "phuluc": re.compile(r"^\s*Phụ\s*lục\s+([A-Z0-9]+)\b", re.IGNORECASE),
}

# Mẫu liệt kê thường gặp cho Khoản/Điểm: 1., 1), (1), a., a), (a)
ENUM_KHOAN = [
    re.compile(r"^\s*(\d+)[\.\)]\s+"),
    re.compile(r"^\s*\((\d+)\)\s+"),
]
ENUM_DIEM = [
    re.compile(r"^\s*([a-z])[\.\)]\s+"),
    re.compile(r"^\s*\(([a-z])\)\s+"),
]

def detect_heading_info(text: str) -> Dict[str, Optional[str]]:
    info = {"chuong":None, "muc":None, "dieu":None, "khoan":None, "diem":None, "phu_luc":None}
    t = (text or "").strip()
    if not t:
        return info
    for k, pat in HDR_PATTERNS.items():
        m = pat.search(t)
        if m:
            val = m.group(1)
            if k == "phuluc": k = "phu_luc"
            info[k] = val

    # Bổ sung: nhận dạng liệt kê ở đầu dòng
    if info["khoan"] is None:
        for p in ENUM_KHOAN:
            m = p.match(t)
            if m:
                info["khoan"] = m.group(1)
                break
    if info["diem"] is None:
        for p in ENUM_DIEM:
            m = p.match(t)
            if m:
                info["diem"] = m.group(1)
                break
    return info

def update_state_with(info: Dict[str, Optional[str]], state: Dict[str, Optional[str]]) -> Dict[str, Optional[str]]:
    """Cập nhật state theo mức độ; khi thay Điều → reset Khoản/Điểm; khi thay Khoản → reset Điểm."""
    st = dict(state)
    changed = False
    if info.get("chuong"):
        st["chuong"] = info["chuong"]; changed = True
    if info.get("muc"):
        st["muc"] = info["muc"]; changed = True
    if info.get("phu_luc"):
        st["phu_luc"] = info["phu_luc"]; changed = True
    if info.get("dieu"):
        st["dieu"] = info["dieu"]; st["khoan"] = None; st["diem"] = None; changed = True
    if info.get("khoan"):
        st["khoan"] = info["khoan"]; st["diem"] = None; changed = True
    if info.get("diem"):
        st["diem"] = info["diem"]; changed = True
    st["_changed"] = changed
    return st

def make_section_key(state: Dict[str, Optional[str]], fallback_title: str="") -> str:
    parts = []
    if state.get("chuong"): parts.append(f"CHUONG_{state['chuong']}")
    if state.get("muc"):    parts.append(f"MUC_{state['muc']}")
    if state.get("dieu"):   parts.append(f"DIEU_{state['dieu']}")
    if state.get("khoan"):  parts.append(f"KHOAN_{state['khoan']}")
    if state.get("diem"):   parts.append(f"DIEM_{state['diem']}")
    if state.get("phu_luc"):parts.append(f"PHULUC_{state['phu_luc']}")
    if not parts and fallback_title:
        ft = re.sub(r"\s+", "_", fallback_title.strip())[:60]
        parts.append(ft)
    return ".".join(parts) or "SECTION"

def build_citation(van_ban: str, state: Dict[str, Optional[str]], page_start: Optional[int], page_end: Optional[int]) -> str:
    segs = []
    if state.get("chuong"): segs.append(f"Chương {state['chuong']}")
    if state.get("muc"):    segs.append(f"Mục {state['muc']}")
    if state.get("dieu"):   segs.append(f"Điều {state['dieu']}")
    if state.get("khoan"):  segs.append(f"Khoản {state['khoan']}")
    if state.get("diem"):   segs.append(f"Điểm {state['diem']}")
    if state.get("phu_luc"):segs.append(f"Phụ lục {state['phu_luc']}")
    right = ", ".join(segs) if segs else ""
    cite = f"{van_ban}"
    if right: cite += f" — {right}"
    if page_start is not None or page_end is not None:
        a = page_start if page_start is not None else page_end
        b = page_end if page_end is not None else page_start
        if a is not None and b is not None:
            cite += f" | trang {a}–{b}" if a != b else f" | trang {a}"
    return cite

def build_article_header_line(state: Dict[str, Optional[str]], title: str) -> str:
    """Chuẩn hoá header 'Điều x — Khoản y, Điểm z: <title>' để tăng recall."""
    bits = []
    if state.get("dieu"): bits.append(f"Điều {state['dieu']}")
    if state.get("khoan"): bits.append(f"Khoản {state['khoan']}")
    if state.get("diem"): bits.append(f"Điểm {state['diem']}")
    head = " — ".join([bits[0], ", ".join(bits[1:])]) if len(bits) > 1 else (bits[0] if bits else "")
    if head and title:
        return f"{head}: {title}"
    if head:
        return head
    return title or ""

# =========================
# Chunking theo token
# =========================

def chunk_sentences(sentences: List[str], max_tokens: int, overlap_tokens: int) -> List[str]:
    chunks = []
    cur: List[str] = []
    cur_tok = 0

    def flush():
        nonlocal cur, cur_tok
        if cur:
            chunks.append(" ".join(cur).strip())
            cur = []
            cur_tok = 0

    for s in sentences:
        t = tok_len(s)
        if t > max_tokens:
            # cắt cứng câu quá dài
            txt = s
            while tok_len(txt) > max_tokens:
                if ENC:
                    ids = ENC.encode(txt)
                    seg = ENC.decode(ids[:max_tokens])
                    rest = ENC.decode(ids[max_tokens:])
                else:
                    seg = txt[:max_tokens*2]
                    rest = txt[max_tokens*2:]
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
            # tạo overlap theo token từ chunk trước
            flush()
            if chunks and overlap_tokens > 0:
                prev = chunks[-1]
                if tok_len(prev) > overlap_tokens:
                    if ENC:
                        ids = ENC.encode(prev)
                        overlap_txt = ENC.decode(ids[-overlap_tokens:])
                    else:
                        overlap_txt = prev[-overlap_tokens*2:]
                    cur = [overlap_txt]
                    cur_tok = tok_len(overlap_txt)
            if cur_tok + t > max_tokens:
                flush()
            cur.append(s); cur_tok += t

    flush()
    return [c for c in chunks if c]

# =========================
# Gom element → section → chunk (với hợp nhất Title)
# =========================

@dataclass
class SectionBuf:
    title: str
    state: Dict[str, Optional[str]]
    texts: List[str]
    elem_ids: List[str]
    pages: List[int]

def smart_chunk_one_doc(
    elems: List[RawElement],
    doc_id: str,
    source_sha1: str,
    chunk_tokens: int = 520,
    overlap_tokens: int = 80,
    prepend_article_header: bool = True,
    flush_on_khoan: bool = True,
    flush_on_diem: bool = True,
) -> List[dict]:
    """
    elems: các element của 1 tài liệu, nên theo thứ tự tự nhiên từ Unstructured.
    - Hợp nhất Title liên tiếp: gom vào title_buffer cho tới khi gặp nội dung kế tiếp.
    - Phát hiện heading/enum: nếu gặp Điều/Khoản/Điểm mới → flush section (tuỳ tham số).
    - Chèn header 'Điều x — Khoản y, Điểm z: <title>' vào đầu chunk (tuỳ chọn).
    """
    current = SectionBuf(
        title="",
        state={"chuong":None,"muc":None,"dieu":None,"khoan":None,"diem":None,"phu_luc":None},
        texts=[], elem_ids=[], pages=[]
    )
    sections: List[SectionBuf] = []
    title_buffer: List[str] = []  # hợp nhất nhiều Title liên tiếp

    def flush_section():
        nonlocal current
        # nếu chưa gán title mà buffer có nội dung → set title
        if not current.title and title_buffer:
            current.title = " — ".join([t.strip() for t in title_buffer if t.strip()])
        if current.texts:
            sections.append(current)
        # reset
        current = SectionBuf(
            title="", 
            state={"chuong":None,"muc":None,"dieu":None,"khoan":None,"diem":None,"phu_luc":None},
            texts=[], elem_ids=[], pages=[]
        )

    def ensure_title_from_buffer():
        if not current.title and title_buffer:
            current.title = " — ".join([t.strip() for t in title_buffer if t.strip()])

    for el in elems:
        t = (el.text or "").strip()
        if not t:
            continue
        etype = (el.element_type or "").lower()

        if etype == "title":
            # tích lũy title, đồng thời cập nhật state nếu phát hiện heading
            title_buffer.append(t)
            info = detect_heading_info(t)
            # Nếu title chứa Điều/Khoản/Điểm mới → flush section cũ trước (nếu có nội dung)
            next_state = update_state_with(info, current.state)
            if next_state["_changed"]:
                if current.texts:
                    flush_section()
                current.state = {k:v for k,v in next_state.items() if k != "_changed"}
            # chưa flush: chờ tới khi có nội dung thực
            # (không thêm title vào texts để tránh lặp)
            continue

        # Non-title: trước hết chốt title từ buffer nếu có
        ensure_title_from_buffer()

        # Nhận diện heading inline ở đầu đoạn (Điều/Khoản/Điểm dạng liệt kê)
        info_inline = detect_heading_info(t)
        if any(info_inline.values()):
            # nếu Điều thay đổi → luôn flush
            if info_inline.get("dieu"):
                if current.texts:
                    flush_section()
                current.state = {k:v for k,v in update_state_with(info_inline, current.state).items() if k != "_changed"}
            else:
                # Khoản/Điểm: tuỳ chọn flush để chia nhỏ hơn
                changed_state = update_state_with(info_inline, current.state)
                if changed_state["_changed"]:
                    if (info_inline.get("khoan") and flush_on_khoan and current.texts) or \
                       (info_inline.get("diem")  and flush_on_diem  and current.texts):
                        flush_section()
                    current.state = {k:v for k,v in changed_state.items() if k != "_changed"}

        # Ghi nội dung vào section hiện tại
        current.texts.append(t)
        if el.element_id: current.elem_ids.append(el.element_id)
        if el.page_number is not None: current.pages.append(el.page_number)

    # flush cuối
    if current.texts or title_buffer:
        flush_section()

    # 2) Section → Chunk
    chunks: List[dict] = []
    van_ban = doc_id  # fallback nếu không có tên văn bản chuẩn
    for sec in sections:
        body = "\n".join(sec.texts).strip()
        if not body:
            continue

        # Xây header textual để tăng recall
        pre_header = ""
        if prepend_article_header:
            pre_header = build_article_header_line(sec.state, sec.title).strip()
        header = ""
        if pre_header:
            header = pre_header + "\n"

        text_full = (header + body).strip()
        sents = split_sentences_vi(text_full)
        parts = chunk_sentences(sents, max_tokens=chunk_tokens, overlap_tokens=overlap_tokens)

        page_start = min(sec.pages) if sec.pages else None
        page_end   = max(sec.pages) if sec.pages else None

        section_key = make_section_key(sec.state, fallback_title=sec.title)
        citation = build_citation(van_ban, sec.state, page_start, page_end)

        for j, part in enumerate(parts):
            base = f"{source_sha1}|{section_key}|{j}"
            cid = hashlib.sha1(base.encode("utf-8")).hexdigest()[:16]
            chunk = {
                "chunk_id": cid,
                "text": part,
                "doc_id": doc_id,
                "source_sha1": source_sha1,
                "metadata": {
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
                    "source_name": "",  # có thể gán từ elems[0] nếu muốn
                    "source_path": "",
                }
            }
            chunks.append(chunk)

    return chunks

# =========================
# Pipeline incremental
# =========================

def run_incremental_chunking(
    in_jsonl: Path,
    out_jsonl: Path,
    chunk_tokens: int = 520,
    overlap_tokens: int = 80,
    prepend_article_header: bool = True,
    flush_on_khoan: bool = True,
    flush_on_diem: bool = True,
) -> Tuple[int, int, int]:
    """
    Trả về: (n_docs_tổng, n_docs_mới, n_chunks_ghi)
    - n_docs_* theo (source_sha1, doc_id)
    """
    groups, dup_paths = group_elements_by_source(in_jsonl)

    if dup_paths:
        print("[WARN] Phát hiện file trùng (cùng nội dung, khác path) trong input:")
        for h, lst in dup_paths.items():
            print(f"  SHA1={h}")
            for p in lst:
                print(f"    - {p}")

    processed = collect_processed_sha1(out_jsonl)
    if processed:
        print(f"[INFO] Đã thấy {len(processed)} nguồn (SHA-1) trong {out_jsonl} — sẽ bỏ qua nếu trùng.")

    n_total_docs = len(groups)
    n_new_docs = 0
    n_chunks = 0

    for (sha1, doc_id), elems in groups.items():
        if sha1 in processed:
            print(f"[SKIP] Đã chunk trước đó: {doc_id} (SHA1={sha1[:12]}...)")
            continue

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

# =========================
# CLI
# =========================

def parse_args():
    p = argparse.ArgumentParser(description="Chunking 'smart' incremental cho PCCC (nâng cấp)")
    p.add_argument("--in_jsonl", type=str, required=True, help="Đường dẫn pccc_word_elements.jsonl")
    p.add_argument("--out_jsonl", type=str, default="./pccc_chunks.jsonl", help="File JSONL đầu ra (append)")
    p.add_argument("--chunk_tokens", type=int, default=520, help="Số token tối đa mỗi chunk")
    p.add_argument("--overlap_tokens", type=int, default=80, help="Số token overlap giữa các chunk liên tiếp")
    p.add_argument("--no_article_header", action="store_true", help="Không chèn tiêu đề Điều/Khoản/Điểm vào đầu chunk")
    p.add_argument("--no_flush_khoan", action="store_true", help="Không tách section khi gặp Khoản mới")
    p.add_argument("--no_flush_diem", action="store_true", help="Không tách section khi gặp Điểm mới")
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
    )

if __name__ == "__main__":
    main()

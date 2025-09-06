#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Smart chunking cho văn bản luật PCCC đã được partition (từ Unstructured API)
và normalize ở bước trước (JSONL các element).

Nguyên tắc:
- Giữ ranh giới pháp lý: không gộp qua Điều khác nhau; ưu tiên không vượt Khoản/Điểm nếu đã nhận diện.
- Bảng (Table) để nguyên 1 chunk.
- Chunk theo nhóm element cùng section, sau đó cắt theo câu ~ max_chars; có overlap theo số câu.
- Tự động tạo 'citation' (van_ban + dieu/khoan/diem + page range nếu có).
- Kết quả: JSONL sẵn sàng cho indexing/embedding.
"""

from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Any, Iterable, Optional, Tuple
import json, re, hashlib, uuid
from dataclasses import dataclass, field
from collections import defaultdict

###############
# Cấu trúc dữ liệu
###############

@dataclass
class Elem:
    element_id: str
    type: str
    text: str
    metadata: Dict[str, Any]

    @property
    def page(self) -> Optional[int]:
        md = self.metadata or {}
        return md.get("page_number")

    @property
    def section_keys(self) -> Dict[str, Optional[str]]:
        md = self.metadata or {}
        return {
            "van_ban": md.get("van_ban"),
            "chuong": md.get("chuong"),
            "muc": md.get("muc"),
            "dieu": md.get("dieu"),
            "khoan": md.get("khoan"),
            "diem": md.get("diem"),
            "phu_luc": md.get("phu_luc"),
        }

@dataclass
class Chunk:
    doc_id: str
    source_sha1: str
    chunk_id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    element_ids: List[str] = field(default_factory=list)

###############
# Tiện ích
###############

_SENT_SPLIT = re.compile(r"(?<=[\.?!…])\s+(?=[A-ZÀ-ỴA-ZĐ0-9])|(?<=\.)\s+\n", re.UNICODE)

def split_sentences(text: str) -> List[str]:
    text = re.sub(r"\s+\n", "\n", text.strip())
    bullets = re.split(r"\n(?=[\-•●◦▪·]|\d+\)|[a-z]\))", text)
    out = []
    for seg in bullets:
        seg = seg.strip()
        if not seg:
            continue
        parts = re.split(_SENT_SPLIT, seg)
        for p in parts:
            p = p.strip()
            if p:
                out.append(p)
    return out

def mk_chunk_id(doc_id: str, section_key: str, idx: int) -> str:
    base = f"{doc_id}::{section_key}::{idx}"
    return hashlib.sha1(base.encode("utf-8")).hexdigest()[:16]

def build_section_key(md: Dict[str, Any]) -> str:
    keys = []
    for k in ("van_ban","chuong","muc","dieu","khoan","diem","phu_luc"):
        v = md.get(k)
        if v:
            keys.append(f"{k}:{v}")
    return "|".join(keys) if keys else "unknown"

def infer_heading_title(elem: Elem) -> Optional[str]:
    if elem.type and elem.type.lower() == "title":
        return elem.text.strip() or None
    if re.match(r"^\s*(Chương|Mục|Điều)\b", elem.text, flags=re.IGNORECASE):
        return elem.text.strip()
    return None

def compose_citation(md: Dict[str, Any], page_start: Optional[int], page_end: Optional[int]) -> str:
    vb = md.get("van_ban") or "Văn bản"
    parts = []
    if md.get("dieu"):  parts.append(f"Điều {md['dieu']}")
    if md.get("khoan"): parts.append(f"Khoản {md['khoan']}")
    if md.get("diem"):  parts.append(f"Điểm {md['diem']}")
    if not parts and md.get("muc"):
        parts.append(f"Mục {md['muc']}")
    if not parts and md.get("chuong"):
        parts.append(f"Chương {md['chuong']}")
    sec = ", ".join(parts) if parts else None

    pages = None
    if page_start and page_end and page_start != page_end:
        pages = f"trang {page_start}–{page_end}"
    elif page_start:
        pages = f"trang {page_start}"

    if sec and pages:
        return f"{vb} — {sec} ({pages})"
    if sec:
        return f"{vb} — {sec}"
    if pages:
        return f"{vb} ({pages})"
    return vb

###############
# Lõi chunking
###############

def group_elements_by_section(elems: List[Elem]) -> List[List[Elem]]:
    groups: List[List[Elem]] = []
    cur: List[Elem] = []
    cur_key: Tuple = None

    def key_of(e: Elem) -> Tuple:
        md = e.section_keys
        return (md.get("van_ban"), md.get("dieu") or "", md.get("khoan") or "", md.get("diem") or "",
                md.get("muc") or "", md.get("chuong") or "")

    for e in elems:
        if e.type and e.type.lower() == "table":
            if cur:
                groups.append(cur); cur = []
            groups.append([e])
            cur_key = None
            continue

        k = key_of(e)
        if cur_key is None:
            cur_key = k
            cur = [e]
            continue

        if k != cur_key:
            groups.append(cur)
            cur = [e]
            cur_key = k
        else:
            cur.append(e)

    if cur:
        groups.append(cur)
    return groups

def chunk_paragraphs_to_sized_chunks(
    paras: List[str],
    doc_id: str,
    common_md: Dict[str, Any],
    element_ids: List[str],
    pages: List[Optional[int]],
    max_chars: int = 1400,
    min_chars: int = 400,
    overlap_sents: int = 2,
    section_key: str = "unknown",
    start_index: int = 0,
) -> List[Chunk]:
    all_text = "\n".join([p.strip() for p in paras if p and p.strip()])
    sents = split_sentences(all_text)
    if not sents:
        sents = [all_text] if all_text else []

    chunks: List[Chunk] = []
    buf: List[str] = []
    start_idx = 0

    def flush(idx: int):
        nonlocal buf, start_idx
        if not buf:
            return
        txt = " ".join(buf).strip()
        if not txt:
            buf = []
            start_idx = idx
            return
        p_start = min([p for p in pages if p is not None], default=None)
        p_end   = max([p for p in pages if p is not None], default=None)
        citation = compose_citation(common_md, p_start, p_end)

        global_idx = start_index + len(chunks)
        cid = mk_chunk_id(doc_id, section_key, global_idx)
        meta = {
            **common_md,
            "page_start": p_start,
            "page_end": p_end,
            "citation": citation,
            "section_key": section_key,
            "heading_title": common_md.get("heading_title"),
        }
        chunks.append(Chunk(
            doc_id=doc_id,
            source_sha1=common_md.get("source_sha1",""),
            chunk_id=cid,
            text=txt,
            metadata=meta,
            element_ids=element_ids[:],
        ))
        if overlap_sents > 0:
            tail = buf[-overlap_sents:] if len(buf) > overlap_sents else buf[:]
            buf = tail[:]
        else:
            buf = []
        start_idx = idx

    for i, s in enumerate(sents):
        cur_len = sum(len(x) + 1 for x in buf)
        if cur_len + len(s) > max_chars and cur_len >= min_chars:
            flush(i)
        buf.append(s)

    flush(len(sents))
    return chunks

def smart_chunk_elements(
    raw_elements: List[Dict[str, Any]],
    max_chars: int = 1400,
    min_chars: int = 400,
    overlap_sents: int = 2,
) -> List[Chunk]:
    elems: List[Elem] = []
    for el in raw_elements:
        elems.append(Elem(
            element_id=el.get("element_id") or el.get("id") or str(uuid.uuid4()),
            type=el.get("type") or "",
            text=(el.get("text") or "").strip(),
            metadata=el.get("metadata") or {},
        ))

    # đoán doc_id/source_sha1
    doc_id = None
    source_sha1 = None
    for e in elems:
        if e.metadata.get("doc_id"):
            doc_id = e.metadata.get("doc_id")
        if e.metadata.get("source_sha1"):
            source_sha1 = e.metadata.get("source_sha1")
    if not doc_id and raw_elements:
        doc_id = raw_elements[0].get("doc_id", "")
    if not source_sha1 and raw_elements:
        source_sha1 = raw_elements[0].get("source_sha1", "")

    groups = group_elements_by_section(elems)
    out_chunks: List[Chunk] = []
    sec_counters = defaultdict(int)  # đếm chunk theo section_key

    for grp in groups:
        # metadata gốc
        md0 = grp[0].metadata.copy() if grp and grp[0].metadata else {}
        md0.setdefault("doc_id", doc_id or "")
        md0.setdefault("source_sha1", source_sha1 or "")

        heading = None
        for e in grp:
            h = infer_heading_title(e)
            if h:
                heading = h
                break
        md0["heading_title"] = heading
        section_key = build_section_key(md0)

        # Bảng đơn – tạo ngay chunk riêng với chỉ số toàn cục
        if len(grp) == 1 and (grp[0].type or "").lower() == "table":
            e = grp[0]
            md = {**(e.metadata or {})}
            md.setdefault("doc_id", doc_id or "")
            md.setdefault("source_sha1", source_sha1 or "")
            md["heading_title"] = infer_heading_title(e)
            p = e.page
            citation = compose_citation(md, p, p)

            idx = sec_counters[section_key]
            cid = mk_chunk_id(doc_id or "", section_key, idx)
            sec_counters[section_key] += 1

            out_chunks.append(Chunk(
                doc_id=doc_id or "",
                source_sha1=source_sha1 or "",
                chunk_id=cid,
                text=e.text or "[TABLE]",
                metadata={
                    **md,
                    "page_start": p,
                    "page_end": p,
                    "citation": citation,
                    "section_key": section_key,
                    "heading_title": md.get("heading_title"),
                },
                element_ids=[e.element_id],
            ))
            continue

        # Gom paragraphs, ids, pages
        paras, eids, pages = [], [], []
        for e in grp:
            if e.text:
                paras.append(e.text)
            eids.append(e.element_id)
            pages.append(e.page)

        start_index = sec_counters[section_key]
        chs = chunk_paragraphs_to_sized_chunks(
            paras=paras,
            doc_id=doc_id or "",
            common_md=md0,
            element_ids=eids,
            pages=pages,
            max_chars=max_chars,
            min_chars=min_chars,
            overlap_sents=overlap_sents,
            section_key=section_key,
            start_index=start_index,
        )
        out_chunks.extend(chs)
        sec_counters[section_key] += len(chs)

    return out_chunks

###############
# I/O helpers
###############

def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out

def write_jsonl(records: Iterable[Dict[str, Any]], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n

###############
# CLI nhỏ
###############

def run_chunking(in_jsonl: Path, out_jsonl: Path,
                 max_chars: int = 1400, min_chars: int = 400, overlap_sents: int = 2) -> int:
    raw = read_jsonl(in_jsonl)
    chunks = smart_chunk_elements(raw, max_chars=max_chars, min_chars=min_chars, overlap_sents=overlap_sents)
    serial = []
    for c in chunks:
        serial.append({
            "doc_id": c.doc_id,
            "source_sha1": c.source_sha1,
            "chunk_id": c.chunk_id,
            "text": c.text,
            "metadata": c.metadata,
            "element_ids": c.element_ids,
        })
    n = write_jsonl(serial, out_jsonl)
    print(f"✅ Đã tạo {n} chunks → {out_jsonl}")
    return n

if __name__ == "__main__":
    in_jsonl  = Path("./pccc_word_elements.jsonl")
    out_jsonl = Path("./pccc_chunks.jsonl")
    run_chunking(in_jsonl, out_jsonl, max_chars=1400, min_chars=400, overlap_sents=2)

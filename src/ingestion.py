"""Read the raw HTML snapshots, split them into articles (Điều) and normalize text.

Raw layer  : data/raw/*.html (untouched snapshot, never edited)
Clean layer: data/clean/units.jsonl (one record per article, NFC-normalized)
"""
from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

from bs4 import BeautifulSoup

from .models import DocumentPage

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ROOT / "data" / "sources.json"

_ARTICLE_RE = re.compile(r"^Điều\s+(\d+)\.\s*(.*)$")
_CHAPTER_RE = re.compile(r"^Chương\s+([IVXLC\d]+)\b")
_EFFECTIVE_RE = re.compile(r"có hiệu lực thi hành từ ngày (\d{1,2}) tháng (\d{1,2}) năm (\d{4})")


def normalize_text(text: str) -> str:
    """NFC Unicode, unify spaces, keep paragraph breaks (one paragraph per line)."""
    text = unicodedata.normalize("NFC", text).replace(" ", " ").replace("​", "")
    lines = [re.sub(r"[ \t\r\f\v]+", " ", line).strip() for line in text.split("\n")]
    lines = [re.sub(r" ([.,;:)])", r"\1", re.sub(r"\( ", "(", line)) for line in lines]
    return "\n".join(line for line in lines if line)


def _paragraphs(html_path: Path) -> list[str]:
    soup = BeautifulSoup(html_path.read_text(encoding="utf-8"), "html.parser")
    first = soup.find(id="dieu_1")
    if first is None:
        raise ValueError(f"{html_path}: cannot find id=dieu_1 (layout changed?)")
    # Chapter heading of chapter 1 sits right before Điều 1; start from it.
    start = soup.find(id="chuong_1") or first
    out = []
    for p in [start, *start.find_all_next("p")]:
        # line breaks inside one <p> are HTML source wrapping, not paragraph breaks
        txt = normalize_text(p.get_text(" ").replace("\n", " "))
        if txt:
            out.append(txt)
    return out


def load_law(source: dict, errors: list[dict]) -> list[DocumentPage]:
    """Return one DocumentPage per article, with chapter and version metadata."""
    paras = _paragraphs(ROOT / source["raw_path"])
    pages: list[DocumentPage] = []
    chapter, chapter_title, pending_chapter = None, None, False
    current: dict | None = None
    end_markers = ("Luật này được Quốc hội", "Luật này đã được Quốc hội")

    def flush():
        if current:
            body = "\n".join(current["lines"])
            pages.append(DocumentPage(
                document_id=source["document_id"], source=source["url"], page=current["n"],
                text=body,
                metadata={
                    "unit_id": f"{source['document_id']}#dieu-{current['n']}",
                    "article": current["n"], "article_title": current["title"],
                    "chapter": current["chapter"], "chapter_title": current["chapter_title"] or "",
                    "law_number": source["law_number"], "version_year": source["version_year"],
                    "is_current": source["is_current"], "effective_date": source["effective_date"],
                    "char_count": len(body),
                }))

    for line in paras:
        if line.startswith(end_markers):
            break
        m_ch = _CHAPTER_RE.match(line)
        if m_ch and len(line) < 40:
            flush(); current = None
            chapter, chapter_title, pending_chapter = m_ch.group(1), None, True
            continue
        if pending_chapter and not _ARTICLE_RE.match(line) and line.isupper():
            chapter_title, pending_chapter = line, False
            continue
        if re.match(r"^Mục\s+\d+\b", line) and len(line) < 120:
            continue  # section label (Mục 1, "Mục 3. HÒA GIẢI"); the article keeps its chapter
        m_art = _ARTICLE_RE.match(line)
        if m_art:
            flush()
            pending_chapter = False
            current = {"n": int(m_art.group(1)), "title": m_art.group(2).strip(),
                       "chapter": chapter, "chapter_title": chapter_title, "lines": [line]}
            continue
        if current is None:
            continue
        if line.isupper() and len(line) < 120 and not line[0].isdigit():
            continue  # section title (Mục ... title) in capitals
        current["lines"].append(line)
    flush()

    numbers = [p.page for p in pages]
    expected = list(range(1, source["expected_articles"] + 1))
    if numbers != expected:
        errors.append({"document_id": source["document_id"], "type": "article_sequence",
                       "detail": f"got {len(numbers)} articles, missing={sorted(set(expected) - set(numbers))}"})
    for p in pages:
        if p.metadata["char_count"] < 80:
            errors.append({"document_id": p.document_id, "type": "short_unit",
                           "detail": f"Điều {p.page}: {p.metadata['char_count']} chars"})
    return pages


def check_effective_date(pages: list[DocumentPage], source: dict, errors: list[dict]) -> None:
    """effective_date in sources.json must be found verbatim in the law's own text."""
    found = None
    for p in pages:
        m = _EFFECTIVE_RE.search(p.text)
        if m:
            d, mth, y = (int(x) for x in m.groups())
            found = f"{y:04d}-{mth:02d}-{d:02d}"
            break
    if found != source["effective_date"]:
        errors.append({"document_id": source["document_id"], "type": "effective_date_mismatch",
                       "detail": f"sources.json={source['effective_date']} text={found}"})


def load_corpus(errors: list[dict]) -> list[DocumentPage]:
    sources = json.loads(SOURCES.read_text(encoding="utf-8"))["sources"]
    pages: list[DocumentPage] = []
    for s in sources:
        law_pages = load_law(s, errors)
        check_effective_date(law_pages, s, errors)
        pages.extend(law_pages)
    return pages


def write_clean(pages: list[DocumentPage], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for p in pages:
            f.write(json.dumps({"document_id": p.document_id, "source": p.source, "page": p.page,
                                "text": p.text, "metadata": p.metadata}, ensure_ascii=False) + "\n")

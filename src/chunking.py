"""Two chunking strategies. Both work inside one article, so a chunk never mixes articles."""
from .models import Chunk, DocumentPage

_META_KEYS = ("unit_id", "article", "article_title", "chapter", "chapter_title",
              "law_number", "version_year", "is_current", "effective_date")


def _meta(page: DocumentPage, idx: int, start: int, end: int) -> dict:
    m = {k: page.metadata[k] for k in _META_KEYS}
    m.update({"chunk_index": idx, "char_start": start, "char_end": end})
    return m


def fixed_size(page: DocumentPage, size: int, overlap: int) -> list[Chunk]:
    """Sliding window of `size` characters with `overlap`; ignores sentence/clause borders."""
    if not 0 <= overlap < size:
        raise ValueError("need 0 <= overlap < size")
    text, chunks, start, idx = page.text, [], 0, 0
    while start < len(text):
        end = min(start + size, len(text))
        chunks.append(Chunk(f"{page.document_id}:{page.page}:{idx}", page.document_id, page.source,
                            page.page, text[start:end], _meta(page, idx, start, end)))
        if end == len(text):
            break
        start, idx = end - overlap, idx + 1
    return chunks


def by_paragraph(page: DocumentPage, max_size: int) -> list[Chunk]:
    """Group whole clauses (one paragraph = one khoản/điểm) up to max_size characters.

    Every chunk starts with the article heading "Điều N. Title" so it can be read alone.
    A single paragraph longer than max_size is split at sentence ends ('. ' / '; ').
    """
    lines = page.text.split("\n")
    heading, body = lines[0], lines[1:]
    budget = max_size - len(heading) - 1
    pieces: list[str] = []
    for para in body:
        if len(para) <= budget:
            pieces.append(para)
            continue
        cur = ""
        for sent in _sentences(para):
            if cur and len(cur) + len(sent) > budget:
                pieces.append(cur.strip()); cur = ""
            cur += sent
        if cur.strip():
            pieces.append(cur.strip())

    groups: list[list[str]] = []
    for piece in pieces:
        if groups and sum(len(x) + 1 for x in groups[-1]) + len(piece) <= budget:
            groups[-1].append(piece)
        else:
            groups.append([piece])
    if not groups:
        groups = [[]]

    chunks, cursor = [], 0
    for idx, g in enumerate(groups):
        text = "\n".join([heading, *g])
        start = page.text.find(g[0], cursor) if g else 0
        end = (start + len("\n".join(g))) if g else len(heading)
        cursor = max(cursor, start)
        chunks.append(Chunk(f"{page.document_id}:{page.page}:{idx}", page.document_id, page.source,
                            page.page, text, _meta(page, idx, start, end)))
    return chunks


def _sentences(para: str) -> list[str]:
    out, cur = [], ""
    for i, ch in enumerate(para):
        cur += ch
        if ch in ".;" and (i + 1 == len(para) or para[i + 1] == " "):
            out.append(cur); cur = ""
    if cur:
        out.append(cur)
    return out

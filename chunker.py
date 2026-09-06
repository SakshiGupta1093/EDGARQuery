"""Split cleaned filing sections into overlapping fixed-size chunks.

This is the naive baseline: a sliding window of `chunk_size` tokens advancing
by `chunk_size - overlap` each step, with no regard for sentence or paragraph
boundaries. It exists to get an end-to-end retrieval path working and to serve
as the control that smarter strategies are measured against.

Tokens here are whitespace-separated words, not model tokens - see
`count_tokens` for what that means for the embedding model's input limit.

Chunk text is sliced out of the original string by character offset rather
than rebuilt by joining tokens, so a chunk is always a verbatim substring of
the section it came from. That keeps the offsets in each chunk's metadata
usable for citing back to the source.
"""
import re
from pathlib import Path

# A "token" for windowing purposes: any run of non-whitespace characters.
_TOKEN_RE = re.compile(r"\S+")

DEFAULT_CHUNK_SIZE = 500
DEFAULT_OVERLAP = 50


def count_tokens(text: str) -> int:
    """Count tokens the way the chunker windows them (whitespace-separated)."""
    return sum(1 for _ in _TOKEN_RE.finditer(text))


def _token_spans(text: str) -> list[tuple[int, int]]:
    """Return (start, end) character offsets for every token in `text`."""
    return [(m.start(), m.end()) for m in _TOKEN_RE.finditer(text)]


def chunk_text(
    text: str,
    *,
    source: str,
    section: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
    metadata: dict | None = None,
) -> list[dict]:
    """Window `text` into overlapping chunks, newest metadata attached to each.

    Returns a list of dicts with the chunk `text` plus the metadata needed to
    cite it later: `source`, `section`, `chunk_index`, the character span it
    occupies in `text`, and anything passed in `metadata`.
    """
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    if overlap < 0:
        raise ValueError(f"overlap must be non-negative, got {overlap}")
    if overlap >= chunk_size:
        # The window would advance by zero or fewer tokens and never terminate.
        raise ValueError(f"overlap ({overlap}) must be less than chunk_size ({chunk_size})")

    spans = _token_spans(text)
    if not spans:
        return []

    step = chunk_size - overlap
    chunks: list[dict] = []
    start = 0
    while start < len(spans):
        window = spans[start : start + chunk_size]
        char_start, char_end = window[0][0], window[-1][1]

        # A trailing window this short sits entirely inside the previous
        # chunk's overlap, so emitting it would only duplicate text.
        is_last = start + chunk_size >= len(spans)
        if is_last and chunks and len(window) <= overlap:
            break

        chunks.append(
            {
                "text": text[char_start:char_end],
                "source": source,
                "section": section,
                "chunk_index": len(chunks),
                "token_count": len(window),
                "char_start": char_start,
                "char_end": char_end,
                "chunk_id": f"{source}:{section}:{len(chunks)}",
                **(metadata or {}),
            }
        )
        if is_last:
            break
        start += step

    return chunks


def chunk_sections(
    sections: dict[str, str],
    *,
    source: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
    metadata: dict | None = None,
) -> list[dict]:
    """Chunk every section of a parsed filing.

    `chunk_index` restarts at zero per section, so a chunk is identified by the
    (source, section, chunk_index) triple rather than by index alone.
    """
    chunks = []
    for section, text in sections.items():
        chunks.extend(
            chunk_text(
                text,
                source=source,
                section=section,
                chunk_size=chunk_size,
                overlap=overlap,
                metadata=metadata,
            )
        )
    return chunks


def chunk_filing(
    sections: dict[str, str],
    filing: dict,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> list[dict]:
    """Chunk a filing's sections, carrying its identifying fields onto each chunk.

    `filing` is a dict as returned by `EdgarClient.search_filings`. The ticker,
    form type, filing date and URL ride along on every chunk so a retrieved
    result can be cited without a second lookup.
    """
    source = f"{filing['ticker']}_{filing['form_type']}_{filing['filing_date']}"
    metadata = {
        key: filing[key]
        for key in ("ticker", "form_type", "filing_date", "accession_number", "url")
        if key in filing
    }
    return chunk_sections(
        sections,
        source=source,
        chunk_size=chunk_size,
        overlap=overlap,
        metadata=metadata,
    )


def source_from_path(path: str | Path) -> str:
    """Derive a source identifier from a downloaded filing's filename.

    Files are saved as `{ticker}_{form}_{date}.htm`, so the stem is already the
    identifier used elsewhere.
    """
    return Path(path).stem

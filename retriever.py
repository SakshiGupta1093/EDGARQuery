"""Answer a query string with the top-k most similar chunks from the index.

This ties the pieces together: embed the query with the same model that built
the index, search, and hand back the hits with enough context to read and cite
them.

Recovering the chunk text is the part with substance. The index stores vectors
and the sidecar stores metadata, but neither stores the text - so a hit's text
is sliced back out of the source filing using the `source`, `section` and
character offsets recorded at chunk time. That keeps the index small and makes
the text and the citation come from the same place, but it does mean the raw
filings have to still be on disk, and that they have to parse to the same text
they did when indexed. Both conditions are checked rather than assumed: a
missing filing or a shifted offset degrades to a hit without text and a
warning, not a wrong quotation.
"""
import warnings
from pathlib import Path

import embedder as em
import filing_parser as fp
import vector_store as vs
from edgar_client import RAW_DATA_DIR

DEFAULT_TOP_K = 5


class Retriever:
    """Search a `VectorStore`, returning hits with their text and metadata."""

    def __init__(
        self,
        store: vs.VectorStore,
        raw_dir: str | Path = RAW_DATA_DIR,
        model_name: str | None = None,
    ):
        self.store = store
        self.raw_dir = Path(raw_dir)
        # Default to whatever model built the index, not the library default:
        # embedding a query with a different model than the documents returns
        # confident nonsense rather than an error.
        self.model_name = model_name or store.model_name
        self._section_cache: dict[str, dict[str, str]] = {}

    @classmethod
    def load(
        cls,
        store_dir: str | Path = vs.DEFAULT_STORE_DIR,
        raw_dir: str | Path = RAW_DATA_DIR,
    ) -> "Retriever":
        """Load the store at `store_dir` and wrap it in a retriever."""
        return cls(vs.VectorStore.load(store_dir), raw_dir=raw_dir)

    def _find_filing(self, source: str) -> Path | None:
        """Locate the downloaded filing a chunk's `source` refers to.

        `source` is the filename stem the downloader used, but the extension
        varies - primary documents are usually .htm, older filings .txt.
        """
        matches = sorted(self.raw_dir.glob(f"{source}.*"))
        return matches[0] if matches else None

    def _sections_for(self, source: str) -> dict[str, str]:
        """Parse and cache the cleaned sections of one filing."""
        if source not in self._section_cache:
            path = self._find_filing(source)
            if path is None:
                self._section_cache[source] = {}
            else:
                self._section_cache[source] = fp.parse_filing_file(path)
        return self._section_cache[source]

    def _text_for(self, hit: dict) -> str | None:
        """Slice a hit's text back out of its source filing, or None."""
        sections = self._sections_for(hit["source"])
        if not sections:
            warnings.warn(
                f"no filing for {hit['source']!r} under {self.raw_dir}; returning "
                f"hit {hit['chunk_id']} without text",
                stacklevel=3,
            )
            return None

        section_text = sections.get(hit["section"])
        if section_text is None:
            warnings.warn(
                f"{hit['source']!r} no longer yields a {hit['section']!r} section; "
                f"returning hit {hit['chunk_id']} without text",
                stacklevel=3,
            )
            return None

        start, end = hit["char_start"], hit["char_end"]
        if end > len(section_text):
            # The parser's output moved since the index was built, so these
            # offsets point somewhere else now. Quoting the slice anyway would
            # attribute the wrong text to the citation.
            warnings.warn(
                f"offsets {start}-{end} exceed the current {hit['section']!r} section "
                f"of {hit['source']!r} ({len(section_text)} chars); the index is stale "
                f"relative to the parser. Re-index to restore chunk text.",
                stacklevel=3,
            )
            return None

        return section_text[start:end]

    def retrieve(
        self,
        query: str,
        k: int = DEFAULT_TOP_K,
        include_text: bool = True,
    ) -> list[dict]:
        """Return the `k` chunks most similar to `query`, best first.

        Each hit is its chunk metadata plus `score`, `rank` and - unless
        `include_text` is off - the chunk `text`. Fewer than `k` hits come back
        if the index holds fewer vectors than that.
        """
        if not query or not query.strip():
            raise ValueError("query must be a non-empty string")

        query_vector = em.embed_query(query, model_name=self.model_name)
        hits = self.store.search(query_vector, k=k)

        if include_text:
            for hit in hits:
                hit["text"] = self._text_for(hit)
        return hits


def retrieve(
    query: str,
    k: int = DEFAULT_TOP_K,
    store_dir: str | Path = vs.DEFAULT_STORE_DIR,
    raw_dir: str | Path = RAW_DATA_DIR,
    include_text: bool = True,
) -> list[dict]:
    """One-shot retrieval against a store on disk.

    Convenient for a single query; loading the store and the embedding model
    costs a few seconds, so hold a `Retriever` instead when issuing several.
    """
    return Retriever.load(store_dir, raw_dir=raw_dir).retrieve(
        query, k=k, include_text=include_text
    )


def format_hits(hits: list[dict], text_chars: int = 300) -> str:
    """Render hits as readable text, for CLI output and eyeballing results."""
    if not hits:
        return "(no results)"

    blocks = []
    for hit in hits:
        header = (
            f"[{hit['rank']}] score {hit['score']:.4f}  "
            f"{hit['source']} / {hit['section']} / chunk {hit['chunk_index']}"
        )
        if hit.get("url"):
            header += f"\n    {hit['url']}"
        text = hit.get("text")
        if text is None:
            body = "    (text unavailable - see warnings)"
        else:
            collapsed = " ".join(text.split())
            if len(collapsed) > text_chars:
                collapsed = collapsed[:text_chars].rstrip() + "..."
            body = f"    {collapsed}"
        blocks.append(f"{header}\n{body}")
    return "\n\n".join(blocks)

"""Persist embedded filing chunks in a FAISS index with a metadata sidecar.

The index holds vectors only - FAISS has nowhere to put a chunk's source
filing, section or character offsets - so every store is a pair of files that
have to travel together:

    index.faiss     the vectors, in insertion order
    metadata.json   one entry per vector, in the same order, plus a manifest

Row `i` of the index and entry `i` of the metadata describe the same chunk, and
that correspondence is the only thing making a search result citable. The two
are wrapped in a single `VectorStore` rather than returned as a loose pair so
there is no opportunity to save, load or slice one without the other.

The index is an `IndexFlatIP`: an exact (non-approximate) inner-product search.
Because `embedder` returns unit vectors, inner product is cosine similarity.
Flat means no training step and exact results, which is what we want while the
corpus is small and retrieval quality is still being measured - swapping in an
approximate index is a later optimization, not a starting point.

The manifest records which embedding model produced the vectors. Querying an
index with a different model than built it returns plausible-looking nonsense
rather than an error, so `search` checks the query's dimension and `load`
surfaces the model name for callers to verify.
"""
import json
from pathlib import Path

import numpy as np

INDEX_FILENAME = "index.faiss"
METADATA_FILENAME = "metadata.json"

DEFAULT_STORE_DIR = Path("data/index")


def _json_default(value):
    """Coerce numpy scalars that json cannot serialize on its own."""
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _as_float32_2d(vectors: np.ndarray) -> np.ndarray:
    """Return `vectors` as the contiguous float32 2-D array FAISS requires."""
    array = np.asarray(vectors)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2:
        raise ValueError(f"expected a 2-D array of vectors, got shape {array.shape}")
    return np.ascontiguousarray(array, dtype=np.float32)


class VectorStore:
    """A FAISS index and the chunk metadata describing each of its rows."""

    def __init__(self, index, metadata: list[dict], model_name: str):
        if index.ntotal != len(metadata):
            raise ValueError(
                f"index holds {index.ntotal} vectors but got {len(metadata)} metadata "
                f"entries; the two must correspond row for row"
            )
        self.index = index
        self.metadata = metadata
        self.model_name = model_name

    def __len__(self) -> int:
        return self.index.ntotal

    @property
    def dimension(self) -> int:
        return self.index.d

    @classmethod
    def build(
        cls,
        vectors: np.ndarray,
        metadata: list[dict],
        model_name: str,
    ) -> "VectorStore":
        """Build an index over `vectors`, pairing row `i` with `metadata[i]`."""
        import faiss

        vectors = _as_float32_2d(vectors)
        if len(vectors) != len(metadata):
            raise ValueError(
                f"got {len(vectors)} vectors but {len(metadata)} metadata entries"
            )
        if len(vectors) == 0:
            raise ValueError("cannot build an index from zero vectors")

        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        return cls(index, list(metadata), model_name)

    def save(self, directory: str | Path = DEFAULT_STORE_DIR) -> Path:
        """Write the index and its metadata sidecar, returning the directory."""
        import faiss

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(directory / INDEX_FILENAME))

        sidecar = {
            "model_name": self.model_name,
            "dimension": self.dimension,
            "count": len(self),
            "metric": "inner_product",
            "chunks": self.metadata,
        }
        with open(directory / METADATA_FILENAME, "w", encoding="utf-8") as handle:
            json.dump(sidecar, handle, ensure_ascii=False, indent=2, default=_json_default)

        return directory

    @classmethod
    def load(cls, directory: str | Path = DEFAULT_STORE_DIR) -> "VectorStore":
        """Load a store previously written by `save`."""
        import faiss

        directory = Path(directory)
        index_path = directory / INDEX_FILENAME
        metadata_path = directory / METADATA_FILENAME
        for path in (index_path, metadata_path):
            if not path.exists():
                raise FileNotFoundError(f"no vector store at {directory}: missing {path.name}")

        index = faiss.read_index(str(index_path))
        with open(metadata_path, encoding="utf-8") as handle:
            sidecar = json.load(handle)

        store = cls(index, sidecar["chunks"], sidecar["model_name"])
        # A dimension mismatch here means the two files came from different
        # runs, which would otherwise show up as silently wrong search results.
        if sidecar["dimension"] != store.dimension:
            raise ValueError(
                f"metadata records dimension {sidecar['dimension']} but the index is "
                f"{store.dimension}-dimensional; the sidecar does not match the index"
            )
        return store

    def search(self, query_vector: np.ndarray, k: int = 5) -> list[dict]:
        """Return the `k` nearest chunks to `query_vector`, best first.

        Each result is the chunk's metadata plus its similarity `score` and
        `rank`. Fewer than `k` results come back if the index is smaller than
        `k`. Note the metadata carries no chunk text - use the `source`,
        `section` and character offsets to recover it from the parsed filing.
        """
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")

        query = _as_float32_2d(query_vector)
        if query.shape[0] != 1:
            raise ValueError(f"expected a single query vector, got {query.shape[0]}")
        if query.shape[1] != self.dimension:
            raise ValueError(
                f"query is {query.shape[1]}-dimensional but the index is "
                f"{self.dimension}-dimensional; was it embedded with {self.model_name}?"
            )

        scores, indices = self.index.search(query, min(k, len(self)))

        results = []
        for rank, (score, row) in enumerate(zip(scores[0], indices[0]), start=1):
            # FAISS pads with -1 when it cannot fill k.
            if row < 0:
                continue
            results.append({**self.metadata[row], "score": float(score), "rank": rank})
        return results


def build_store(
    vectors: np.ndarray,
    metadata: list[dict],
    model_name: str,
) -> VectorStore:
    """Convenience wrapper mirroring `embed_chunks`'s return signature."""
    return VectorStore.build(vectors, metadata, model_name)

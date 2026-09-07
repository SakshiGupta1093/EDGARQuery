"""Encode filing chunks into dense vectors with BAAI/bge-small-en-v1.5.

The model is loaded lazily and cached per name, because loading it costs a few
seconds and every caller in a single run wants the same one.

Two details of the bge family matter here and are handled rather than left to
the caller:

* Its input window is 512 wordpiece tokens. Anything longer is silently
  truncated - the tail is dropped with no error - so `embed_chunks` measures
  chunk lengths against the real tokenizer and warns when text is being lost.
* Retrieval is asymmetric. Documents are embedded as-is, but queries are meant
  to carry an instruction prefix (`QUERY_PREFIX`); using it on one side only,
  or on both, both degrade recall. Hence separate `embed_texts` and
  `embed_query` entry points.

Vectors are L2-normalized, which makes the inner product equal to cosine
similarity and lets FAISS use its plain `IndexFlatIP` for cosine search.
"""
import warnings

import numpy as np

DEFAULT_MODEL_NAME = "BAAI/bge-small-en-v1.5"

# bge asks for this instruction on the query side of a retrieval pair only.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# The model's own hard limit; text beyond it is dropped at encode time.
MODEL_MAX_TOKENS = 512

_model_cache: dict[str, object] = {}


def load_model(model_name: str = DEFAULT_MODEL_NAME):
    """Load (and cache) the sentence-transformers model.

    Imported inside the function so that the rest of the pipeline - parsing and
    chunking - stays importable without torch installed.
    """
    if model_name not in _model_cache:
        from sentence_transformers import SentenceTransformer

        _model_cache[model_name] = SentenceTransformer(model_name)
    return _model_cache[model_name]


def embedding_dimension(model_name: str = DEFAULT_MODEL_NAME) -> int:
    """Return the model's output vector width."""
    model = load_model(model_name)
    # Renamed in sentence-transformers 6.0; requirements.txt is unpinned, so
    # accept either spelling rather than tying the module to one version.
    getter = getattr(model, "get_embedding_dimension", None) or (
        model.get_sentence_embedding_dimension
    )
    return getter()


def count_model_tokens(texts: list[str], model_name: str = DEFAULT_MODEL_NAME) -> list[int]:
    """Count wordpiece tokens per text, as the model itself counts them.

    This is the number that matters for truncation, and it is not the word
    count the chunker windows on - filing prose runs well over one token per
    word once figures and tickers are involved.
    """
    from transformers import logging as hf_logging

    tokenizer = load_model(model_name).tokenizer
    # Measuring past the model's window is the whole point here, so silence
    # the tokenizer's "sequence longer than maximum" advisory; the caller is
    # deliberately asking for the untruncated length.
    previous_verbosity = hf_logging.get_verbosity()
    hf_logging.set_verbosity_error()
    try:
        encoded = tokenizer(texts, add_special_tokens=True, truncation=False)["input_ids"]
    finally:
        hf_logging.set_verbosity(previous_verbosity)
    return [len(ids) for ids in encoded]


def embed_texts(
    texts: list[str],
    model_name: str = DEFAULT_MODEL_NAME,
    batch_size: int = 32,
    show_progress: bool = False,
) -> np.ndarray:
    """Embed raw strings, returning an (n, dim) float32 array of unit vectors."""
    if not texts:
        return np.zeros((0, embedding_dimension(model_name)), dtype=np.float32)

    model = load_model(model_name)
    vectors = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=show_progress,
    )
    return vectors.astype(np.float32, copy=False)


def embed_query(query: str, model_name: str = DEFAULT_MODEL_NAME) -> np.ndarray:
    """Embed a search query, with the instruction prefix bge expects.

    Returns a 1-D vector so it can be compared against the document matrix
    directly; FAISS callers will need to reshape to (1, dim).
    """
    return embed_texts([QUERY_PREFIX + query], model_name=model_name)[0]


def embed_chunks(
    chunks: list[dict],
    model_name: str = DEFAULT_MODEL_NAME,
    batch_size: int = 32,
    show_progress: bool = False,
) -> tuple[np.ndarray, list[dict]]:
    """Embed chunk dicts, returning the vectors and their metadata.

    The returned array's row `i` corresponds to `metadata[i]`, and metadata
    entries are the chunk dicts minus their `text`: the vector replaces the
    text as the searchable representation, and keeping both would duplicate
    the whole corpus in the index sidecar.

    Warns if any chunk exceeds the model's input window, since the excess is
    dropped silently and would otherwise go unnoticed until recall was bad.
    """
    if not chunks:
        return np.zeros((0, embedding_dimension(model_name)), dtype=np.float32), []

    texts = [chunk["text"] for chunk in chunks]

    token_counts = count_model_tokens(texts, model_name=model_name)
    over = [count for count in token_counts if count > MODEL_MAX_TOKENS]
    if over:
        warnings.warn(
            f"{len(over)} of {len(chunks)} chunks exceed the model's "
            f"{MODEL_MAX_TOKENS}-token window (largest {max(over)}); the text past "
            f"the limit will be dropped from the embedding. Reduce the chunker's "
            f"chunk_size.",
            stacklevel=2,
        )

    vectors = embed_texts(
        texts, model_name=model_name, batch_size=batch_size, show_progress=show_progress
    )

    metadata = []
    for chunk, token_count in zip(chunks, token_counts):
        entry = {key: value for key, value in chunk.items() if key != "text"}
        entry["model_token_count"] = token_count
        entry["truncated"] = token_count > MODEL_MAX_TOKENS
        metadata.append(entry)

    return vectors, metadata

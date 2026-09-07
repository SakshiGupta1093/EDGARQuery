# Build Progress

Quick-glance status log for the EDGARQuery system. One line of context per task.

## Phase 1: Foundation

- [x] **Repo init** — `.gitignore`, `requirements.txt`, README stub.
- [x] **SEC EDGAR client** — ticker→CIK lookup, list 10-K/10-Q filings with dates and document URLs.
- [x] **Filing downloader** — fetch the primary document HTML for a filing and cache it locally under `data/raw/`.
- [x] **HTML→text extraction** — strip markup with BeautifulSoup/lxml, keep readable filing body text.
- [x] **Section segmentation** — isolate MD&A, Risk Factors, and Financial Statements by item heading.
- [x] **Text cleaning** — normalize encoding, drop page furniture, rejoin numbers split across table cells.
- [x] **Chunking** — naive fixed-size window with overlap, carrying filing/section metadata and char offsets. Counts words, not model tokens; see the open question below.
- [x] **Embeddings** — encode chunks with `bge-small-en-v1.5` into normalized 384-d vectors, warning when a chunk overruns the model's window.
- [x] **FAISS index** — exact `IndexFlatIP` over the normalized vectors, saved to `data/index/` as `index.faiss` plus a `metadata.json` sidecar and reloaded bit-exactly.
- [x] **Retrieval** — top-k cosine search from a query string, returning hits with source metadata and chunk text sliced back out of the filing by offset.
- [ ] **Generation** — answer with `Qwen2.5-1.5B-Instruct` over retrieved context.
- [ ] **Citations** — attach filing, section, and URL to every claim in the answer.
- [ ] **CLI** — end-to-end ask-a-question entry point.

## Phase 2: Retrieval Quality

- [ ] **Hybrid search** — combine BM25 keyword scores with dense vectors.
- [ ] **Reranking** — cross-encoder pass over the top-N candidates.
- [ ] **Metadata filtering** — restrict by ticker, form type, fiscal period, section.
- [ ] **Chunking sweep** — compare chunk sizes and overlaps against retrieval metrics.
- [ ] **Query rewriting** — expand or decompose vague questions before retrieval.

## Phase 3: Deep Evaluation

- [ ] **Eval set** — question/answer pairs with gold source chunks from real filings.
- [ ] **Retrieval metrics** — recall@k, MRR, nDCG.
- [ ] **Answer metrics** — faithfulness, groundedness, citation correctness.
- [ ] **LLM-as-judge** — scored rubric for answer quality.
- [ ] **Regression harness** — one command to score a change against the previous baseline.

## Phase 4: Multi-Agent Architecture

- [ ] **Framework choice** — pick LangChain vs CrewAI and record the reasoning.
- [ ] **Router agent** — classify the question and pick a retrieval strategy.
- [ ] **Retrieval agent** — run and refine searches until it has enough evidence.
- [ ] **Synthesis agent** — compose the cited answer from gathered evidence.
- [ ] **Comparison workflows** — multi-filing and multi-company questions across periods.
- [ ] **Tool use** — numeric lookups and calculations over filing data.

## Phase 5: Multimodal Retrieval

- [ ] **Table extraction** — pull financial tables out of filing HTML as structured rows.
- [ ] **Table retrieval** — make tables searchable alongside text chunks.
- [ ] **Figure extraction** — pull charts and images from filings.
- [ ] **CLIP embeddings** — encode figures for image/text search.
- [ ] **ColPali page retrieval** — page-image retrieval over rendered filing pages.
- [ ] **Modality fusion** — merge text, table, and image hits into one ranked result set.

## Decisions Log

- **SEC submissions API over full-text search** — `data.sec.gov/submissions/CIK*.json` gives every filing for a company in one request, no pagination or scraping.
- **Descriptive User-Agent with contact email** — SEC blocks requests without one; required by their developer policy.
- **Section splitting on text, not the DOM** — filers use wildly different markup for headings, but the "Item N." text convention is mandated, so regex over extracted text is more portable than CSS/XPath selectors.
- **Longest-span wins for duplicate item headings** — every filing repeats its headings in a table of contents; the real section is always the longest span between one heading and the next.
- **Clean after segmentation, not before** — heading detection reads the line structure that cleaning collapses, so the two passes cannot be reordered.
- **Page footers must carry a page number to be dropped** — matching bare "Form 10-K" would delete prose about the filing, and treating bare `(4)` as a page marker would silently turn negative four into nothing.
- **Rejoin table cells rather than drop stray symbols** — `get_text` puts each cell on its own line, stranding `$`, `%` and the parentheses around negative numbers; dropping them would corrupt figures, so they are reattached to their number.
- **Chunks are sliced by character offset, not rejoined from tokens** — every chunk is a verbatim substring of its section, so the stored offsets stay usable for citing back to the source.
- **Vectors are L2-normalized at encode time** — inner product then equals cosine similarity, so the FAISS step can use a plain `IndexFlatIP` with no normalization pass of its own.
- **Metadata drops the chunk text** — the vector is the searchable representation; carrying the text into the index sidecar too would duplicate the whole corpus.
- **Query and document embeddings are separate calls** — bge is asymmetric and wants its instruction prefix on the query side only, so `embed_query` applies it and `embed_texts` does not.
- **Truncation warns instead of raising** — a truncated embedding is still usable, just degraded, and failing the run would block the pipeline over a tuning problem.
- **Flat exact index, not an approximate one** — `IndexFlatIP` needs no training and returns exact neighbours, so retrieval metrics measure the embeddings rather than an approximation's recall loss. Revisit only when the corpus outgrows brute force.
- **Index and metadata are one object, never a loose pair** — row `i` of the index and entry `i` of the sidecar are the same chunk, and that correspondence is what makes a hit citable, so `VectorStore` owns both and refuses to construct if their lengths disagree.
- **The sidecar records the embedding model** — querying an index with a different model than built it returns plausible nonsense rather than an error, so the model name is persisted and the query's dimension is checked on every search.
- **Hit text is sliced from the filing, not stored in the index** — the offsets recorded at chunk time already locate it, so the text and the citation come from one source and the index stays small. The cost is that the raw filings must still be on disk and must parse to the same text they did when indexed; both are checked, and a miss yields a hit with no text rather than a misattributed quotation.
- **The retriever embeds queries with the index's model, not the library default** — taken from the sidecar, so a changed default cannot silently mismatch the query against the documents.

## Open Questions

- **The chunker's word-based window cannot respect the embedder's token limit.** Measured against `bge-small-en-v1.5` (512 wordpiece tokens) on the AAPL 10-K: at the current 500-word setting, 46 of 48 chunks overflow, mean 693 tokens, worst 1,165, with 9,069 tokens dropped in total. Shrinking the window does not fix it cleanly, because the tokens-per-word ratio swings from 1.17 on prose to 2.33 on number-dense tables:

  | words | chunks | mean tokens | max tokens | over 512 |
  |------:|-------:|------------:|-----------:|---------:|
  | 500 | 48 | 693 | 1,165 | 46/48 |
  | 350 | 68 | 489 | 830 | 21/68 |
  | 250 | 95 | 351 | 612 | 6/95 |
  | 200 | 118 | 284 | 509 | 0/118 |

  Only 200 words eliminates overflow, and it wastes more than half the window on prose chunks. The real fix is to window on the model's tokenizer rather than on whitespace, so every chunk fills the budget without exceeding it. Worth doing as part of the Phase 2 chunking sweep rather than by picking a smaller word count.

- **Retrieval works but ranks factual lookups poorly, and truncation is not the reason.** "What risks does the company face from relying on single-source suppliers?" ranks three correct Risk Factors chunks at 0.72-0.77. But "Why did Greater China net sales decrease?" puts a raw segment table first and the chunk actually containing the answer second, and "How much cash did the company return through buybacks?" returns cash-flow and commercial-paper chunks rather than the repurchase disclosure. Truncation was the obvious suspect and it is wrong: the Greater China answer sits at token 406 of its chunk, inside the 512-token window the model reads. The likelier causes are 500-word chunks averaging several topics into one vector, and dense-only search being weak on numeric lookups. Both have Phase 2 tasks already — hybrid BM25 search and reranking — so measure there rather than guessing. Needs the Phase 3 eval set before any of it can be scored.

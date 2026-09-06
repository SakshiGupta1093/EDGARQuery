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
- [ ] **Embeddings** — encode chunks with `bge-small-en-v1.5`.
- [ ] **FAISS index** — build, persist, and reload the vector index with its metadata sidecar.
- [ ] **Retrieval** — top-k similarity search returning chunks plus source metadata.
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

## Open Questions

- **Chunk size is 500 *words*, but `bge-small-en-v1.5` caps input at 512 *wordpiece tokens*.** English prose runs ~1.3-1.5 wordpiece tokens per word, and filing tables full of figures run higher, so a 500-word chunk is likely ~650-750 tokens and would be silently truncated at embedding time — the tail of every chunk would never be indexed. Resolve before the embeddings step, either by dropping the window to ~350 words or by windowing on the real tokenizer. Needs measuring with the actual tokenizer rather than the estimate above.

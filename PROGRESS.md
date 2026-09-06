# Build Progress

Quick-glance status log for the EDGARQuery system. One line of context per task.

## Phase 1: Foundation

- [x] **Repo init** — `.gitignore`, `requirements.txt`, README stub.
- [x] **SEC EDGAR client** — ticker→CIK lookup, list 10-K/10-Q filings with dates and document URLs.
- [x] **Filing downloader** — fetch the primary document HTML for a filing and cache it locally under `data/raw/`.
- [x] **HTML→text extraction** — strip markup with BeautifulSoup/lxml, keep readable filing body text.
- [x] **Section segmentation** — isolate MD&A, Risk Factors, and Financial Statements by item heading.
- [ ] **Chunking** — token-aware chunks with overlap, carrying filing/section metadata.
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

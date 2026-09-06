"""SEC EDGAR API client for discovering and downloading company filings.

Provides lookup of a company's CIK by ticker, listing of its 10-K/10-Q
filings (form type, filing date, and document URL) via SEC's submissions API,
and download of a filing's raw document to local storage. Parsing filing
content is out of scope for this module.
"""
from pathlib import Path

import requests

# SEC requires a descriptive User-Agent identifying the requester and a contact.
# See https://www.sec.gov/os/webmaster-faq#developers
SEC_USER_AGENT = "SEC-Filings-RAG research-tool gupta.s8@northeastern.edu"

TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL_TEMPLATE = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
ARCHIVES_BASE_URL = "https://www.sec.gov/Archives/edgar/data"

VALID_FORM_TYPES = {"10-K", "10-Q"}

RAW_DATA_DIR = Path("data/raw")


class EdgarClient:
    def __init__(self, user_agent: str = SEC_USER_AGENT):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": user_agent})
        self._ticker_to_cik = None

    def _load_ticker_map(self) -> dict:
        if self._ticker_to_cik is None:
            resp = self.session.get(TICKER_MAP_URL, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            self._ticker_to_cik = {
                entry["ticker"].upper(): entry["cik_str"] for entry in data.values()
            }
        return self._ticker_to_cik

    def get_cik(self, ticker: str) -> int:
        ticker_map = self._load_ticker_map()
        ticker = ticker.upper()
        if ticker not in ticker_map:
            raise ValueError(f"Unknown ticker: {ticker}")
        return ticker_map[ticker]

    def search_filings(self, ticker: str, form_type: str, limit: int = None) -> list[dict]:
        """Return recent filings of `form_type` for `ticker`, newest first."""
        if form_type not in VALID_FORM_TYPES:
            raise ValueError(f"form_type must be one of {VALID_FORM_TYPES}, got {form_type!r}")

        cik = self.get_cik(ticker)
        url = SUBMISSIONS_URL_TEMPLATE.format(cik=cik)
        resp = self.session.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        recent = data.get("filings", {}).get("recent", {})
        forms = recent.get("form", [])
        accession_numbers = recent.get("accessionNumber", [])
        filing_dates = recent.get("filingDate", [])
        primary_documents = recent.get("primaryDocument", [])

        results = []
        for form, accession_number, filing_date, primary_doc in zip(
            forms, accession_numbers, filing_dates, primary_documents
        ):
            if form != form_type:
                continue
            accession_no_dashes = accession_number.replace("-", "")
            filing_url = f"{ARCHIVES_BASE_URL}/{cik}/{accession_no_dashes}/{primary_doc}"
            results.append(
                {
                    "ticker": ticker.upper(),
                    "form_type": form,
                    "filing_date": filing_date,
                    "accession_number": accession_number,
                    "url": filing_url,
                }
            )
            if limit is not None and len(results) >= limit:
                break

        return results

    def download_filing(
        self,
        filing: dict,
        dest_dir: Path = RAW_DATA_DIR,
        overwrite: bool = False,
    ) -> Path:
        """Download a filing's raw document and return the local path.

        `filing` is a dict as returned by `search_filings`. Files are named
        `{ticker}_{form_type}_{filing_date}{ext}`, e.g. AAPL_10-K_2024-11-01.htm,
        so a filing maps to exactly one path and can be re-used across runs.
        Skips the network call if the file already exists unless `overwrite`.
        """
        dest_dir = Path(dest_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)

        url = filing["url"]
        # Preserve the source extension: primary documents are usually .htm,
        # but older filings can be .txt and XBRL instance docs .xml.
        extension = Path(url).suffix or ".htm"
        filename = f"{filing['ticker']}_{filing['form_type']}_{filing['filing_date']}{extension}"
        path = dest_dir / filename

        if path.exists() and not overwrite:
            return path

        resp = self.session.get(url, timeout=30)
        resp.raise_for_status()
        path.write_bytes(resp.content)
        return path

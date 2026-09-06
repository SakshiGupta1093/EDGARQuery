"""Isolate the major narrative sections of a SEC 10-K/10-Q filing.

Filings are structured as numbered "Item" sections, but the item numbers mean
different things in a 10-K than in a 10-Q, and every filing repeats its item
headings in a table of contents. This module converts filing HTML to plain
text, locates the real (non-TOC) item headings, and returns the sections we
care about for retrieval: MD&A, Risk Factors, and Financial Statements.
"""
import re
import warnings
from pathlib import Path

from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

# Modern filings are inline-XBRL (XHTML). Parsing them with the HTML parser is
# deliberate - it is more forgiving of the malformed markup filers produce, and
# we only want the text - so bs4's "this looks like XML" warning is just noise.
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

# Item numbers differ by form type: Item 7 is MD&A in an annual report but
# Properties in a quarterly one, so the map has to be chosen per filing.
SECTION_ITEMS_10K = {
    "risk_factors": ("1", "A"),
    "mda": ("7", None),
    "financial_statements": ("8", None),
}
SECTION_ITEMS_10Q = {
    "financial_statements": ("1", None),
    "mda": ("2", None),
    "risk_factors": ("1", "A"),
}

# Matches an item heading at the start of a line, e.g. "Item 1A." / "ITEM 7 -".
ITEM_HEADING_RE = re.compile(
    r"^[ \t]*item[ \t]*(\d{1,2})[ \t]*\(?([a-c])?\)?[ \t]*[.:\-–—]?",
    re.IGNORECASE | re.MULTILINE,
)

# A real section body is far longer than a table-of-contents line; anything
# this short is a cross-reference or TOC entry, not the section itself.
MIN_SECTION_CHARS = 500


def html_to_text(html: str | bytes) -> str:
    """Strip filing HTML down to normalized, line-structured plain text."""
    soup = BeautifulSoup(html, "lxml")

    # Inline-XBRL filings carry a hidden metadata block plus scripts/styles
    # that would otherwise pollute the text with tag soup and fact values.
    for tag in soup.find_all(["script", "style", "ix:header"]):
        tag.decompose()

    text = soup.get_text("\n")
    text = text.replace("\xa0", " ").replace("​", "")
    text = re.sub(r"[ \t]+", " ", text)
    # Collapse the runs of blank lines left behind by table/div markup.
    text = re.sub(r"\n[ \t]*\n+", "\n", text)
    return text.strip()


def detect_form_type(text: str) -> str:
    """Infer 10-K vs 10-Q from the filing cover page."""
    head = text[:5000].lower()
    if "quarterly report" in head or "form 10-q" in head:
        return "10-Q"
    return "10-K"


def _find_item_headings(text: str) -> list[tuple[int, int, tuple[str, str | None]]]:
    """Return (start, end, item_key) for every item heading found in `text`."""
    headings = []
    for match in ITEM_HEADING_RE.finditer(text):
        number = match.group(1)
        letter = match.group(2).upper() if match.group(2) else None
        headings.append((match.start(), match.end(), (number, letter)))
    return headings


def _extract_section(text: str, headings: list, item_key: tuple) -> str | None:
    """Return the body of `item_key`, picking the longest candidate span.

    Each item heading appears at least twice - once in the table of contents
    and once at the real section - so the longest span between a heading and
    the next heading of any kind is the actual section body.
    """
    best = None
    for index, (_, heading_end, key) in enumerate(headings):
        if key != item_key:
            continue
        # The section runs until the next heading for a *different* item;
        # a repeated heading is usually a page header on the same section.
        section_end = len(text)
        for next_start, _, next_key in headings[index + 1 :]:
            if next_key != item_key:
                section_end = next_start
                break
        body = text[heading_end:section_end].strip()
        if best is None or len(body) > len(best):
            best = body

    if best is None or len(best) < MIN_SECTION_CHARS:
        return None
    return best


def parse_filing(html: str | bytes, form_type: str | None = None) -> dict[str, str]:
    """Split filing HTML into its major sections.

    Returns a dict keyed by section name (`risk_factors`, `mda`,
    `financial_statements`). Sections that cannot be located are omitted
    rather than returned empty - a 10-Q, for example, often has no risk
    factors section of its own.
    """
    text = html_to_text(html)
    if form_type is None:
        form_type = detect_form_type(text)

    section_items = SECTION_ITEMS_10K if form_type == "10-K" else SECTION_ITEMS_10Q
    headings = _find_item_headings(text)

    sections = {}
    for name, item_key in section_items.items():
        body = _extract_section(text, headings, item_key)
        if body is not None:
            sections[name] = body
    return sections


def parse_filing_file(path: str | Path, form_type: str | None = None) -> dict[str, str]:
    """Parse a filing already downloaded to disk."""
    path = Path(path)
    return parse_filing(path.read_bytes(), form_type=form_type)

"""Section A — Data ingestion and cleaning.

Loads PDF / HTML / TXT financial filings, normalizes text, and extracts
metadata (company, year, filing_type, section, page) so downstream retrieval
can filter and cite precisely.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterable

from bs4 import BeautifulSoup
from pypdf import PdfReader


# ---------- data classes ----------

@dataclass
class Document:
    """A single ingested unit (typically one page or one logical section)."""
    text: str
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


# ---------- file loaders ----------

def load_pdf(path: Path | str) -> list[Document]:
    """Load a PDF page-by-page so we can keep page-level citations."""
    path = Path(path)
    reader = PdfReader(str(path))
    docs: list[Document] = []
    for page_idx, page in enumerate(reader.pages, start=1):
        try:
            raw = page.extract_text() or ""
        except Exception:
            raw = ""
        cleaned = clean_text(raw)
        if not cleaned.strip():
            continue
        docs.append(Document(text=cleaned, metadata={"source": path.name, "page": page_idx}))
    return docs


_XBRL_DECOMPOSE_PREFIXES = ("link:", "xbrldi:", "xbrli:")
_XBRL_UNWRAP_PREFIXES = ("ix:",)
_XBRL_HREF_RE = re.compile(r"https?://[^\s]+(?:fasb\.org|xbrl\.org|sec\.gov)[^\s]*")


def _strip_xbrl(soup: BeautifulSoup) -> None:
    """Modern 10-Ks are iXBRL: HTML with inline XBRL tags. Strip the plumbing,
    keep the prose. We decompose pure-XBRL elements (link/xbrldi/xbrli) and
    unwrap <ix:*> wrappers so their text content survives."""
    for tag in soup.find_all(True):
        name = (tag.name or "").lower()
        if any(name.startswith(p) for p in _XBRL_DECOMPOSE_PREFIXES):
            tag.decompose()
        elif any(name.startswith(p) for p in _XBRL_UNWRAP_PREFIXES):
            tag.unwrap()


def load_html(path: Path | str) -> list[Document]:
    path = Path(path)
    raw = path.read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(raw, "lxml")
    for tag in soup(["script", "style", "noscript", "head"]):
        tag.decompose()
    _strip_xbrl(soup)
    text = soup.get_text(separator="\n")
    text = _XBRL_HREF_RE.sub(" ", text)  # drop XBRL namespace URIs that survive in text
    cleaned = clean_text(text)
    if not cleaned.strip():
        return []
    return [Document(text=cleaned, metadata={"source": path.name, "page": 1})]


def load_txt(path: Path | str) -> list[Document]:
    path = Path(path)
    raw = path.read_text(encoding="utf-8", errors="ignore")
    cleaned = clean_text(raw)
    if not cleaned.strip():
        return []
    return [Document(text=cleaned, metadata={"source": path.name, "page": 1})]


LOADERS = {".pdf": load_pdf, ".html": load_html, ".htm": load_html, ".txt": load_txt}


def load_file(path: Path | str) -> list[Document]:
    path = Path(path)
    loader = LOADERS.get(path.suffix.lower())
    if loader is None:
        return []
    return loader(path)


def load_directory(root: Path | str) -> list[Document]:
    """Walk a directory and ingest every supported file."""
    root = Path(root)
    docs: list[Document] = []
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in LOADERS:
            file_docs = load_file(p)
            file_meta = extract_file_metadata(p)
            section_aware: list[Document] = []
            for d in file_docs:
                d.metadata.update(file_meta)
                d.metadata["section"] = guess_section(d.text)
                section_aware.append(d)
            docs.extend(section_aware)
    return docs


# ---------- cleaning ----------

_WHITESPACE_RE = re.compile(r"[ \t]+")
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")
_PAGE_NUMBER_LINE_RE = re.compile(r"^\s*(page\s+)?\d{1,4}\s*$", re.IGNORECASE)
_FORM_FOOTER_RE = re.compile(r"^(table of contents|index)\s*$", re.IGNORECASE)
# iXBRL leftovers and other low-signal lines:
#   - lines that are mostly digits/punctuation (XBRL value blobs)
#   - lines that are URI-like or pure namespace prefixes
_NOISE_LINE_RE = re.compile(
    r"^("
    r"[\d\s.,$()%-]+"           # pure number/punct lines
    r"|P\d+[YMD]+(\s+P\d+[YMD]+)*"  # ISO 8601 durations like P1Y P1Y
    r"|(true|false|FY|Q[1-4])(\s+(true|false|FY|Q[1-4]|0))*"  # boolean / period markers
    r")$",
    re.IGNORECASE,
)


def _is_low_signal(line: str) -> bool:
    if _NOISE_LINE_RE.match(line):
        return True
    # ratio of alphabetic chars; iXBRL leftovers are dominated by digits/punct
    if len(line) > 30:
        alpha = sum(1 for ch in line if ch.isalpha())
        if alpha / len(line) < 0.15:
            return True
    return False


def clean_text(text: str) -> str:
    """Normalize whitespace, strip page-number-only lines and obvious noise."""
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = []
    for line in text.split("\n"):
        s = line.strip()
        if not s:
            lines.append("")
            continue
        if _PAGE_NUMBER_LINE_RE.match(s):
            continue
        if _FORM_FOOTER_RE.match(s):
            continue
        if _is_low_signal(s):
            continue
        s = _WHITESPACE_RE.sub(" ", s)
        lines.append(s)
    out = "\n".join(lines)
    out = _MULTI_NEWLINE_RE.sub("\n\n", out)
    return out.strip()


# ---------- metadata extraction ----------

# Common ticker → company map for nicer citations; extend as needed.
_TICKER_TO_COMPANY = {
    "AAPL": "Apple",
    "MSFT": "Microsoft",
    "GOOGL": "Alphabet",
    "GOOG": "Alphabet",
    "AMZN": "Amazon",
    "META": "Meta",
    "NVDA": "NVIDIA",
    "TSLA": "Tesla",
    "JPM": "JPMorgan Chase",
    "BAC": "Bank of America",
    "XOM": "Exxon Mobil",
    "WMT": "Walmart",
}

_FILING_TYPE_RE = re.compile(r"\b(10[- ]?K|10[- ]?Q|8[- ]?K|20[- ]?F)\b", re.IGNORECASE)
_YEAR_RE = re.compile(r"\b(20\d{2})\b")
_FILENAME_TICKER_RE = re.compile(r"\b([A-Z]{1,5})\b")


def extract_file_metadata(path: Path) -> dict:
    """Best-effort metadata inference from filename + filepath.

    Note: \\b in regex doesn't break on underscores (they're word chars), so
    we normalize underscores to spaces before pattern matching.
    """
    name = path.stem
    norm = name.replace("_", " ")
    meta: dict = {"source": path.name, "filepath": str(path)}

    m = _FILING_TYPE_RE.search(norm)
    if m:
        meta["filing_type"] = m.group(1).upper().replace(" ", "-").replace("--", "-")

    m = _YEAR_RE.search(norm)
    if m:
        meta["year"] = int(m.group(1))

    for c in _FILENAME_TICKER_RE.findall(norm):
        if c in _TICKER_TO_COMPANY:
            meta["ticker"] = c
            meta["company"] = _TICKER_TO_COMPANY[c]
            break
    if "company" not in meta:
        parent = path.parent.name
        meta["company"] = parent if parent and parent.lower() not in {"raw", "data"} else name
    return meta


# Section heading patterns common to 10-K / 10-Q filings
_SECTION_PATTERNS = [
    (re.compile(r"\bitem\s*1a[\.\s].*?risk factors", re.IGNORECASE), "Item 1A - Risk Factors"),
    (re.compile(r"\bitem\s*1[\.\s].*?business", re.IGNORECASE), "Item 1 - Business"),
    (re.compile(r"\bitem\s*2[\.\s].*?properties", re.IGNORECASE), "Item 2 - Properties"),
    (re.compile(r"\bitem\s*3[\.\s].*?legal proceedings", re.IGNORECASE), "Item 3 - Legal Proceedings"),
    (re.compile(r"\bitem\s*7a[\.\s]", re.IGNORECASE), "Item 7A - Quantitative & Qualitative Disclosures"),
    (re.compile(r"\bitem\s*7[\.\s].*?management.{0,40}discussion", re.IGNORECASE), "Item 7 - MD&A"),
    (re.compile(r"\bmanagement.{0,5}s\s+discussion\s+and\s+analysis", re.IGNORECASE), "Item 7 - MD&A"),
    (re.compile(r"\bitem\s*8[\.\s].*?financial statements", re.IGNORECASE), "Item 8 - Financial Statements"),
    (re.compile(r"\bforward[- ]looking\s+statements", re.IGNORECASE), "Forward-Looking Statements"),
]


def guess_section(text: str) -> str:
    """Look at the first ~600 chars to classify which 10-K section a chunk belongs to."""
    head = text[:600]
    for pat, label in _SECTION_PATTERNS:
        if pat.search(head):
            return label
    return "Other"


# ---------- CLI ----------

if __name__ == "__main__":
    import argparse, json, sys
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="File or directory to ingest")
    parser.add_argument("--limit", type=int, default=3, help="Number of docs to print")
    args = parser.parse_args()

    p = Path(args.path)
    docs = load_directory(p) if p.is_dir() else load_file(p)
    print(f"Loaded {len(docs)} document units", file=sys.stderr)
    for d in docs[: args.limit]:
        snippet = d.text[:300].replace("\n", " ")
        print(json.dumps({"metadata": d.metadata, "preview": snippet}, indent=2))

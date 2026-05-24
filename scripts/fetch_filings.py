"""Fetch 10-K filings directly from SEC EDGAR into data/raw/.

SEC EDGAR's fair-access policy requires a User-Agent header with a real
contact email. We pass that to sec-edgar-downloader. The downloader saves
to a nested folder structure (sec-edgar-filings/<TICKER>/10-K/<accession>/)
and we then flatten + rename into data/raw/<TICKER>_10-K_<YEAR>.html so the
ingestor's filename-based metadata sniffing picks up company + year.
"""
from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import settings


DEFAULT_TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]


def _find_primary_doc(filing_dir: Path) -> Path | None:
    """sec-edgar-downloader v5 saves a single full-submission.txt per filing."""
    txts = list(filing_dir.rglob("*.txt"))
    if txts:
        return max(txts, key=lambda p: p.stat().st_size)
    htms = list(filing_dir.rglob("*.htm")) + list(filing_dir.rglob("*.html"))
    if htms:
        return max(htms, key=lambda p: p.stat().st_size)
    return None


_DOC_BLOCK_RE = re.compile(
    r"<DOCUMENT>\s*<TYPE>([^\n<]+).*?<TEXT>(.*?)</TEXT>\s*</DOCUMENT>",
    re.IGNORECASE | re.DOTALL,
)


def extract_10k_html(submission_path: Path) -> str | None:
    """Extract the inner HTML/text of the TYPE=10-K DOCUMENT block from a SEC submission file."""
    raw = submission_path.read_text(encoding="utf-8", errors="ignore")
    for m in _DOC_BLOCK_RE.finditer(raw):
        doc_type = m.group(1).strip().upper()
        if doc_type == "10-K":
            return m.group(2).strip()
    return None


_ACCESSION_YEAR_RE = re.compile(r"-(\d{2})-")


def _accession_year(accession: str) -> int | None:
    """Parse year from an accession number like 0000320193-23-000106."""
    m = _ACCESSION_YEAR_RE.search(accession)
    if not m:
        return None
    yy = int(m.group(1))
    return 2000 + yy if yy < 80 else 1900 + yy


def fetch(tickers: list[str], limit: int, company: str, email: str) -> list[Path]:
    from sec_edgar_downloader import Downloader

    raw_root = settings.DATA_RAW
    raw_root.mkdir(parents=True, exist_ok=True)
    download_root = raw_root / "_edgar_download"
    download_root.mkdir(exist_ok=True)

    dl = Downloader(company, email, str(download_root))
    written: list[Path] = []

    for ticker in tickers:
        print(f"\n=== {ticker} ===")
        try:
            n = dl.get("10-K", ticker, limit=limit)
            print(f"  downloaded {n} 10-K filing(s)")
        except Exception as e:
            print(f"  ERROR: {e}")
            continue

        ticker_dir = download_root / "sec-edgar-filings" / ticker / "10-K"
        if not ticker_dir.exists():
            print(f"  no filings folder at {ticker_dir}")
            continue

        for filing_dir in sorted(ticker_dir.iterdir()):
            if not filing_dir.is_dir():
                continue
            primary = _find_primary_doc(filing_dir)
            if primary is None:
                continue
            year = _accession_year(filing_dir.name) or 0
            out_name = f"{ticker}_10-K_{year}.html" if year else f"{ticker}_10-K_{filing_dir.name}.html"
            out_path = raw_root / out_name

            extracted = extract_10k_html(primary) if primary.suffix.lower() == ".txt" else None
            if extracted:
                out_path.write_text(extracted, encoding="utf-8")
                print(f"  wrote {out_path.name}  ({len(extracted) // 1024} KB, 10-K body only)")
            else:
                shutil.copyfile(primary, out_path)
                print(f"  wrote {out_path.name}  ({primary.stat().st_size // 1024} KB, full submission)")
            written.append(out_path)

    # Clean up the nested download tree
    shutil.rmtree(download_root, ignore_errors=True)
    return written


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS)
    parser.add_argument("--limit", type=int, default=2,
                        help="Number of recent 10-K filings to fetch per ticker")
    parser.add_argument("--company", default="Capstone Research",
                        help="User-Agent company name (SEC fair-access policy)")
    parser.add_argument("--email", default="capstone@example.com",
                        help="User-Agent email (SEC fair-access policy — use a real address for production)")
    args = parser.parse_args()

    written = fetch(args.tickers, args.limit, args.company, args.email)
    print(f"\nDone. {len(written)} filing(s) saved to {settings.DATA_RAW}")
    if written:
        print("Next: python scripts/build_index.py --rebuild")


if __name__ == "__main__":
    main()

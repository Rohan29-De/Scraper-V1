#!/usr/bin/env python3
"""
main.py
-------
CLI entry-point for the Scraper-V1.

Usage:
    python main.py --url "https://www.trustpilot.com..." [OPTIONS]

Run `python main.py --help` for full options.
"""

import argparse
import logging
import sys
from pathlib import Path

# Load .env file if present (sets GROQ_API_KEY without shell export)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv optional; user can export manually

# Ensure src/ is on the path when running from project root
sys.path.insert(0, str(Path(__file__).parent / "src"))

from scraper import scrape_reviews
from preprocessor import preprocess_all
from llm_client import analyse_all, DEFAULT_MODEL
from storage import save_results


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    fmt = "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s"
    logging.basicConfig(level=level, format=fmt, datefmt="%H:%M:%S")
    # Quieten noisy third-party loggers
    for lib in ("urllib3", "httpx", "httpcore", "openai", "charset_normalizer"):
        logging.getLogger(lib).setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scrape Trustpilot reviews and analyse them with Groq LLM.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--url",
        required=True,
        help=(
            "Trustpilot URL or bare company domain to scrape. "
            "e.g. 'https://www.trustpilot.com/review/www.netflix.com' or 'netflix.com'"
        ),
    )
    parser.add_argument(
        "--pages",
        type=int,
        default=3,
        metavar="N",
        help="Maximum number of review pages to scrape (each page ~10 reviews).",
    )
    parser.add_argument(
        "--max-reviews",
        type=int,
        metavar="N",
        help="Maximum total reviews to collect (unlimited if not specified).",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Groq model to use (e.g. llama3-8b-8192, mixtral-8x7b-32768).",
    )
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Directory to write CSV and JSON output files.",
    )
    parser.add_argument(
        "--max-chunk-tokens",
        type=int,
        default=400,
        help="Maximum tokens per review chunk sent to the LLM.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.5,
        help="Seconds to wait between LLM API calls (rate-limit buffer).",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug-level logging.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_pipeline(args: argparse.Namespace) -> None:
    log = logging.getLogger("main")

    # ── Step 1: Scrape ──────────────────────────────────────────────────────
    log.info("═" * 60)
    log.info("STEP 1/4  Scraping reviews from Trustpilot…")
    log.info("URL    : %s", args.url)
    log.info("Pages  : %d", args.pages)
    if args.max_reviews:
        log.info("Max Reviews: %d", args.max_reviews)
    else:
        log.info("Max Reviews: Unlimited")
    log.info("═" * 60)

    reviews = scrape_reviews(args.url, max_pages=args.pages, max_reviews=args.max_reviews)

    if not reviews:
        log.error("No reviews were scraped. Check the URL and try again.")
        sys.exit(1)

    log.info("✓ Scraped %d reviews.", len(reviews))

    # ── Step 2: Preprocess ──────────────────────────────────────────────────
    log.info("═" * 60)
    log.info("STEP 2/4  Preprocessing review text…")
    log.info("═" * 60)

    processed = preprocess_all(reviews, max_chunk_tokens=args.max_chunk_tokens)
    log.info("✓ Preprocessed %d reviews.", len(processed))

    # Show token distribution
    tokens = [p.approx_tokens for p in processed]
    log.info("  Token stats — min: %d  max: %d  avg: %.0f",
             min(tokens), max(tokens), sum(tokens) / len(tokens))
    chunked_count = sum(1 for p in processed if len(p.chunks) > 1)
    if chunked_count:
        log.info("  %d review(s) were split into multiple chunks.", chunked_count)

    # ── Step 3: LLM analysis ────────────────────────────────────────────────
    log.info("═" * 60)
    log.info("STEP 3/4  Sending reviews to Groq (%s)…", args.model)
    log.info("═" * 60)

    llm_results = analyse_all(processed, model=args.model, inter_request_delay=args.delay)

    # Summary stats
    sentiments = [r.sentiment for r in llm_results if not r.error]
    from collections import Counter
    dist = Counter(sentiments)
    log.info("✓ LLM analysis complete.")
    log.info("  Sentiment distribution: %s", dict(dist))

    errors = sum(1 for r in llm_results if r.error)
    if errors:
        log.warning("  %d review(s) failed LLM analysis.", errors)

    # ── Step 4: Save ────────────────────────────────────────────────────────
    log.info("═" * 60)
    log.info("STEP 4/4  Saving results to %s…", args.output_dir)
    log.info("═" * 60)

    paths = save_results(processed, llm_results, output_dir=args.output_dir)

    log.info("✓ Results saved:")
    for fmt, path in paths.items():
        log.info("  [%s] %s", fmt.upper(), path)

    log.info("═" * 60)
    log.info("Pipeline complete! Processed %d reviews.", len(processed))
    log.info("═" * 60)


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = _parse_args()
    _setup_logging(args.verbose)
    run_pipeline(args)
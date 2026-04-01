"""
storage.py
----------
Serialises scraped reviews + LLM results to CSV and JSON.

Output schema (one record per review):
  - author, rating, date, verified_purchase
  - title, review_text (cleaned)
  - sentiment, sentiment_score, summary, key_points
  - url, model_used, error
"""

import json
import logging
from datetime import datetime, UTC
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_records(processed_reviews: list, llm_results: list) -> list[dict]:
    """Zip processed reviews with LLM results into flat dictionaries."""
    records = []
    for review, result in zip(processed_reviews, llm_results):
        records.append({
            # Review metadata
            "author": review.author,
            "location": getattr(review, "location", ""),
            "rating": review.rating,
            "date": review.date,
            "verified_purchase": review.verified_purchase,
            "helpful_votes": review.helpful_votes,
            # Review content
            "title": review.title,
            "review_text": review.cleaned_text,
            "approx_tokens": review.approx_tokens,
            # LLM outputs
            "sentiment": result.sentiment,
            "sentiment_score": result.sentiment_score,
            "summary": result.summary,
            "key_points": result.key_points,       # list – serialised differently per format
            "model_used": result.model_used,
            "llm_error": result.error or "",
            # Source
            "source_url": review.url,
            "scraped_at": datetime.now(UTC).isoformat() + "Z",
        })
    return records


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def save_results(
    processed_reviews: list,
    llm_results: list,
    output_dir: str = "output",
    filename_stem: str = "reviews",
) -> dict[str, Path]:
    """
    Save results to both CSV and JSON in `output_dir`.

    Args:
        processed_reviews: List[ProcessedReview] from preprocessor.
        llm_results:       List[LLMResult] from llm_client (same order).
        output_dir:        Directory to write files into (created if missing).
        filename_stem:     Base filename without extension.

    Returns:
        Dict with keys "csv" and "json" mapping to absolute Path objects.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    records = _build_records(processed_reviews, llm_results)

    if not records:
        logger.warning("No records to save.")
        return {}

    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    stem = f"{filename_stem}_{timestamp}"

    # ---- CSV ----------------------------------------------------------------
    # key_points list → semicolon-separated string for flat CSV
    csv_records = []
    for r in records:
        flat = dict(r)
        flat["key_points"] = "; ".join(r["key_points"]) if r["key_points"] else ""
        csv_records.append(flat)

    df = pd.DataFrame(csv_records)
    csv_path = out / f"{stem}.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")   # utf-8-sig for Excel compat
    logger.info("CSV saved → %s  (%d rows)", csv_path, len(df))

    # ---- JSON ---------------------------------------------------------------
    json_payload = {
        "metadata": {
            "total_reviews": len(records),
            "scraped_at": datetime.now(UTC).isoformat() + "Z",
            "model": records[0]["model_used"] if records else "unknown",
        },
        "reviews": records,
    }
    json_path = out / f"{stem}.json"
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(json_payload, fh, ensure_ascii=False, indent=2)
    logger.info("JSON saved → %s", json_path)

    return {"csv": csv_path, "json": json_path}


def load_csv(path: str) -> pd.DataFrame:
    """Convenience loader for the output CSV."""
    return pd.read_csv(path, encoding="utf-8-sig")
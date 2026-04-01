"""
tests/test_storage.py
---------------------
Unit tests for the storage (CSV + JSON output) module.
Run with:  pytest tests/ -v
"""

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from preprocessor import ProcessedReview
from llm_client import LLMResult
from storage import save_results, load_csv


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_processed(n: int = 3) -> list[ProcessedReview]:
    reviews = []
    for i in range(n):
        reviews.append(ProcessedReview(
            author=f"User {i}",
            rating=float(i % 5 + 1),
            title=f"Title {i}",
            cleaned_title=f"Title {i}",
            cleaned_text=f"Review text for product {i}. It was {'great' if i % 2 == 0 else 'bad'}.",
            chunks=[f"Review text for product {i}."],
            date=f"2024-01-{i+1:02d}",
            verified_purchase=i % 2 == 0,
            helpful_votes=i * 2,
            url=f"https://flipkart.com/review/{i}",
            approx_tokens=50,
        ))
    return reviews


def _make_results(n: int = 3) -> list[LLMResult]:
    sentiments = ["Positive", "Negative", "Neutral"]
    results = []
    for i in range(n):
        results.append(LLMResult(
            sentiment=sentiments[i % 3],
            sentiment_score=float(i % 5 + 1),
            summary=f"Summary for review {i}.",
            key_points=[f"Point A{i}", f"Point B{i}"],
            model_used="llama3-8b-8192",
        ))
    return results


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSaveResults:
    def test_creates_output_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = save_results(_make_processed(), _make_results(), output_dir=tmpdir)
            assert "csv" in paths
            assert "json" in paths
            assert paths["csv"].exists()
            assert paths["json"].exists()

    def test_csv_row_count(self):
        n = 5
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = save_results(_make_processed(n), _make_results(n), output_dir=tmpdir)
            df = load_csv(str(paths["csv"]))
            assert len(df) == n

    def test_csv_required_columns(self):
        required = {
            "author", "rating", "sentiment", "sentiment_score",
            "summary", "review_text", "date", "source_url",
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = save_results(_make_processed(), _make_results(), output_dir=tmpdir)
            df = load_csv(str(paths["csv"]))
            assert required.issubset(set(df.columns))

    def test_json_structure(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = save_results(_make_processed(2), _make_results(2), output_dir=tmpdir)
            with open(paths["json"]) as f:
                data = json.load(f)
            assert "metadata" in data
            assert "reviews" in data
            assert data["metadata"]["total_reviews"] == 2
            assert len(data["reviews"]) == 2

    def test_json_key_points_is_list(self):
        """key_points should remain a list in JSON (not flattened to string)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = save_results(_make_processed(1), _make_results(1), output_dir=tmpdir)
            with open(paths["json"]) as f:
                data = json.load(f)
            assert isinstance(data["reviews"][0]["key_points"], list)

    def test_csv_key_points_is_string(self):
        """key_points should be semicolon-joined string in CSV."""
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = save_results(_make_processed(1), _make_results(1), output_dir=tmpdir)
            df = load_csv(str(paths["csv"]))
            val = df["key_points"].iloc[0]
            assert isinstance(val, str)
            assert ";" in val

    def test_empty_reviews_returns_empty(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = save_results([], [], output_dir=tmpdir)
            assert result == {}

    def test_creates_output_dir_if_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            new_dir = Path(tmpdir) / "new" / "nested" / "dir"
            paths = save_results(_make_processed(1), _make_results(1), output_dir=str(new_dir))
            assert new_dir.exists()
            assert paths["csv"].exists()
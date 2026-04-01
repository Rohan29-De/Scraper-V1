"""
tests/test_preprocessor.py
--------------------------
Unit tests for the preprocessing pipeline.
Run with:  pytest tests/ -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from preprocessor import clean_text, chunk_text, preprocess_review, _approx_token_count
from scraper import Review


# ---------------------------------------------------------------------------
# clean_text
# ---------------------------------------------------------------------------

class TestCleanText:
    def test_html_entities(self):
        assert "&amp;" not in clean_text("Good &amp; bad")
        assert "Good & bad" == clean_text("Good &amp; bad")

    def test_strip_html_tags(self):
        result = clean_text("<b>Great</b> product <br/>highly recommended")
        assert "<b>" not in result
        assert "Great" in result

    def test_unicode_normalization(self):
        # NFKC: fullwidth A (U+FF21) → A
        result = clean_text("\uff21 great product")
        assert "A great product" == result

    def test_collapse_whitespace(self):
        result = clean_text("good   product\t\there")
        assert "  " not in result
        assert "\t" not in result

    def test_collapse_repeated_punctuation(self):
        result = clean_text("Amazing!!!!!!")
        assert "!!!!!!" not in result
        assert "!!!" in result

    def test_empty_string(self):
        assert clean_text("") == ""

    def test_control_characters_removed(self):
        result = clean_text("hello\x00world\x07")
        assert "\x00" not in result
        assert "\x07" not in result


# ---------------------------------------------------------------------------
# chunk_text
# ---------------------------------------------------------------------------

class TestChunkText:
    def test_short_text_no_chunking(self):
        text = "Great product. I love it."
        chunks = chunk_text(text, max_tokens=400)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_long_text_is_chunked(self):
        # ~500 tokens worth of text (4 chars/token → 2000 chars)
        long_text = "This is a sentence with some words. " * 60  # ~2160 chars
        chunks = chunk_text(long_text, max_tokens=100)
        assert len(chunks) > 1

    def test_chunks_are_non_empty(self):
        long_text = "Word " * 500
        chunks = chunk_text(long_text, max_tokens=50)
        for c in chunks:
            assert c.strip() != ""

    def test_total_content_preserved(self):
        """All words should appear somewhere across chunks."""
        text = " ".join(f"word{i}" for i in range(200))
        chunks = chunk_text(text, max_tokens=50)
        reconstructed = " ".join(chunks)
        for i in range(200):
            assert f"word{i}" in reconstructed


# ---------------------------------------------------------------------------
# preprocess_review
# ---------------------------------------------------------------------------

class TestPreprocessReview:
    def _make_review(self, text="", title="", rating=4.0):
        return Review(
            author="Test User",
            rating=rating,
            title=title,
            text=text,
            date="15 Jan, 2024",
            verified_purchase=True,
            helpful_votes=5,
            url="https://flipkart.com/test",
        )

    def test_basic_preprocessing(self):
        review = self._make_review(text="Good product!!", title="Nice")
        result = preprocess_review(review)
        assert result.cleaned_text == "Good product!!"
        assert result.cleaned_title == "Nice"
        assert len(result.chunks) >= 1

    def test_html_entities_in_review(self):
        review = self._make_review(text="Good &amp; durable product")
        result = preprocess_review(review)
        assert "&amp;" not in result.cleaned_text
        assert "Good & durable product" == result.cleaned_text

    def test_metadata_preserved(self):
        review = self._make_review(text="Nice", rating=3.5)
        result = preprocess_review(review)
        assert result.rating == 3.5
        assert result.author == "Test User"
        assert result.verified_purchase is True

    def test_approx_tokens_positive(self):
        review = self._make_review(text="Some review text here.")
        result = preprocess_review(review)
        assert result.approx_tokens > 0

    def test_empty_text(self):
        review = self._make_review(text="", title="")
        result = preprocess_review(review)
        assert result.chunks == [""]  # graceful handling


# ---------------------------------------------------------------------------
# Token counter
# ---------------------------------------------------------------------------

class TestApproxTokenCount:
    def test_empty(self):
        assert _approx_token_count("") >= 1

    def test_scales_with_length(self):
        short = _approx_token_count("hi")
        long = _approx_token_count("hi " * 100)
        assert long > short
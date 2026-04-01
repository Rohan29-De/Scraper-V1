"""
preprocessor.py
---------------
Cleans and prepares review text before sending to the LLM.

Steps applied to each review:
  1. Decode / fix encoding artefacts (mojibake, HTML entities)
  2. Strip HTML tags that sneak through BeautifulSoup
  3. Normalize Unicode (NFKC) and whitespace
  4. Remove zero-width and control characters
  5. Collapse repeated punctuation
  6. Token-count-aware chunking for very long reviews
"""

import re
import unicodedata
import html as html_lib
import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Token counting (lightweight – avoids full tiktoken dependency)
# ---------------------------------------------------------------------------

def _approx_token_count(text: str) -> int:
    """
    Rough token estimate: ~4 characters per token (GPT-family heuristic).
    Good enough for chunk gating; swap for tiktoken if you need precision.
    """
    return max(1, len(text) // 4)


# ---------------------------------------------------------------------------
# Cleaning pipeline
# ---------------------------------------------------------------------------

# Strip residual HTML tags
_HTML_TAG_RE = re.compile(r"<[^>]+>")
# Collapse runs of whitespace (spaces, tabs, non-breaking spaces) to single space
_WHITESPACE_RE = re.compile(r"[ \t\u00a0\u200b]+")
# Remove control characters (except newline / carriage return)
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
# Collapse 3+ repeated punctuation (e.g. "!!!!!!" → "!!!")
_REPEAT_PUNCT_RE = re.compile(r"([!?.]){3,}")
# Collapse 3+ newlines into two
_MULTI_NL_RE = re.compile(r"\n{3,}")


def clean_text(raw: str) -> str:
    """Apply the full cleaning pipeline to a single string."""
    if not raw:
        return ""

    # 1. Decode HTML entities (&amp; &nbsp; etc.)
    text = html_lib.unescape(raw)

    # 2. Strip residual HTML tags
    text = _HTML_TAG_RE.sub(" ", text)

    # 3. Unicode normalisation (NFKC folds ligatures, fullwidth chars, etc.)
    text = unicodedata.normalize("NFKC", text)

    # 4. Remove control characters
    text = _CONTROL_RE.sub("", text)

    # 5. Normalise whitespace
    text = _WHITESPACE_RE.sub(" ", text)

    # 6. Collapse repeated punctuation
    text = _REPEAT_PUNCT_RE.sub(r"\1\1\1", text)

    # 7. Collapse excess newlines
    text = _MULTI_NL_RE.sub("\n\n", text)

    return text.strip()


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def chunk_text(text: str, max_tokens: int = 400, overlap_tokens: int = 50) -> list[str]:
    """
    Split text into chunks that fit within `max_tokens`.
    Uses sentence-aware splitting where possible.

    Args:
        text:           Cleaned review text.
        max_tokens:     Maximum tokens per chunk.
        overlap_tokens: Token overlap between consecutive chunks (context preservation).

    Returns:
        List of text chunks. If the text fits in one chunk, returns [text].
    """
    if _approx_token_count(text) <= max_tokens:
        return [text]

    # Split on sentence boundaries first
    sentences = re.split(r"(?<=[.!?])\s+", text)

    chunks: list[str] = []
    current_sentences: list[str] = []
    current_tokens = 0
    overlap_buffer: list[str] = []

    for sentence in sentences:
        sentence_tokens = _approx_token_count(sentence)

        # Single sentence exceeds limit – hard split by words
        if sentence_tokens > max_tokens:
            words = sentence.split()
            word_chunk: list[str] = []
            word_tokens = 0
            for word in words:
                wt = _approx_token_count(word + " ")
                if word_tokens + wt > max_tokens:
                    if word_chunk:
                        chunks.append(" ".join(word_chunk))
                    word_chunk = [word]
                    word_tokens = wt
                else:
                    word_chunk.append(word)
                    word_tokens += wt
            if word_chunk:
                current_sentences = word_chunk
                current_tokens = word_tokens
            continue

        if current_tokens + sentence_tokens > max_tokens:
            if current_sentences:
                chunks.append(" ".join(current_sentences))
            # Start new chunk with overlap
            current_sentences = list(overlap_buffer) + [sentence]
            current_tokens = sum(_approx_token_count(s) for s in current_sentences)
            overlap_buffer = []
        else:
            current_sentences.append(sentence)
            current_tokens += sentence_tokens

        # Maintain overlap buffer (last ~overlap_tokens worth of sentences)
        overlap_buffer.append(sentence)
        while sum(_approx_token_count(s) for s in overlap_buffer) > overlap_tokens:
            overlap_buffer.pop(0)

    if current_sentences:
        chunks.append(" ".join(current_sentences))

    logger.debug("Chunked text into %d pieces (max_tokens=%d)", len(chunks), max_tokens)
    return chunks


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@dataclass
class ProcessedReview:
    """Cleaned + chunked version of a scraped review."""
    author: str
    rating: Optional[float]
    title: str
    cleaned_title: str
    cleaned_text: str
    chunks: list[str]
    date: str
    verified_purchase: bool
    helpful_votes: int
    url: str
    approx_tokens: int


def preprocess_review(review, max_chunk_tokens: int = 400) -> ProcessedReview:
    """
    Clean and chunk a single Review object.

    Args:
        review:           A Review dataclass instance from scraper.py.
        max_chunk_tokens: Max tokens per chunk (default 400 → safe for Groq's context).

    Returns:
        ProcessedReview ready for LLM consumption.
    """
    cleaned_title = clean_text(review.title)
    cleaned_text = clean_text(review.text)

    # Combine title + text for chunking; title acts as context anchor
    combined = f"{cleaned_title}. {cleaned_text}".strip(". ") if cleaned_title else cleaned_text
    chunks = chunk_text(combined, max_tokens=max_chunk_tokens)

    return ProcessedReview(
        author=clean_text(review.author),
        rating=review.rating,
        title=review.title,
        cleaned_title=cleaned_title,
        cleaned_text=cleaned_text,
        chunks=chunks,
        date=review.date,
        verified_purchase=review.verified_purchase,
        helpful_votes=review.helpful_votes,
        url=review.url,
        approx_tokens=_approx_token_count(combined),
    )


def preprocess_all(reviews: list, max_chunk_tokens: int = 400) -> list[ProcessedReview]:
    """Preprocess a list of Review objects."""
    processed = []
    for i, review in enumerate(reviews):
        try:
            processed.append(preprocess_review(review, max_chunk_tokens))
        except Exception as exc:
            logger.warning("Failed to preprocess review %d: %s", i, exc)
    logger.info("Preprocessed %d / %d reviews.", len(processed), len(reviews))
    return processed
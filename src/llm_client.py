"""
llm_client.py
-------------
Sends preprocessed review chunks to Groq's OpenAI-compatible API
and returns a structured sentiment + summary for each review.

Features:
  - Exponential back-off on rate-limit (429) and server errors (5xx)
  - Per-review token budget awareness
  - Multi-chunk aggregation: if a review is chunked, each chunk is
    summarised and the summaries are merged in a second LLM call
  - Configurable model (default: llama3-8b-8192 on Groq – free tier)
"""

import os
import time
import logging
import random
from dataclasses import dataclass
from typing import Optional

from openai import OpenAI, RateLimitError, APIStatusError, APIConnectionError, APITimeoutError

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_MODEL = "moonshotai/kimi-k2-instruct-0905"          # fast, free-tier Groq model
MAX_RETRIES = 5
BASE_BACKOFF = 2.0                         # seconds
MAX_BACKOFF = 60.0


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class LLMResult:
    sentiment: str          # "Positive" | "Negative" | "Neutral" | "Mixed"
    sentiment_score: float  # 1.0 (very negative) → 5.0 (very positive)
    summary: str            # 1–3 sentence concise summary
    key_points: list[str]   # bullet-point pros/cons extracted
    model_used: str
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

SINGLE_CHUNK_PROMPT = """\
You are a product review analyst. Analyse the following customer review and respond \
with a JSON object — no markdown, no extra text — with EXACTLY these fields:

{{
  "sentiment": "<Positive|Negative|Neutral|Mixed>",
  "sentiment_score": <float 1.0–5.0, where 1=very negative, 5=very positive>,
  "summary": "<1–3 sentence summary of the review>",
  "key_points": ["<point 1>", "<point 2>", ...]
}}

Review:
\"\"\"
{review_text}
\"\"\"
"""

MERGE_CHUNKS_PROMPT = """\
You are a product review analyst. Below are partial summaries of different segments \
of the same customer review. Merge them into a single coherent analysis and respond \
with a JSON object — no markdown, no extra text — with EXACTLY these fields:

{{
  "sentiment": "<Positive|Negative|Neutral|Mixed>",
  "sentiment_score": <float 1.0–5.0, where 1=very negative, 5=very positive>,
  "summary": "<1–3 sentence consolidated summary>",
  "key_points": ["<point 1>", "<point 2>", ...]
}}

Partial summaries:
{partial_summaries}
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_client() -> OpenAI:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GROQ_API_KEY environment variable is not set. "
            "Get your free key at https://console.groq.com/keys"
        )
    return OpenAI(api_key=api_key, base_url=GROQ_BASE_URL)


def _call_with_retry(client: OpenAI, model: str, prompt: str) -> str:
    """Call the Groq API with exponential back-off on transient errors."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,      # low temp → consistent, factual output
                max_tokens=512,
                response_format={"type": "json_object"},
            )
            return response.choices[0].message.content or ""

        except RateLimitError as e:
            wait = min(BASE_BACKOFF * (2 ** attempt) + random.uniform(0, 1), MAX_BACKOFF)
            logger.warning("Rate limit hit (attempt %d/%d). Waiting %.1fs…", attempt, MAX_RETRIES, wait)
            if attempt == MAX_RETRIES:
                raise
            time.sleep(wait)

        except APIStatusError as e:
            if e.status_code >= 500:
                wait = min(BASE_BACKOFF * (2 ** attempt), MAX_BACKOFF)
                logger.warning("Server error %d (attempt %d/%d). Waiting %.1fs…",
                               e.status_code, attempt, MAX_RETRIES, wait)
                if attempt == MAX_RETRIES:
                    raise
                time.sleep(wait)
            else:
                raise   # 4xx (other than 429) are not retryable

        except (APIConnectionError, APITimeoutError) as e:
            wait = min(BASE_BACKOFF * (2 ** attempt), MAX_BACKOFF)
            logger.warning("Network error (attempt %d/%d): %s. Waiting %.1fs…",
                           attempt, MAX_RETRIES, e, wait)
            if attempt == MAX_RETRIES:
                raise
            time.sleep(wait)

    return ""   # unreachable, but satisfies type checker


def _parse_llm_json(raw: str) -> dict:
    """Parse JSON from LLM response, stripping markdown fences if present."""
    import json
    # Strip ```json ... ``` fences
    clean = raw.strip()
    if clean.startswith("```"):
        clean = clean.split("```", 2)[1]
        if clean.startswith("json"):
            clean = clean[4:]
        clean = clean.rsplit("```", 1)[0]
    try:
        return json.loads(clean)
    except json.JSONDecodeError as e:
        logger.error("JSON decode failed: %s\nRaw: %s", e, raw[:300])
        return {}


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------

def analyse_review(
    processed_review,
    client: Optional[OpenAI] = None,
    model: str = DEFAULT_MODEL,
) -> LLMResult:
    """
    Analyse a single ProcessedReview and return an LLMResult.

    If the review has multiple chunks, each chunk is analysed separately
    and the results are merged in a second LLM call.
    """
    if client is None:
        client = _build_client()

    try:
        chunks = processed_review.chunks
        if not chunks:
            return LLMResult(
                sentiment="Neutral",
                sentiment_score=3.0,
                summary="No review text available.",
                key_points=[],
                model_used=model,
                error="empty_review",
            )

        if len(chunks) == 1:
            raw = _call_with_retry(client, model, SINGLE_CHUNK_PROMPT.format(review_text=chunks[0]))
            data = _parse_llm_json(raw)
        else:
            # Step 1: summarise each chunk independently
            partial: list[str] = []
            for i, chunk in enumerate(chunks, 1):
                logger.debug("Processing chunk %d/%d", i, len(chunks))
                raw = _call_with_retry(client, model, SINGLE_CHUNK_PROMPT.format(review_text=chunk))
                chunk_data = _parse_llm_json(raw)
                partial.append(
                    f"Chunk {i}: {chunk_data.get('summary', raw[:200])}"
                )

            # Step 2: merge partial summaries
            merged_prompt = MERGE_CHUNKS_PROMPT.format(partial_summaries="\n".join(partial))
            raw = _call_with_retry(client, model, merged_prompt)
            data = _parse_llm_json(raw)

        return LLMResult(
            sentiment=str(data.get("sentiment", "Neutral")),
            sentiment_score=float(data.get("sentiment_score", 3.0)),
            summary=str(data.get("summary", "")),
            key_points=list(data.get("key_points", [])),
            model_used=model,
        )

    except Exception as exc:
        logger.error("LLM analysis failed: %s", exc)
        return LLMResult(
            sentiment="Unknown",
            sentiment_score=0.0,
            summary="",
            key_points=[],
            model_used=model,
            error=str(exc),
        )


def analyse_all(
    processed_reviews: list,
    model: str = DEFAULT_MODEL,
    inter_request_delay: float = 0.5,
) -> list[LLMResult]:
    """
    Analyse a list of ProcessedReview objects.

    Args:
        processed_reviews:    Output of preprocessor.preprocess_all().
        model:                Groq model identifier.
        inter_request_delay:  Seconds to sleep between reviews (rate-limit buffer).

    Returns:
        List of LLMResult objects in the same order.
    """
    client = _build_client()
    results: list[LLMResult] = []

    for i, review in enumerate(processed_reviews):
        logger.info("Analysing review %d/%d by %s…", i + 1, len(processed_reviews), review.author or "Unknown")
        result = analyse_review(review, client=client, model=model)
        results.append(result)
        logger.info("  → Sentiment: %s (%.1f) | Summary: %s",
                    result.sentiment, result.sentiment_score, result.summary[:80])

        if i < len(processed_reviews) - 1:
            time.sleep(inter_request_delay)

    logger.info("LLM analysis complete. %d results.", len(results))
    return results
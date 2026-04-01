# Scraper-V1 🔍🤖

> **AI Engineer Intern – Assignment 2**
> Scrapes public customer reviews from **Trustpilot**, preprocesses the text, and uses the **Groq LLM API** to generate per-review sentiment analysis and concise summaries. Results are persisted as both **CSV** and **JSON**.

---

## Table of Contents

1. [Demo Product URL](#demo-product-url)
2. [Architecture Overview](#architecture-overview)
3. [Project Structure](#project-structure)
4. [Setup & Installation](#setup--installation)
5. [Running the Application](#running-the-application)
6. [Output Schema](#output-schema)
7. [Design Choices](#design-choices)
8. [Limitations & Known Issues](#limitations--known-issues)
9. [Running Tests](#running-tests)

---

## Demo Product URL

```
https://www.trustpilot.com/review/www.netflix.com
```

Netflix on Trustpilot has **thousands of real reviews** spanning all 5 star ratings, a wide variety of text lengths, and multiple languages — ideal for demonstrating the full pipeline including chunking of long reviews and mixed sentiment detection.

You can substitute any other company, for example:
```
https://www.trustpilot.com/review/www.apple.com
https://www.trustpilot.com/review/www.amazon.com
netflix.com          ← bare domain also works; the scraper auto-expands it
```

---

## Architecture Overview

```
┌───────────────────┐  paginated HTML  ┌──────────────────┐  cleaned text  ┌──────────────────┐
│   Trustpilot      │ ───────────────► │   scraper.py     │ ─────────────► │ preprocessor.py  │
│   Company Page    │                  │  requests + BS4  │  chunk if long  │ clean · chunk    │
│ (server-rendered) │                  │  retry + backoff │                 │ token-count      │
└───────────────────┘                  └──────────────────┘                 └────────┬─────────┘
                                                                                      │ per-review chunks
                                                                                      ▼
                                                                            ┌──────────────────┐
                                                                            │  llm_client.py   │
                                                                            │  Groq API        │
                                                                            │  llama3-8b-8192  │
                                                                            │  rate-limit safe │
                                                                            └────────┬─────────┘
                                                                                      │ sentiment + summary
                                                                                      ▼
                                                                            ┌──────────────────┐
                                                                            │   storage.py     │
                                                                            │  CSV  +  JSON    │
                                                                            └──────────────────┘
```

---

## Project Structure

```
Scraper-V1/       ← project root
├── main.py                     # CLI entry-point; runs the 4-step pipeline
├── requirements.txt
├── conftest.py                 # pytest path setup
├── .env.example                # copy → .env, add GROQ_API_KEY
├── .gitignore
│
├── src/
│   ├── scraper.py              # Trustpilot scraper (paginated, retries, UA rotation)
│   ├── preprocessor.py         # Text cleaning + token-aware sentence chunking
│   ├── llm_client.py           # Groq API client (JSON mode, exponential back-off)
│   └── storage.py              # CSV + JSON serialisation with timestamps
│
├── output/                     # Auto-created; timestamped output files go here
│
└── tests/
    ├── test_preprocessor.py    # 13 unit tests
    └── test_storage.py         # 8 unit tests
```

---

## Setup & Installation

### Prerequisites

- Python **3.11+**
- A free [Groq API key](https://console.groq.com/keys) — no credit card required

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/trustpilot-review-analyser.git
cd trustpilot-review-analyser
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure your Groq API key

**Option A – .env file (recommended)**

```bash
cp .env.example .env
# Open .env and set:  GROQ_API_KEY=gsk_...
```

**Option B – shell export**

```bash
export GROQ_API_KEY="gsk_your_key_here"
```

> ⚠️ Never commit `.env` or hardcode keys in source code.

---

## Running the Application

### Quickstart

```bash
python main.py --url "https://www.trustpilot.com/review/www.netflix.com"
```

### Full CLI reference

```
usage: main.py [-h] --url URL [--pages N] [--max-reviews N] [--model MODEL]
               [--output-dir OUTPUT_DIR] [--max-chunk-tokens MAX_CHUNK_TOKENS]
               [--delay DELAY] [--verbose]

Required:
  --url TEXT              Trustpilot URL or bare company domain
                          e.g. "netflix.com" or the full Trustpilot URL

Optional:
  --pages N               Max review pages to scrape (default: 3 → ~60 reviews)
  --max-reviews N         Maximum total reviews to collect (unlimited if not specified)
  --model MODEL           Groq model to use (e.g. llama3-8b-8192, mixtral-8x7b-32768)
  --output-dir OUTPUT_DIR Directory to write CSV and JSON output files
  --max-chunk-tokens N    Max tokens per LLM chunk (default: 400)
  --delay FLOAT           Seconds between LLM calls (default: 0.5)
  --verbose / -v          Enable debug-level logging
```
### Controlling Review Count

The scraper supports configurable review limits for different use cases:

#### For Testing (Limited Reviews)
When developing or testing, use `--max-reviews` to limit the number of reviews processed:

```bash
# Scrape only 5 reviews for quick testing
python main.py --url "https://www.trustpilot.com/review/www.netflix.com" --max-reviews 5

# Scrape only 3 reviews with verbose logging
python main.py --url "https://www.trustpilot.com/review/www.netflix.com" --max-reviews 3 --verbose
```

#### For Presentations/Full Analysis (All Reviews)
For comprehensive analysis or presentations, omit `--max-reviews` to scrape all available reviews:

```bash
# Scrape all reviews from 5 pages (unlimited reviews)
python main.py --url "https://www.trustpilot.com/review/www.netflix.com" --pages 5

# Scrape all reviews with custom model and output directory
python main.py \
  --url "https://www.trustpilot.com/review/www.netflix.com" \
  --pages 10 \
  --model "moonshotai/kimi-k2-instruct-0905" \
  --output-dir "./presentation-results"
```

**Note**: Each page typically contains ~20-30 reviews. The scraper will stop at your specified `--max-reviews` limit or when all pages are exhausted, whichever comes first.

### Example with all options

```bash
python main.py \
  --url "https://www.trustpilot.com/review/www.netflix.com" \
  --pages 5 \
  --model "moonshotai/kimi-k2-instruct-0905" \
  --output-dir "./results" \
  --delay 1.0 \
  --verbose
```

### Console output

```
10:15:01  INFO      main  ════════════════════════════════════════════════════════
10:15:01  INFO      main  STEP 1/4  Scraping reviews from Trustpilot…
10:15:01  INFO      main  URL    : https://www.trustpilot.com/review/www.netflix.com
10:15:03  INFO   scraper  Found 20 review cards on page
10:15:07  INFO   scraper  Found 20 review cards on page
10:15:11  INFO   scraper  Found 20 review cards on page
10:15:11  INFO      main  ✓ Scraped 60 reviews.
10:15:11  INFO      main  STEP 2/4  Preprocessing review text…
10:15:11  INFO      main  ✓ Preprocessed 60 reviews.
10:15:11  INFO      main    Token stats — min: 8  max: 312  avg: 74
10:15:11  INFO      main  STEP 3/4  Sending reviews to Groq (llama3-8b-8192)…
10:15:12  INFO llm_client  Analysing review 1/60 by Sarah M…
10:15:12  INFO llm_client    → Sentiment: Negative (1.5) | Summary: Customer frustrated by…
...
10:15:58  INFO      main  ✓ LLM analysis complete.
10:15:58  INFO      main    Sentiment distribution: {'Negative': 28, 'Positive': 22, 'Neutral': 6, 'Mixed': 4}
10:15:58  INFO      main  STEP 4/4  Saving results to output/…
10:15:58  INFO   storage  CSV saved → output/reviews_20250115_101558.csv  (60 rows)
10:15:58  INFO   storage  JSON saved → output/reviews_20250115_101558.json
```

---

## Output Schema

| Field               | Type     | Description                                           |
|---------------------|----------|-------------------------------------------------------|
| `author`            | str      | Reviewer display name                                 |
| `location`          | str      | Reviewer country (from Trustpilot profile)            |
| `rating`            | float    | Star rating 1–5                                       |
| `date`              | str      | ISO 8601 datetime of the review                       |
| `verified_purchase` | bool     | Whether marked as Verified                            |
| `title`             | str      | Review headline                                       |
| `review_text`       | str      | Cleaned review body                                   |
| `approx_tokens`     | int      | Approximate token count (debugging aid)               |
| `sentiment`         | str      | `Positive` / `Negative` / `Neutral` / `Mixed`        |
| `sentiment_score`   | float    | 1.0 (very negative) → 5.0 (very positive)            |
| `summary`           | str      | 1–3 sentence LLM-generated summary                   |
| `key_points`        | list/str | Extracted pros/cons (list in JSON, `;`-joined in CSV) |
| `model_used`        | str      | Groq model identifier                                 |
| `llm_error`         | str      | Error message if LLM call failed (empty if OK)        |
| `source_url`        | str      | Trustpilot page URL                                   |
| `scraped_at`        | str      | UTC scrape timestamp (ISO 8601)                       |

---

## Design Choices

### Why Trustpilot?
Amazon, Flipkart, and BestBuy aggressively block scrapers with CAPTCHAs, JavaScript rendering requirements, and IP bans. Trustpilot renders reviews **server-side** (no JS required), is publicly accessible, and does not require login — making it the most practical real-world source for this assignment.

### Scraper
- **requests + BeautifulSoup** over Selenium/Playwright: Trustpilot's review pages are server-side rendered, so no headless browser is needed. This keeps the solution lightweight and fast.
- **Multiple CSS selector fallbacks + data-attribute targeting**: Trustpilot's class names are obfuscated. The scraper primarily targets stable `data-*` attributes (e.g. `data-service-review-rating`, `data-consumer-name-typography`) which are more resilient to UI updates than class names.
- **Session warm-up**: A homepage visit before scraping review pages picks up cookies and makes the session look more browser-like.
- **User-Agent rotation + polite random delays** (2–4s between pages) to avoid bot detection.

### Preprocessor
- **NFKC Unicode normalisation** handles ligatures, fullwidth characters, and encoding artefacts.
- **Sentence-boundary-aware chunking** with overlap buffer: long reviews are split at sentence boundaries, not word boundaries, preserving semantic coherence. A configurable overlap (default 50 tokens) ensures context isn't lost at chunk boundaries.
- **Lightweight token estimate** (÷4 chars/token) avoids a hard tiktoken dependency while being accurate enough (~5% error on English text) for chunk gating.

### LLM Client
- **Groq llama3-8b-8192**: ~500 tokens/sec throughput, free tier, and accurate enough for sentiment/summarisation. Can be swapped for `llama3-70b-8192` or `mixtral-8x7b-32768` via `--model`.
- **JSON mode** (`response_format={"type": "json_object"}`): forces structured output, eliminating fragile regex parsing.
- **Two-pass chunking**: reviews exceeding `max_chunk_tokens` are summarised chunk-by-chunk then merged in a second call — cheaper and more accurate than blind truncation.
- **Exponential back-off with jitter** on 429 (rate limit) and 5xx errors.

### Storage
- **`utf-8-sig` BOM on CSV**: ensures the file opens correctly in Excel without garbled Unicode characters.
- **Timestamped filenames**: prevents accidental overwrites across multiple runs.
- **JSON retains `key_points` as a native array**: makes downstream processing trivial without split/parse gymnastics.

---

## Limitations & Known Issues

1. **Trustpilot bot detection at scale**: For > 10 pages per run, Trustpilot may serve a Cloudflare challenge. The default 2–4s inter-page delay mitigates this for small scrapes. A Playwright-based scraper would be needed for bulk collection.

2. **No JavaScript fallback**: The scraper uses `requests` and cannot execute JavaScript. If Trustpilot migrates to client-side rendering, a headless browser would be required.

3. **CSS selector drift**: Trustpilot's `data-*` attributes are stable but not guaranteed forever. The selector list was verified in April 2025.

4. **Approximate token counting**: The ÷4 heuristic is accurate for English but may undercount for non-Latin scripts (Arabic, Chinese, etc.) where tokens map to fewer characters. Swap `_approx_token_count` for `tiktoken` if multilingual precision is needed.

5. **Groq free-tier rate limits**: 30 requests/min on llama3-8b. The default `--delay 0.5` is conservative; increase to `--delay 2.0` if you hit 429 errors.

---

## Running Tests

```bash
pytest tests/ -v
```

Expected:

```
tests/test_preprocessor.py::TestCleanText::test_html_entities          PASSED
tests/test_preprocessor.py::TestCleanText::test_strip_html_tags        PASSED
tests/test_preprocessor.py::TestCleanText::test_unicode_normalization   PASSED
tests/test_preprocessor.py::TestCleanText::test_collapse_whitespace    PASSED
tests/test_preprocessor.py::TestCleanText::test_collapse_repeated_punctuation PASSED
tests/test_preprocessor.py::TestCleanText::test_empty_string           PASSED
tests/test_preprocessor.py::TestCleanText::test_control_characters_removed PASSED
tests/test_preprocessor.py::TestChunkText::test_short_text_no_chunking PASSED
tests/test_preprocessor.py::TestChunkText::test_long_text_is_chunked   PASSED
tests/test_preprocessor.py::TestChunkText::test_chunks_are_non_empty   PASSED
tests/test_preprocessor.py::TestChunkText::test_total_content_preserved PASSED
tests/test_preprocessor.py::TestPreprocessReview::...                  PASSED
tests/test_storage.py::TestSaveResults::...                             PASSED
=================== 21 passed in 0.51s ===================
```

---

## Dependencies

| Package          | Purpose                                      |
|------------------|----------------------------------------------|
| `requests`       | HTTP scraping                                |
| `beautifulsoup4` | HTML parsing                                 |
| `lxml`           | Fast BS4 parser backend                      |
| `pandas`         | DataFrame operations + CSV output            |
| `numpy`          | Numerical utilities                          |
| `openai`         | OpenAI-compatible client (used with Groq)    |
| `tiktoken`       | Precise token counting (optional)            |
| `python-dotenv`  | Load `.env` file for `GROQ_API_KEY`          |
| `pytest`         | Unit testing                                 |

---

*Built for the AI Engineer Intern assignment. All scraping is performed responsibly on publicly available Trustpilot review pages for educational purposes only.*
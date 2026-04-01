"""
scraper.py
----------
Scrapes customer reviews from a Trustpilot company page.

Trustpilot renders reviews server-side, so plain requests + BeautifulSoup
works reliably without JavaScript execution.

Example URL formats accepted:
  https://www.trustpilot.com/review/www.amazon.com
  https://www.trustpilot.com/review/netflix.com
  www.amazon.com                          (bare domain – auto-expanded)
"""

import time
import random
import logging
import re
from dataclasses import dataclass
from typing import Optional

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Review:
    author: str = ""
    rating: Optional[float] = None
    title: str = ""
    text: str = ""
    date: str = ""
    verified_purchase: bool = False
    helpful_votes: int = 0
    url: str = ""
    location: str = ""          # Trustpilot includes reviewer country


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

HEADERS_POOL = [
    {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "DNT": "1",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
    },
    {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/605.1.15 (KHTML, like Gecko) "
            "Version/17.4.1 Safari/605.1.15"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-GB,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
    },
    {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64; rv:125.0) "
            "Gecko/20100101 Firefox/125.0"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
    },
]


def _get_headers() -> dict:
    return dict(random.choice(HEADERS_POOL))


def _normalize_url(url: str) -> str:
    """
    Accept either a full Trustpilot URL or a bare domain and return
    the canonical Trustpilot review URL.

    Examples:
      "www.amazon.com"                            → "https://www.trustpilot.com/review/www.amazon.com"
      "https://www.trustpilot.com/review/x.com"  → unchanged
      "netflix.com"                               → "https://www.trustpilot.com/review/netflix.com"
    """
    url = url.strip()
    if "trustpilot.com/review/" in url:
        if not url.startswith("http"):
            url = "https://" + url
        return url
    domain = re.sub(r"^https?://", "", url).rstrip("/")
    return f"https://www.trustpilot.com/review/{domain}"


def _page_url(base_review_url: str, page: int) -> str:
    """Append ?page=N to the Trustpilot review URL."""
    base = base_review_url.rstrip("/")
    return f"{base}?page={page}"


# ---------------------------------------------------------------------------
# Fetch with retries
# ---------------------------------------------------------------------------

def _fetch_html(url: str, session: requests.Session, retries: int = 4) -> Optional[str]:
    """Fetch a page with exponential back-off on errors."""
    for attempt in range(1, retries + 1):
        try:
            headers = _get_headers()
            response = session.get(url, headers=headers, timeout=20)

            if response.status_code == 404:
                logger.error("404 – page not found: %s", url)
                return None

            if response.status_code == 403:
                logger.warning("403 Forbidden on attempt %d – backing off…", attempt)
                time.sleep(5 * attempt)
                continue

            response.raise_for_status()
            return response.text

        except requests.exceptions.Timeout:
            logger.warning("Timeout on attempt %d for %s", attempt, url)
        except requests.exceptions.ConnectionError as e:
            logger.warning("Connection error on attempt %d: %s", attempt, e)
        except requests.exceptions.HTTPError as e:
            logger.warning("HTTP error on attempt %d: %s", attempt, e)
        except requests.exceptions.RequestException as e:
            logger.error("Fatal request error: %s", e)
            return None

        if attempt < retries:
            wait = 2 ** attempt + random.uniform(0.5, 2.0)
            logger.info("Retrying in %.1fs…", wait)
            time.sleep(wait)

    logger.error("All %d attempts failed for %s", retries, url)
    return None


# ---------------------------------------------------------------------------
# Parse a single Trustpilot review page
# ---------------------------------------------------------------------------

def _extract_rating(card) -> Optional[float]:
    """
    Trustpilot encodes star rating in multiple ways; try all of them.
    """
    if card is None:
        return None

    # Method 1: data attribute directly on the article card
    rating_attr = card.get("data-service-review-rating")
    if rating_attr:
        try:
            return float(rating_attr)
        except ValueError:
            pass

    # Method 2: <img alt="Rated N out of 5 stars">
    img = card.find("img", alt=re.compile(r"Rated \d", re.I))
    if img:
        match = re.search(r"Rated (\d+(?:\.\d+)?)", img.get("alt", ""), re.I)
        if match:
            return float(match.group(1))

    # Method 3: aria-label on star container
    star_div = card.find(attrs={"aria-label": re.compile(r"\d.*star", re.I)})
    if star_div:
        match = re.search(r"(\d+(?:\.\d+)?)", star_div.get("aria-label", ""))
        if match:
            return float(match.group(1))

    return None


def _parse_reviews(html: str, page_url: str) -> list[Review]:
    """Parse all review cards from a single Trustpilot review page."""
    soup = BeautifulSoup(html, "lxml")
    reviews: list[Review] = []

    # Trustpilot uses <article> tags with a data-service-review-rating attribute
    cards = (
        soup.find_all("article", attrs={"data-service-review-rating": True})
        or soup.find_all("article", class_=re.compile(r"review", re.I))
        or soup.find_all("div", attrs={"data-review-content": True})
    )

    logger.info("Found %d review cards on page", len(cards))

    for card in cards:
        review = Review(url=page_url)

        # ── Rating ──────────────────────────────────────────────────────────
        review.rating = _extract_rating(card)

        # ── Title ───────────────────────────────────────────────────────────
        title_tag = (
            card.find("h2", attrs={"data-service-review-title-typography": True})
            or card.find("h2", class_=re.compile(r"title|heading", re.I))
            or card.find("h2")
        )
        if title_tag:
            review.title = title_tag.get_text(strip=True)

        # ── Review body ─────────────────────────────────────────────────────
        body_tag = (
            card.find("p", attrs={"data-service-review-text-typography": True})
            or card.find("p", class_=re.compile(r"reviewContent|review-content|body", re.I))
        )
        if not body_tag:
            # Fallback: longest <p> in the card
            all_p = card.find_all("p")
            if all_p:
                body_tag = max(all_p, key=lambda p: len(p.get_text()))
        if body_tag:
            review.text = body_tag.get_text(separator=" ", strip=True)

        # ── Author ──────────────────────────────────────────────────────────
        author_tag = (
            card.find(attrs={"data-consumer-name-typography": True})
            or card.find(class_=re.compile(r"consumerName|consumer-name|author", re.I))
        )
        if author_tag:
            review.author = author_tag.get_text(strip=True)

        # ── Date ────────────────────────────────────────────────────────────
        date_tag = card.find("time")
        if date_tag:
            review.date = date_tag.get("datetime", date_tag.get_text(strip=True))

        # ── Location ────────────────────────────────────────────────────────
        location_tag = card.find(attrs={"data-consumer-country-typography": True})
        if location_tag:
            review.location = location_tag.get_text(strip=True)

        # ── Verified ────────────────────────────────────────────────────────
        verified = card.find(string=re.compile(r"verified", re.I))
        review.verified_purchase = verified is not None

        if review.text or review.title:
            reviews.append(review)

    return reviews


def _has_next_page(html: str) -> bool:
    """Return True if a 'Next page' navigation link exists."""
    soup = BeautifulSoup(html, "lxml")
    return bool(
        soup.find("a", attrs={"data-pagination-button-next-link": True})
        or soup.find("a", attrs={"name": "pagination-button-next"})
        or soup.find(attrs={"aria-label": re.compile(r"next page", re.I)})
        or soup.find("a", string=re.compile(r"^next$", re.I))
    )


# ---------------------------------------------------------------------------
# Public entry-point
# ---------------------------------------------------------------------------

def scrape_reviews(
    product_url: str,
    max_pages: int = 5,
    max_reviews: Optional[int] = None,
    delay_range: tuple = (2.0, 4.0),
) -> list[Review]:
    """
    Scrape up to `max_pages` pages of Trustpilot reviews.

    Args:
        product_url:  Trustpilot URL OR bare company domain.
                      e.g. "https://www.trustpilot.com/review/www.netflix.com"
                       or  "netflix.com"
        max_pages:    Maximum pages to scrape (≈20 reviews/page).
        delay_range:  Polite (min, max) delay in seconds between requests.

    Returns:
        List of Review dataclass instances.
    """
    base_url = _normalize_url(product_url)
    logger.info("Trustpilot review URL: %s", base_url)

    session = requests.Session()
    # Warm the session with a homepage visit to pick up cookies
    try:
        session.get("https://www.trustpilot.com", headers=_get_headers(), timeout=10)
        time.sleep(random.uniform(1.0, 2.0))
    except Exception:
        pass  # non-fatal if this fails

    all_reviews: list[Review] = []

    for page in range(1, max_pages + 1):
        url = _page_url(base_url, page)
        logger.info("Scraping page %d → %s", page, url)

        html = _fetch_html(url, session)
        if not html:
            logger.warning("No HTML returned for page %d. Stopping.", page)
            break

        page_reviews = _parse_reviews(html, url)
        if not page_reviews:
            logger.info("No reviews on page %d — likely past the last page.", page)
            break

        # Limit reviews to reach the target of max_reviews total
        if max_reviews is not None and max_reviews > 0:
            remaining_needed = max_reviews - len(all_reviews)
            if remaining_needed <= 0:
                break
            
            reviews_to_add = page_reviews[:remaining_needed]
            all_reviews.extend(reviews_to_add)
            logger.info("Running total: %d reviews", len(all_reviews))

            if len(all_reviews) >= max_reviews:
                logger.info("Reached %d reviews limit. Stopping.", max_reviews)
                break
        else:
            # No limit - add all reviews from the page
            all_reviews.extend(page_reviews)
            logger.info("Running total: %d reviews", len(all_reviews))

        if not _has_next_page(html):
            logger.info("No next-page link found — reached end of reviews.")
            break

        if page < max_pages:
            sleep = random.uniform(*delay_range)
            logger.debug("Sleeping %.1fs…", sleep)
            time.sleep(sleep)

    logger.info("Done. Collected %d reviews total.", len(all_reviews))
    return all_reviews
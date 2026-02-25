"""
trends_service.py — Fetch Google Trends data via SerpApi.

Uses SerpApi's Google Trends engine for real interest-over-time
data and related queries. Falls back gracefully if the key is
missing or the request fails.
"""

import os
import requests
from logger_config import get_logger

logger = get_logger(__name__)

SERPAPI_BASE = "https://serpapi.com/search"


def get_google_trends(query: str, geo: str = "US", date: str = "today 3-m") -> dict:
    """Fetch Google Trends interest-over-time data for *query*.

    Args:
        query: Search term (max 100 chars).
        geo: Two-letter country code (default US).
        date: Time range — e.g. ``"today 3-m"`` (90 days),
              ``"today 12-m"`` (1 year), ``"now 7-d"`` (7 days).

    Returns the raw SerpApi JSON, or an empty dict on failure.
    """
    api_key = os.environ.get("SERPAPI_API_KEY", "")
    if not api_key:
        logger.debug("SERPAPI_API_KEY not set; skipping Google Trends lookup")
        return {}

    try:
        resp = requests.get(
            SERPAPI_BASE,
            params={
                "engine": "google_trends",
                "q": query[:100],
                "data_type": "TIMESERIES",
                "date": date,
                "geo": geo,
                "api_key": api_key,
            },
            timeout=15,
        )
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as e:
        logger.warning("SerpApi Google Trends request failed: %s", e)
        return {}


def get_related_queries(query: str, geo: str = "US") -> list[str]:
    """Fetch rising related queries for *query* from Google Trends."""
    api_key = os.environ.get("SERPAPI_API_KEY", "")
    if not api_key:
        return []

    try:
        resp = requests.get(
            SERPAPI_BASE,
            params={
                "engine": "google_trends",
                "q": query[:100],
                "data_type": "RELATED_QUERIES",
                "geo": geo,
                "api_key": api_key,
            },
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()

        words = []
        # SerpApi nests under related_queries → rising/top
        for section in ("rising", "top"):
            for item in data.get("related_queries", {}).get(section, []):
                q = item.get("query", "")
                if q:
                    words.append(q)
        return words[:15]
    except requests.RequestException as e:
        logger.warning("SerpApi related queries request failed: %s", e)
        return []


def _trends_to_time_series(trends_data: dict) -> list[dict]:
    """Convert SerpApi Trends response to ``[{date, count}]`` dicts.

    Compatible with ``similar_posts_time_series`` in NewsAnalysisReport.
    """
    time_series = []
    # SerpApi returns interest_over_time → timeline_data → [{date, values}]
    timeline = (
        trends_data
        .get("interest_over_time", {})
        .get("timeline_data", [])
    )
    for point in timeline:
        date = point.get("date", "")
        # values is a list of dicts [{extracted_value, value}] — one per query
        values = point.get("values", [{}])
        value = values[0].get("extracted_value", 0) if values else 0
        if date:
            time_series.append({"date": date, "count": int(value)})

    return time_series


def get_trends_metrics(query: str, geo: str = "US") -> dict:
    """High-level helper: fetch trends + related queries and return
    a dict that can be merged into ``platform_metrics``.

    Returns empty dict if SERPAPI_API_KEY is not configured.
    """
    trends_data = get_google_trends(query, geo)
    if not trends_data:
        return {}

    ts = _trends_to_time_series(trends_data)
    related = get_related_queries(query, geo)

    return {
        "trends_time_series": ts,
        "trends_related_words": related,
    }

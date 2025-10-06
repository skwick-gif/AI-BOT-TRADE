"""
Sentiment aggregation service using NewsAPI and VADER sentiment.
Optional future providers (Reddit/Twitter) can be added behind the same interface.
"""

from __future__ import annotations

import os
import datetime as dt
from typing import List, Dict, Any

import requests
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer  # type: ignore
    _HAS_VADER = True
except Exception:
    _HAS_VADER = False
    try:
        from textblob import TextBlob  # type: ignore
    except Exception:
        TextBlob = None  # type: ignore


def fetch_news_articles(symbol: str, start: dt.datetime, end: dt.datetime, api_key: str, page_size: int = 50) -> List[Dict[str, Any]]:
    """Fetch news articles from NewsAPI for a symbol in a time window.

    Note: NewsAPI Everything endpoint rate-limits; we keep it small and focused.
    """
    if not api_key:
        return []
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": symbol,
        "from": start.strftime("%Y-%m-%d"),
        "to": end.strftime("%Y-%m-%d"),
        "language": "en",
        "sortBy": "relevancy",
        "pageSize": min(max(page_size, 1), 100),
        "apiKey": api_key,
    }
    try:
        r = requests.get(url, params=params, timeout=15)
        r.raise_for_status()
        data = r.json()
        return data.get("articles", [])
    except Exception:
        return []


def score_sentiment_vader(texts: List[str]) -> Dict[str, Any]:
    """Compute VADER sentiment over a list of texts; return aggregate and per-item scores."""
    per = []
    if _HAS_VADER:
        analyzer = SentimentIntensityAnalyzer()
        for t in texts:
            try:
                per.append(analyzer.polarity_scores(t or ""))
            except Exception:
                per.append({"compound": 0.0})
        compounds = [d.get("compound", 0.0) for d in per]
        avg = sum(compounds) / len(compounds) if compounds else 0.0
        return {"avg_compound": avg, "details": per}
    else:
        # Fallback to TextBlob polarity scaled to [-1,1]
        if TextBlob is None:
            return {"avg_compound": 0.0, "details": []}
        vals = []
        for t in texts:
            try:
                vals.append(TextBlob(t or "").sentiment.polarity)
            except Exception:
                vals.append(0.0)
        avg = sum(vals) / len(vals) if vals else 0.0
        return {"avg_compound": avg, "details": [{"compound": v} for v in vals]}


def aggregate_symbol_sentiment(symbol: str, start: dt.datetime, end: dt.datetime) -> Dict[str, Any]:
    """Aggregate symbol sentiment from available sources (NewsAPI only for now)."""
    news_api = os.getenv("NEWSAPI_KEY", "").strip()
    articles = fetch_news_articles(symbol, start, end, news_api)
    texts = [
        f"{a.get('title','')} {a.get('description','')} {a.get('content','')}".strip()
        for a in articles
    ]
    sent = score_sentiment_vader(texts)
    return {
        "source": "newsapi",
        "count": len(texts),
        "avg_compound": sent["avg_compound"],
        "window": {"start": start.isoformat(), "end": end.isoformat()},
    }

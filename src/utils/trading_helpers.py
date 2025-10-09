"""
Trading helper functions ported from Streamlit version
Includes portfolio management, technical analysis, and market data functions
"""

import os
import re
import pandas as pd
# Note: we avoid calling yfinance for offline-first operation; local data loaders are used instead.
try:
    import yfinance as yf
except Exception:
    yf = None
try:
    import pandas_ta as ta
except Exception:
    ta = None
    # don't raise here; some environments may not have a compatible pandas_ta
    # functions that rely on it will handle ta==None gracefully
import requests
from datetime import datetime, date, timedelta
from typing import List, Dict, Any, Optional
import numpy as np
from dotenv import load_dotenv

# Load environment variables from project .env
load_dotenv()

# API Keys and Config from env
PERPLEXITY_API_KEY = os.getenv('PERPLEXITY_API_KEY', '')
PERPLEXITY_MODEL = os.getenv('PERPLEXITY_MODEL', 'sonar-small-chat')
NEWSAPI_KEY = os.getenv('NEWSAPI_KEY', '')
QUANTIQ_API_KEY = os.getenv('QUANTIQ_API', '')
REDDIT_CLIENT_ID = os.getenv('REDDIT_CLIENT_ID', '')
REDDIT_CLIENT_SECRET = os.getenv('REDDIT_CLIENT_SECRET', '')
REDDIT_USER_AGENT = os.getenv('REDDIT_USER_AGENT', 'StockAnalysisBot/1.0')

from utils.logger import get_logger

logger = get_logger("TradingHelpers")

# Portfolio CSV file
PORTFOLIO_CSV = "portfolio.csv"


def update_portfolio(ticker: str, action: str, shares: int, price: float) -> None:
    """
    Update portfolio CSV file with trade
    
    Args:
        ticker: Stock symbol
        action: "BUY" or "SELL"
        shares: Number of shares
        price: Price per share
    """
    try:
        # Create new trade record
        trade = {
            'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'ticker': ticker,
            'action': action,
            'shares': shares,
            'price': price,
            'total': shares * price
        }
        
        # Load existing portfolio or create new
        if os.path.exists(PORTFOLIO_CSV):
            portfolio_df = pd.read_csv(PORTFOLIO_CSV)
        else:
            portfolio_df = pd.DataFrame(columns=['date', 'ticker', 'action', 'shares', 'price', 'total'])
        
        # Add new trade
        new_row = pd.DataFrame([trade])
        portfolio_df = pd.concat([portfolio_df, new_row], ignore_index=True)
        
        # Save updated portfolio
        portfolio_df.to_csv(PORTFOLIO_CSV, index=False)
        logger.info(f"Portfolio updated: {action} {shares} shares of {ticker} at ${price:.2f}")
        
    except Exception as e:
        logger.error(f"Error updating portfolio: {e}")


def get_portfolio_positions() -> Dict[str, int]:
    """
    Get current positions from portfolio CSV
    
    Returns:
        Dictionary mapping ticker to current shares held
    """
    try:
        if not os.path.exists(PORTFOLIO_CSV):
            return {}
        
        portfolio_df = pd.read_csv(PORTFOLIO_CSV)
        if portfolio_df.empty:
            return {}
        
        positions = {}
        for ticker in portfolio_df['ticker'].unique():
            ticker_data = portfolio_df[portfolio_df['ticker'] == ticker]
            buys = ticker_data[ticker_data['action'] == 'BUY']['shares'].sum()
            sells = ticker_data[ticker_data['action'] == 'SELL']['shares'].sum()
            net_position = buys - sells
            if net_position > 0:
                positions[ticker] = net_position
        
        return positions
        
    except Exception as e:
        logger.error(f"Error getting portfolio positions: {e}")
        return {}


def get_current_price(ticker: str) -> Optional[float]:
    """
    Get current stock price using yfinance
    
    Args:
        ticker: Stock symbol
        
    Returns:
        Current price or None if error
    """
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period="1d", interval="1m")
        if not data.empty:
            return float(data['Close'].iloc[-1])
        return None
    except Exception as e:
        logger.error(f"Error getting price for {ticker}: {e}")
        return None


def get_technicals(ticker: str) -> pd.DataFrame:
    """
    Get technical analysis data for a stock
    
    Args:
        ticker: Stock symbol
        
    Returns:
        DataFrame with technical indicators
    """
    try:
        # Offline-first: attempt to load local price data from parquet or stock_data folder
        def _load_local_price_data(ticker_sym: str) -> pd.DataFrame | None:
            from pathlib import Path

            ticker_lower = ticker_sym.lower()

            # 1) check data/bronze/daily/<TICKER>.parquet
            data_dir = Path("data") / "bronze" / "daily"
            try:
                if data_dir.exists():
                    for p in data_dir.glob('*.parquet'):
                        if p.stem.lower() == ticker_lower:
                            logger.info(f"Loading local parquet for {ticker_sym}: {p}")
                            df = pd.read_parquet(p)
                            return df
            except Exception as e:
                logger.warning(f"Error reading parquet files in {data_dir}: {e}")

            # 2) search stock_data/ recursively for matching csv/parquet file
            stock_data_root = Path("stock_data")
            try:
                if stock_data_root.exists():
                    for p in stock_data_root.rglob('*'):
                        if not p.is_file():
                            continue
                        stem = p.stem.lower()
                        # accept exact stem match or stem that starts with ticker (some filenames like TICKER_prices)
                        if stem == ticker_lower or stem.startswith(ticker_lower + '_') or stem.startswith(ticker_lower):
                            try:
                                if p.suffix.lower() == '.parquet':
                                    df = pd.read_parquet(p)
                                elif p.suffix.lower() in ('.csv', '.txt'):
                                    df = pd.read_csv(p)
                                else:
                                    continue
                                logger.info(f"Loading local stock_data file for {ticker_sym}: {p}")
                                return df
                            except Exception:
                                continue
            except Exception as e:
                logger.warning(f"Error searching stock_data folder: {e}")

            return None

        data = _load_local_price_data(ticker)

        # If local data not found, return empty (offline-first: do not call yfinance)
        if data is None or (hasattr(data, 'empty') and data.empty):
            logger.warning(f"No local price data available for {ticker}; offline-only mode")
            return pd.DataFrame()

        # Normalize columns: accept common variants and ensure required columns exist
        try:
            # ensure date column
            date_cols = [c for c in data.columns if c.lower() in ('date', 'datetime', 'time', 'timestamp', 'index')]
            if 'date' not in data.columns and date_cols:
                data = data.rename(columns={date_cols[0]: 'date'})
            if 'date' in data.columns:
                data['date'] = pd.to_datetime(data['date'])
                data = data.sort_values('date').reset_index(drop=True)
            # ensure close/adj_close
            if 'close' not in data.columns and 'Close' in data.columns:
                data = data.rename(columns={'Close': 'close'})
            if 'adj_close' not in data.columns:
                if 'Adj Close' in data.columns:
                    data = data.rename(columns={'Adj Close': 'adj_close'})
                else:
                    data['adj_close'] = data['close'] if 'close' in data.columns else pd.NA
        except Exception as e:
            logger.warning(f"Failed to normalize price data for {ticker}: {e}")
        
        # Calculate technical indicators if pandas_ta is available
        if ta is None:
            logger.warning("pandas_ta not available; skipping technical indicator calculation")
            return data

        try:
            # Use pandas_ta accessor
            data.ta.sma(length=20, append=True)
            data.ta.sma(length=50, append=True)
            data.ta.rsi(append=True)
            data.ta.macd(append=True)
            data.ta.bbands(append=True)
            data.ta.stoch(append=True)
        except Exception as e:
            logger.warning(f"pandas_ta calculation failed: {e}")

        return data
        
    except Exception as e:
        logger.error(f"Error getting technicals for {ticker}: {e}")
        return pd.DataFrame()


def get_company_name(ticker: str) -> str:
    """
    Get company name from ticker
    
    Args:
        ticker: Stock symbol
        
    Returns:
        Company name or ticker if not found
    """
    # Offline mode: company name not available via local parquet; return ticker
    return ticker


def get_financials(ticker: str) -> Dict[str, Any]:
    """
    Get basic financial data for a stock
    
    Args:
        ticker: Stock symbol
        
    Returns:
        Dictionary with financial metrics
    """
    # Offline mode: financials not available via yfinance; return empty dict
    logger.warning(f"get_financials: offline mode, financials not available for {ticker}")
    return {}


def get_small_cap_stocks() -> List[str]:
    """
    Get a list of small/micro-cap US stock tickers via Perplexity.
    Falls back to a static list if API key not present or call fails.
    """
    fallback = ['CRWD', 'ZM', 'DOCU', 'OKTA', 'TWLO', 'SNOW', 'NET', 'DDOG']
    try:
        if not PERPLEXITY_API_KEY:
            logger.warning("Perplexity API key missing; returning fallback small-cap list")
            return fallback[:5]

        headers = {
            'Authorization': f'Bearer {PERPLEXITY_API_KEY}',
            'Content-Type': 'application/json'
        }
        prompt = (
            "Provide a comma-separated list of 10 interesting US micro-cap or small-cap stock tickers. "
            "Only return tickers, no explanations."
        )
        payload = {
            'model': PERPLEXITY_MODEL,
            'messages': [
                {"role": "system", "content": "You are an AI assistant that provides lists of stock tickers."},
                {"role": "user", "content": prompt}
            ],
            'max_tokens': 200,
            'temperature': 0.4
        }
        resp = requests.post(
            'https://api.perplexity.ai/chat/completions',
            headers=headers,
            json=payload,
            timeout=15
        )
        resp.raise_for_status()
        content = resp.json()['choices'][0]['message']['content']
        # Parse and clean symbols
        raw = re.split(r'[\s,]+', content)
        tickers = []
        for item in raw:
            sym = item.strip().upper().strip('$')
            if re.fullmatch(r'[A-Z\.\-]{1,6}', sym):
                tickers.append(sym)
        tickers = list(dict.fromkeys(tickers))  # dedupe, preserve order
        return tickers[:10] if tickers else fallback[:5]
    except Exception as e:
        logger.error(f"Error fetching small-cap stocks from Perplexity: {e}")
        return fallback[:5]


def get_stock_recommendation(ticker: str, financials: Dict[str, Any]) -> str:
    """
    Generate a basic stock recommendation based on financials
    
    Args:
        ticker: Stock symbol
        financials: Financial data dictionary
        
    Returns:
        Recommendation string
    """
    try:
        pe_ratio = financials.get('peRatio', 0)
        pb_ratio = financials.get('pbRatio', 0)
        profit_margin = financials.get('profitMargin', 0)
        debt_to_equity = financials.get('debtToEquity', 0)
        
        # Simple scoring system
        score = 0
        reasons = []
        
        # PE Ratio analysis
        if pe_ratio > 0:
            if pe_ratio < 15:
                score += 2
                reasons.append("Low P/E ratio indicates potential undervaluation")
            elif pe_ratio < 25:
                score += 1
                reasons.append("Moderate P/E ratio")
            else:
                reasons.append("High P/E ratio may indicate overvaluation")
        
        # P/B Ratio analysis
        if pb_ratio > 0:
            if pb_ratio < 1.5:
                score += 1
                reasons.append("Low P/B ratio suggests good value")
            elif pb_ratio > 3:
                score -= 1
                reasons.append("High P/B ratio may indicate overvaluation")
        
        # Profit margin
        if profit_margin > 0.1:
            score += 2
            reasons.append("Strong profit margins")
        elif profit_margin > 0.05:
            score += 1
            reasons.append("Decent profit margins")
        else:
            reasons.append("Low or negative profit margins")
        
        # Debt analysis
        if debt_to_equity > 0:
            if debt_to_equity < 0.3:
                score += 1
                reasons.append("Low debt levels")
            elif debt_to_equity > 1.0:
                score -= 1
                reasons.append("High debt levels")
        
        # Generate recommendation
        if score >= 4:
            action = "BUY"
        elif score >= 2:
            action = "HOLD"
        else:
            action = "SELL"
        
        recommendation = f"{action} - {ticker}\\n\\n"
        recommendation += f"Score: {score}/6\\n\\n"
        recommendation += "Analysis:\\n" + "\\n".join([f"• {reason}" for reason in reasons])
        
        return recommendation
        
    except Exception as e:
        logger.error(f"Error generating recommendation for {ticker}: {e}")
        return f"Unable to generate recommendation for {ticker} due to insufficient data."


def fetch_reddit_posts(ticker: str, limit: int = 50) -> pd.DataFrame:
    """
    Fetch Reddit posts containing stock ticker mentions using PRAW.
    Returns empty DataFrame if credentials are missing.
    """
    cols = ['title', 'score', 'subreddit', 'created_utc', 'url', 'comments', 'body']
    if not all([REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT]):
        logger.warning("Reddit API credentials not configured.")
        return pd.DataFrame(columns=cols)
    try:
        import praw  # Lazy import
        reddit = praw.Reddit(
            client_id=REDDIT_CLIENT_ID,
            client_secret=REDDIT_CLIENT_SECRET,
            user_agent=REDDIT_USER_AGENT
        )
        posts = []
        query = f'${ticker} OR {ticker}'
        for subreddit in ['wallstreetbets', 'stocks', 'investing']:
            try:
                for submission in reddit.subreddit(subreddit).search(query, limit=limit, time_filter='month'):
                    posts.append({
                        'title': submission.title,
                        'score': submission.score,
                        'url': f'https://reddit.com{submission.permalink}',
                        'comments': submission.num_comments,
                        'created_utc': datetime.fromtimestamp(submission.created_utc),
                        'subreddit': subreddit,
                        'flair': getattr(submission, 'link_flair_text', None),
                        'body': (submission.selftext[:500] + '...') if getattr(submission, 'selftext', '') and len(submission.selftext) > 500 else getattr(submission, 'selftext', '')
                    })
            except Exception as e:
                logger.warning(f"Error fetching from r/{subreddit}: {e}")
                continue
        return pd.DataFrame(posts)
    except Exception as e:
        logger.error(f"Error connecting to Reddit API: {e}")
        return pd.DataFrame(columns=cols)


def fetch_news(query: str) -> pd.DataFrame:
    """
    Fetch news articles for a specific query using NewsAPI.org.
    Returns empty DataFrame if NEWSAPI_KEY is missing or no results.
    """
    cols = ['title', 'source', 'published_at', 'url', 'description']
    if not NEWSAPI_KEY or NEWSAPI_KEY.lower().startswith('your_'):
        logger.warning("NEWSAPI_KEY not configured; returning empty news DataFrame")
        return pd.DataFrame(columns=cols)
    try:
        to_date = datetime.now().strftime('%Y-%m-%d')
        from_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        url = (
            f"https://newsapi.org/v2/everything?q={requests.utils.quote(query)}"
            f"&from={from_date}&to={to_date}&sortBy=publishedAt&language=en&apiKey={NEWSAPI_KEY}"
        )
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        if data.get('status') != 'ok' or not data.get('articles'):
            return pd.DataFrame(columns=cols)
        rows = []
        for a in data['articles']:
            rows.append({
                'title': a.get('title'),
                'source': a.get('source', {}).get('name'),
                'published_at': datetime.strptime(a.get('publishedAt'), '%Y-%m-%dT%H:%M:%SZ') if a.get('publishedAt') else None,
                'url': a.get('url'),
                'description': (a.get('description')[:200] + '...') if a.get('description') and len(a.get('description')) > 200 else a.get('description')
            })
        return pd.DataFrame(rows, columns=cols)
    except Exception as e:
        logger.error(f"Error fetching news for {query}: {e}")
        return pd.DataFrame(columns=cols)


def get_government_official_trades(ticker: str) -> str:
    """
    Uses the QuantiQ.live API to get trades done by House and Senate officials on the supplied ticker.
    Returns a concise string summary.
    """
    if not QUANTIQ_API_KEY:
        return "QuantiQ API key not configured. Please set QUANTIQ_API in your .env file."
    try:
        url = f"https://www.quantiq.live/api/get-congress-trades?simbol={ticker}"
        payload = f"apiKey={QUANTIQ_API_KEY}"
        headers = {'Content-Type': 'application/x-www-form-urlencoded'}
        resp = requests.post(url, headers=headers, data=payload, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        # Try to extract a readable summary
        trades = []
        nested = data
        try:
            # Navigate common nesting pattern
            nested = data.get('data', {}).get('data', {})
            # Remove heavy history if present
            if isinstance(nested, dict) and 'history' in nested:
                nested.pop('history', None)
        except Exception:
            pass
        # Build summary string
        return f"Congress trades for {ticker}: {nested if nested else 'No data'}"
    except Exception as e:
        logger.error(f"Error fetching congress trades for {ticker}: {e}")
        return f"Error fetching congress trades for {ticker}: {e}"


def get_polymarket_odds(ticker: str, limit: int = 3) -> str:
    """
    Fetch active Polymarket markets related to a ticker/company and summarize odds.
    """
    try:
        # Try to enrich search with company name
        company_name = get_company_name(ticker)
        keywords = list({ticker, company_name}) if company_name else [ticker]
        # Clean company name by removing common suffixes
        for suffix in [' Inc.', ' Corp.', ' Corporation', ' Company', ' Ltd.', ' Limited', ' PLC', ' NV']:
            if company_name and suffix in company_name:
                keywords.append(company_name.split(suffix)[0])
        keywords = [k for k in set(keywords) if k]

        url = "https://gamma-api.polymarket.com/markets"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json',
            'Referer': 'https://polymarket.com/'
        }
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        markets = data.get('markets') if isinstance(data, dict) else (data if isinstance(data, list) else [])
        if not markets:
            return f"No markets data found for {ticker}."
        filtered = []
        for m in markets:
            if not isinstance(m, dict):
                continue
            state = str(m.get('state', '')).lower()
            if state not in ['open', 'active', 'trading']:
                continue
            outcomes = m.get('outcomes', [])
            if not outcomes or not isinstance(outcomes, list):
                continue
            has_price = any(isinstance(o, dict) and isinstance(o.get('price'), (int, float)) and o.get('price') > 0 for o in outcomes)
            if not has_price:
                continue
            market_text = f"{m.get('question','')} {m.get('title','')} {m.get('description','')}".lower()
            if any(k.lower() in market_text for k in keywords):
                filtered.append(m)
        if not filtered:
            return f"No active prediction markets found for {ticker} ({company_name})."
        results = []
        for m in filtered[:limit]:
            question = m.get('question', m.get('title', 'Unknown Market'))
            outcome_prices = []
            for o in m.get('outcomes', []):
                if isinstance(o, dict) and 'price' in o:
                    price = o.get('price')
                    name = o.get('name', 'Option')
                    if isinstance(price, (int, float)) and price > 0:
                        outcome_prices.append(f"{name}: {price:.0%}")
            if outcome_prices:
                results.append(f"'{question}' → " + " | ".join(outcome_prices))
        return " | ".join(results) if results else f"Markets found for {ticker} but no current price data available."
    except Exception as e:
        logger.error(f"Error fetching Polymarket data for {ticker}: {e}")
        return f"Error fetching Polymarket data for {ticker}: {e}"
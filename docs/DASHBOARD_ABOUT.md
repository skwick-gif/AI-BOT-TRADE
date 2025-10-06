# Dashboard Overview

This document summarizes the Dashboard screen: sections, data sources, update logic, and integration points.

## Sections

- Account Overview
  - Source: IBKR via `ibkr_service` (account summary fields)
  - Metrics: Net Liquidation, Buying Power (or Excess Liquidity), Day P&L (Realized), Unrealized P&L
  - Update cadence: Auto-refresh based on `UI_UPDATE_INTERVAL` (ms) from `.env` (default 1000)

- Recent Activity
  - Source: IBKR connection state + account summary/positions counts
  - Purpose: quick connectivity and heartbeat status

- Portfolio Positions
  - Source: `ibkr_service.get_portfolio()` (live positions, prices, market value, unrealized P&L)
  - Manual sync button: “Sync IBKR” triggers a refresh

- Macro Indicators
  - Source: FRED via `fredapi` with `FRED_API_KEY`
  - Indicators: DFF (Fed Funds), UNRATE (Unemployment), CPIAUCSL (CPI)
  - Graph: click an indicator to open a mini chart (pyqtgraph) for the period 2024-01-01 → today
  - No demo data: if API missing or error, indicators remain empty with status message

- Calendar (Events)
  - Source: Finnhub via `FINNHUB_API_KEY`
  - Range: 2024-01-01 → date selected in the calendar
  - Types: Earnings, Dividends (for selected symbols), Economic (if plan allows), IPOs (US only)
  - Symbols scope for Earnings/Dividends:
    - Portfolio tickers (from local `portfolio.csv`)
    - DEFAULT_SYMBOLS from `.env`
    - Large “market movers”: AAPL, MSFT, GOOGL, AMZN, NVDA, META, TSLA, etc.
  - Interaction: double-click an Earnings event to open an Earnings Preview dialog

## Data Sources and Keys

- IBKR: host/port/clientId in `.env` (IBKR_HOST, IBKR_PORT, IBKR_CLIENT_ID, IBKR_ACCOUNT_CODE optional)
- FRED: `FRED_API_KEY` in `.env`
- Finnhub: `FINNHUB_API_KEY` in `.env`
- NewsAPI (optional sentiment): `NEWSAPI_KEY` in `.env`

## Update Logic

- Auto-Refresh: Dashboard uses a timer (`UI_UPDATE_INTERVAL`) for account and portfolio data.
- Manual Refresh: Top “Refresh” button triggers dashboard update; Calendar has its own Refresh.
- Macro refresh every ~30 minutes and on-demand.

## Short-Term Post-Earnings Model (Preview)

- Location: `src/services/earnings_ml.py`
- Inputs:
  - Recent price features (returns/volatility) fetched just-in-time from yfinance
  - Optional news sentiment (see below)
- Model: lightweight Logistic Regression trained ad-hoc on a small rolling window (demo quality)
- Output: probability of “up” over the next ~half day after the event
- Notes: recommended to replace with an offline-trained, curated model for production

## News Sentiment (Optional)

- Location: `src/services/sentiment_service.py`
- Provider: NewsAPI (Everything endpoint)
- Scoring: VADER if available, else TextBlob fallback; returns average compound score [-1..1]
- Usage: blended into the earnings preview probability (20% weight)

## Files and Modules

- UI: `src/ui/widgets/dashboard_widget.py`, `macro_widget.py`, `calendar_widget.py`
- Services: `src/services/ibkr_service.py`, `finnhub_client.py`, `earnings_ml.py`, `sentiment_service.py`
- Helpers: `src/utils/trading_helpers.py` (portfolio CSV), `src/core/config_manager.py` (env loading)

## Known Limitations / Next Steps

- IBKR connection requires valid API config and account permissions; account_code must be valid.
- Macro indicators limited to 3 for clarity; can add more series or a selector.
- Earnings ML is illustrative; consider replacing with a robust, offline-trained model.
- Economic calendar availability depends on Finnhub plan; fails gracefully if not permitted.

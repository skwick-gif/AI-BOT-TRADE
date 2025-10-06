# AI Trading — Screen Guide and Logic

This guide explains how the AI Trading screen works end-to-end: data flow, configuration, dependencies, and how to operate it safely in Paper and Live modes.

## Overview

The AI Trading tab lets you automate trade decisions based on numeric scores from Perplexity. On a configurable schedule, the app sends a compact prompt, gets back a single number (0–10), and decides BUY/SELL/HOLD using your strategy thresholds with hysteresis and cooldown rules. Execution is either simulated (Paper) or real (Live via IBKR).

High-level flow:

1) Timer fires per asset (e.g., every 1m)
2) AIService asks Perplexity for a numeric score (0–10)
3) The score is mapped to BUY/SELL/HOLD using strategy thresholds
4) Hysteresis and cooldown gate repeated actions
5) Guardrails apply (max trades/day; daily loss limit placeholder)
6) Execute trade (Paper: update portfolio.csv; Live: send IBKR order)
7) Log the decision and update the UI

## Key Dependencies

- Perplexity API: requires PERPLEXITY_API_KEY in .env
- IBKR (Live mode): requires working TWS/IB Gateway API (IBKR_HOST/PORT/CLIENT_ID/etc.)
- yfinance: used to fetch latest price for Paper trades
- Config files:
  - config/ai_trading.json stores global settings, strategies, and assets
  - portfolio.csv (root) records Paper trades

## UI Structure

- Top bar:
  - IBKR status (Connect/Disconnect)
  - Perplexity status + Test
  - Mode: Paper/Live
  - Global Interval (1m/2m/5m/15m/60m)
  - Trading hours only toggle
  - Settings (⚙) dialog
  - Start All / Stop All

- Tabs:
  - Assets: add symbols, choose profile, enable automation; see status and scores
  - Strategies: create/update profiles (thresholds, hysteresis, cooldown, SL/TP, interval)
  - Settings: global mode/interval/hours, guardrails, and Perplexity custom prompt
  - Logs: chronological execution log

## Strategies & Decision Logic

- Thresholds: BUY if score ≥ buy_threshold, SELL if score ≤ sell_threshold, else HOLD
- Hysteresis: require the same non-HOLD signal N cycles in a row before acting
- Cooldown: after an action, wait X minutes before acting again on the same symbol
- Guardrails:
  - Max trades/day: cap the number of actions per day (0 = unlimited)
  - Daily loss limit: placeholder until real PnL wiring; currently blocks after threshold

## Perplexity Prompts (Numeric-Only)

The system uses a compact, profile-aware prompt asking Perplexity to output only a number (0–10). Consistent language improves reliability.

- Built-in prompt: automatically includes profile focus and internal calibration with thresholds
- Custom prompt: in Settings, check “Use custom Perplexity prompt,” click “Insert template,” then edit as needed
- Placeholders: {symbol}, {profile}, {buy}, {sell} are replaced at runtime before sending

Recommended template:

```
Score the opportunity for '{symbol}' on a 0-10 scale for {profile} trading horizon.
Focus on price action, momentum, and near-term catalysts.
Calibrate internally with thresholds BUY≥{buy} and SELL≤{sell},
but output ONLY the number (0-10). No text, no code, no units.
```

Tips:
- Keep Temperature low for numeric-only reliability (the app uses a compact prompt and low max_tokens)
- If you get non-numeric text, ensure the prompt says “Output ONLY the number (0–10)”

## Execution Modes

- Paper:
  - Uses yfinance to fetch the latest price
  - Records trades in portfolio.csv (BUY/SELL, qty, price)
  - Safe for testing your strategies and prompts

- Live (IBKR):
  - Requires a successful connection via Connection > Connect to IBKR
  - Sends market orders via ib_insync (booktrade_ibkr)
  - Ensure TWS/IB Gateway API is enabled and not Read-Only; firewall allows the port

## Step-by-Step (Paper)

1) Add PERPLEXITY_API_KEY to .env and restart the app
2) Go to AI Trading > Settings, set Mode = Paper
3) (Optional) Enable “Use custom Perplexity prompt,” click “Insert template,” and Save
4) Add an Asset (symbol, qty, profile), then enable it (On)
5) Press “Start All” to schedule all enabled assets; watch scores and actions in Logs
6) Check portfolio.csv to see simulated trades

## Step-by-Step (Live)

1) Configure IBKR API in TWS/Gateway and .env (IBKR_HOST/PORT/CLIENT_ID/ACCOUNT_CODE)
2) In app: Connection > Connect to IBKR; confirm the status is Connected
3) Switch Mode = Live in Settings and Save
4) Enable your assets and press “Start All”
5) Watch Logs and your IBKR platform for order updates

## Files & Settings

- config/ai_trading.json (saved automatically):

```json
{
  "globals": {
    "trading_hours_only": true,
    "mode": "Paper",
    "interval": "1m",
    "daily_loss_limit": 0.0,
    "max_trades_day": 0,
    "use_custom_prompt": false,
    "custom_prompt": ""
  },
  "strategies": {
    "Intraday": { "buy_threshold": 8.0, "sell_threshold": 4.0, "hysteresis": 2, "cooldown_min": 3, "sl_pct": 3.0, "tp_pct": 6.0, "interval": "1m" }
  },
  "assets": [
    { "symbol": "AAPL", "quantity": 100, "strategy": "Intraday", "use_global_interval": true, "custom_interval": "1m", "enabled": false }
  ]
}
```

- portfolio.csv (root): appended on Paper trades: date, ticker, action, shares, price, total

## Troubleshooting

- Perplexity errors or non-numeric response:
  - Verify PERPLEXITY_API_KEY and model in .env
  - Use the provided template; keep “Output ONLY the number (0–10)”
  - Use the “Test Prompt Now” button to validate the exact response

- No Paper price available:
  - Markets closed or symbol not found in yfinance; try again during market hours

- IBKR not connecting:
  - Verify TWS/IB Gateway is running; API enabled; “Enable ActiveX and Socket Clients” checked
  - Trusted IPs include 127.0.0.1; firewall allows the selected port
  - Check ports: TWS (7496 live / 7497 paper), Gateway (4001 live / 4002 paper)

## Safety Notes

- This tool is for research and automation; it is not financial advice
- Always test thoroughly in Paper mode
- In Live mode, ensure you understand the risks and guardrails

---
If you need help tuning prompts or strategies, open an issue or ask via the AI Agent tab.

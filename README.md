# AI Trading Bot

An AI-powered trading application that connects to Interactive Brokers for automated trading analysis and execution.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Initial Setup](#initial-setup)
3. [Architecture](#architecture)
4. [Quick Start](#quick-start)
5. [Features](#features)
6. [Troubleshooting](#troubleshooting)
7. [Development](#development)
8. [Configuration](#configuration)
9. [Support](#support)

---

## Prerequisites

Before starting, ensure you have:

- **Windows 10/11** (or any Windows with PowerShell/CMD)
- **.NET 8 SDK** - [Download](https://dotnet.microsoft.com/download/dotnet/8.0)
  - Verify: `dotnet --version` (should show 8.x.x)
- **Python 3.10+** - [Download](https://www.python.org/downloads/)
  - Verify: `python --version` (should show 3.10+)
- **Git** - [Download](https://git-scm.com/)
  - Verify: `git --version`
- **Interactive Brokers Account** (Paper or Live)
  - Paper trading account: Free, for testing
  - Live account: For real trading
- **TWS or IB Gateway** - [Download](https://www.interactivebrokers.com/en/trading/platforms)

---

## Initial Setup

### 1. Clone the Repository

```bash
git clone https://github.com/skwick-gif/AI-BOT-TRADE.git "D:\AI-BOT-TRADE"
cd "D:\AI-BOT-TRADE"
```

### 2. Checkout the Correct Branch

```bash
git checkout backup-before-cleanup
```

### 3. Verify Project Structure

Ensure these files/folders exist:
- `start_bridge.bat` ← Bridge launcher
- `run_app.bat` ← UI launcher
- `tools/InterReactBridge/` ← .NET Bridge
- `src/` ← Python UI code
- `README.md` ← Main documentation

### 4. Create Python Virtual Environment

```bash
python -m venv .venv
```

### 5. Activate Virtual Environment

**Windows PowerShell:**
```powershell
.\.venv\Scripts\Activate.ps1
```

**Windows Command Prompt:**
```cmd
.venv\Scripts\activate.bat
```

**Expected output:** Your prompt should now show `(.venv)` at the beginning.

### 6. Upgrade Pip

```bash
python -m pip install --upgrade pip
```

### 7. Install Dependencies

```bash
pip install -r requirements.txt
```

This installs all Python packages needed for the UI and utilities.

### 8. Verify Installation

```bash
python -c "import PyQt6; import requests; print('Dependencies OK')"
```

### 9. IB Gateway / TWS Setup

You need **either** IB Gateway or TWS (Trader Workstation) running to trade.

#### Option A: IB Gateway (Recommended)

IB Gateway is lighter and better for automated trading.

1. Download from [Interactive Brokers](https://www.interactivebrokers.com/en/trading/platforms)
2. Start IB Gateway
3. Log in with your IBKR account credentials
4. Go to Settings > API > Settings
5. Enable "Enable ActiveX and Socket Clients"
6. Add `127.0.0.1` to Trusted IPs
7. Note the Socket Port (default: 4002 for Paper, 4001 for Live)

#### Option B: TWS (Trader Workstation)

TWS is the full trading platform.

1. Start TWS
2. Go to Global Configuration > API > Settings
3. Enable "Enable ActiveX and Socket Clients"
4. For Paper: Set Socket Port to 7497
5. For Live: Set Socket Port to 7496
6. Add `127.0.0.1` to Trusted IPs

---

## Architecture

### Three-Tier Design

**Tier 1: Python UI (PyQt6)**
- User interface for trading operations
- Real-time portfolio monitoring
- Market data visualization
- Order placement and management
- Location: `main.py`, `src/ui/`

**Tier 2: .NET Bridge (ASP.NET Core)**
- REST API server on `http://localhost:5080`
- Manages all IBKR connections
- Handles authentication and session management
- Provides WebSocket/SSE for real-time data
- Location: `tools/InterReactBridge/`

**Tier 3: Interactive Brokers**
- TWS (Trader Workstation) or IB Gateway
- Actual trading and market data
- Port: 4001 (Gateway Live), 4002 (Gateway Paper), 7496 (TWS Live), 7497 (TWS Paper)

### Data Flow

```
Python UI (Port varies)
    ↓ (HTTP/REST)
.NET Bridge (http://localhost:5080)
    ↓ (TWS API Protocol)
Interactive Brokers (Port 7496/4002/etc)
```

### Complete Folder Structure

```
📁 src/
├── 📁 main.py                          # Entry point
├── 📁 ui/
│   ├── 📁 windows/
│   │   ├── main_window.py              # MainWindow class
│   │   └── settings_window.py          # Settings dialog
│   ├── 📁 widgets/
│   │   ├── __init__.py
│   │   ├── chat_widget.py              # AI Chat functionality
│   │   ├── ai_trading_widget.py        # AI Trading control panel
│   │   ├── trading_widget.py           # Trading interface
│   │   ├── dashboard_widget.py         # Dashboard overview
│   │   ├── macro_widget.py             # Economic indicators
│   │   ├── ml_widget.py                # 🤖 ML Training & Models
│   │   ├── watchlist_widget.py         # 👁️ Stock Watchlist Management
│   │   ├── scanner_widget.py           # 🔍 Stock Scanner & Screener
│   │   └── charts_widget.py            # Chart components
│   ├── 📁 components/
│   │   ├── __init__.py
│   │   ├── navigation_bar.py           # Sidebar navigation
│   │   ├── status_bar.py               # Status indicators
│   │   ├── metric_cards.py             # Reusable metric displays
│   │   ├── message_bubble.py           # Chat message components
│   │   ├── table_components.py         # Enhanced table widgets
│   │   ├── ml_components.py            # 🧠 ML-specific UI components
│   │   ├── watchlist_components.py     # 📋 Watchlist items & cards
│   │   └── scanner_components.py       # 🔎 Scanner result components
│   └── 📁 dialogs/
│       ├── __init__.py
│       ├── trade_dialog.py             # Trade execution dialog
│       ├── settings_dialog.py          # App settings
│       ├── ml_training_dialog.py       # 🎯 ML Training configuration
│       ├── watchlist_dialog.py         # 📝 Add/Edit watchlist items
│       ├── scanner_config_dialog.py    # ⚙️ Scanner filter settings
│       └── about_dialog.py             # About dialog
├── 📁 core/
│   ├── __init__.py
│   ├── config.py                       # Configuration management
│   ├── signals.py                      # Custom PyQt signals
│   └── app_state.py                    # Application state management
├── 📁 services/
│   ├── __init__.py
│   ├── ibkr_service.py                 # IBKR API integration
│   ├── ai_service.py                   # AI APIs (OpenAI, Perplexity)
│   ├── market_data_service.py          # Market data fetching
│   ├── portfolio_service.py            # Portfolio calculations
│   ├── macro_data_service.py           # Economic data (FRED)
│   ├── ml_service.py                   # 🧠 ML Training & Prediction service
│   ├── watchlist_service.py            # 📋 Watchlist data management
│   └── scanner_service.py              # 🔍 Stock screening service
├── 📁 ml/                              # 🤖 Machine Learning Module
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── price_predictor.py          # Stock price prediction models
│   │   ├── trend_analyzer.py           # Market trend analysis
│   │   ├── risk_assessor.py            # Portfolio risk assessment
│   │   └── sentiment_analyzer.py       # News/social sentiment analysis
│   ├── trainers/
│   │   ├── __init__.py
│   │   ├── base_trainer.py             # Base training class
│   │   ├── lstm_trainer.py             # LSTM model trainer
│   │   ├── rf_trainer.py               # Random Forest trainer
│   │   └── ensemble_trainer.py         # Ensemble methods
│   ├── data/
│   │   ├── __init__.py
│   │   ├── data_loader.py              # Data loading utilities
│   │   ├── preprocessor.py             # Data preprocessing
│   │   └── feature_engineer.py         # Feature engineering
│   └── evaluation/
│       ├── __init__.py
│       ├── metrics.py                  # ML evaluation metrics
│       └── backtester.py               # Strategy backtesting
├── 📁 scanner/                         # 🔍 Stock Scanner Module
│   ├── __init__.py
│   ├── filters/
│   │   ├── __init__.py
│   │   ├── technical_filters.py        # Technical analysis filters
│   │   ├── fundamental_filters.py      # Fundamental analysis filters
│   │   ├── price_filters.py            # Price/volume filters
│   │   └── custom_filters.py           # User-defined filters
│   ├── screeners/
│   │   ├── __init__.py
│   │   ├── momentum_screener.py        # Momentum stocks
│   │   ├── value_screener.py           # Value stocks
│   │   ├── breakout_screener.py        # Breakout patterns
│   │   └── earnings_screener.py        # Earnings-based screening
│   └── strategies/
│       ├── __init__.py
│       ├── preset_strategies.py        # Pre-built screening strategies
│       └── strategy_builder.py         # Custom strategy creation
├── 📁 watchlist/                       # 👁️ Watchlist Module
│   ├── __init__.py
│   ├── managers/
│   │   ├── __init__.py
│   │   ├── watchlist_manager.py        # Watchlist CRUD operations
│   │   ├── alert_manager.py            # Price/news alerts
│   │   └── sync_manager.py             # Cloud sync for watchlists
│   ├── analyzers/
│   │   ├── __init__.py
│   │   ├── performance_analyzer.py     # Watchlist performance tracking
│   │   ├── correlation_analyzer.py     # Inter-stock correlations
│   │   └── sector_analyzer.py          # Sector distribution analysis
│   └── exporters/
│       ├── __init__.py
│       ├── csv_exporter.py             # CSV export functionality
│       └── pdf_reporter.py             # PDF report generation
├── 📁 models/
│   ├── __init__.py
│   ├── portfolio.py                    # Portfolio data models
│   ├── trade.py                        # Trade data models
│   ├── market_data.py                  # Market data structures
│   ├── chat_message.py                 # Chat message models
│   ├── ml_model.py                     # 🧠 ML model metadata
│   ├── watchlist_item.py               # 📋 Watchlist item model
│   └── scanner_result.py               # 🔍 Scanner result model
├── 📁 utils/
│   ├── __init__.py
│   ├── formatters.py                   # Data formatting utilities
│   ├── validators.py                   # Input validation
│   ├── calculators.py                  # Financial calculations
│   ├── threading.py                    # Threading utilities
│   ├── ml_utils.py                     # 🤖 ML helper functions
│   ├── scanner_utils.py                # 🔍 Scanner utilities
│   └── watchlist_utils.py              # 📋 Watchlist utilities
└── 📁 data/
    ├── __init__.py
    ├── cache_manager.py                # Data caching
    ├── data_store.py                   # Local data storage
    ├── ml_data/                        # 🧠 ML training data storage
    │   ├── models/                     # Saved ML models
    │   ├── datasets/                   # Training datasets
    │   └── features/                   # Feature data
    ├── watchlists/                     # 📋 Saved watchlists
    │   ├── personal/                   # User watchlists
    │   └── presets/                    # Preset watchlists
    └── scanner/                        # 🔍 Scanner data
        ├── results/                    # Scan results history
        └── strategies/                 # Saved scan strategies

📁 assets/
├── 📁 styles/
│   ├── dark_theme.qss                  # Main stylesheet
│   ├── components.qss                  # Component-specific styles
│   ├── ml_theme.qss                    # 🧠 ML-specific styling
│   ├── watchlist_theme.qss             # 📋 Watchlist styling
│   ├── scanner_theme.qss               # 🔍 Scanner styling
│   └── animations.qss                  # Animation definitions
├── 📁 icons/
│   ├── app_icon.png
│   ├── navigation/
│   │   ├── ml_icon.svg                 # 🧠 ML tab icon
│   │   ├── watchlist_icon.svg          # 📋 Watchlist tab icon
│   │   └── scanner_icon.svg            # 🔍 Scanner tab icon
│   └── status/
└── 📁 fonts/
    └── custom_fonts.ttf

📁 tests/
├── 📁 unit/
│   ├── test_services/
│   ├── test_models/
│   ├── test_ml/
│   ├── test_scanner/
│   ├── test_watchlist/
│   └── test_utils/
└── 📁 integration/
    ├── test_ibkr_integration.py
    ├── test_ml_integration.py
    └── test_ui_workflow.py

📁 docs/
├── API.md                              # API documentation
├── SETUP.md                            # Setup instructions
├── ARCHITECTURE.md                     # System architecture
├── ML_GUIDE.md                         # ML module guide
├── SCANNER_GUIDE.md                    # Scanner module guide
└── WATCHLIST_GUIDE.md                  # Watchlist module guide

📁 legacy-streamlit/                    # Previous version (backup)
```

### Bridge API Endpoints

The Bridge exposes a REST API for all IBKR operations:

#### Health & Status
- `GET /health` - Check if bridge is running
- `GET /connect/status` - Get current connection status

#### Connection Management
- `POST /connect?host=127.0.0.1&port=7496&clientId=101` - Connect to IBKR
- `POST /connect/retry` - Retry connection

#### Data Retrieval
- `GET /account` - Get account summary
- `GET /portfolio` - Get portfolio positions
- `GET /marketdata?symbol=AAPL&secType=STK&exchange=SMART&durationSeconds=5` - Get one-shot market data
- `GET /livedata?symbol=AAPL&secType=STK&exchange=SMART` - Stream live market data (SSE)
- `GET /optionschain?underlying=AAPL&exchange=SMART` - Get options chain

#### Order Management
- `POST /orders/bracket` - Place bracket order (entry + stop loss + take profit)
- `POST /orders/oco` - Place OCO order (One-Cancels-Other)
- `POST /orders/combo` - Place combo order

---

## Quick Start

### Step 1: Start the Bridge (Terminal 1)

The Bridge is a .NET service that handles all communication with Interactive Brokers. It runs independently and stays in the background.

```bash
start_bridge.bat
```

You'll be prompted:

```
========================================
 Select IBKR Connection Target
========================================

 1. IB Gateway (Live)   - Port 4001
 2. IB Gateway (Paper)  - Port 4002
 3. TWS (Paper)         - Port 7497
 4. TWS (Live)          - Port 7496
 5. Custom Port

Enter your choice (1-5):
```

**Choose based on what you're running:**
- Running IB Gateway Paper? → Type `2`
- Running TWS Live? → Type `4`
- Running IB Gateway Live? → Type `1`
- Custom setup? → Type `5` and enter details

**Expected output after selection:**

```
========================================
 Building InterReactBridge...
========================================
Build succeeded in X.Xs

========================================
 Launching InterReactBridge...
========================================
Bridge will run at: http://localhost:5080

Now listening on: http://localhost:5080
Application started. Press Ctrl+C to shut down.
```

**⚠️ Important:** Keep this terminal **open**. The bridge must remain running.

### Step 2: Start the UI (Terminal 2)

In a **separate terminal**, run the Python UI:

```bash
run_app.bat
```

This launches the PyQt6 application. It will:
1. Activate the Python virtual environment
2. Start the UI
3. Connect to the Bridge on `http://localhost:5080`
4. Fetch account and portfolio data
5. Display live market data (if available)

### Step 3: Verify Connection

In the UI:

1. Look for connection status (usually in the bottom-left or title bar)
2. Go to **Connection** menu
3. Click **Sync From Bridge Status** if available
4. You should see:
   - ✓ Connected to IBKR
   - Account information loaded
   - Portfolio positions visible

### Basic Workflow

1. **Dashboard**: Check account status and market overview
2. **Portfolio**: Monitor existing positions and performance
3. **AI Agent**: Ask questions about your portfolio or market
4. **Watchlist**: Add symbols you want to monitor
5. **Scanner**: Find new trading opportunities
6. **ML Training**: Build predictive models

---

## Features

### 📊 Dashboard
- **Account Overview**: Net liquidation, buying power, day P&L
- **Portfolio Summary**: Position count, diversity score, largest position
- **Market Overview**: Real-time market indices (S&P 500, NASDAQ, DOW, VIX)
- **Recent Activity**: Latest trading activity and events

### 💼 Portfolio
- **Position Management**: Real-time position tracking with live P&L
- **Portfolio Metrics**: Total value, day change, total return
- **Interactive Table**: Sort, filter, and analyze positions
- **Position Details**: Detailed view for selected positions
- **Quick Actions**: Right-click context menu for trading

### 🤖 AI Agent
- **Intelligent Chat**: Natural language trading assistant
- **Message Bubbles**: Professional chat interface with user/AI distinction
- **Quick Actions**: Pre-defined prompts for common queries
- **Multi-threaded**: Non-blocking AI processing
- **Smart Responses**: Context-aware trading advice

### 🧠 ML Training
- **Model Configuration**: Choose from multiple ML algorithms (Random Forest, XGBoost, Neural Network, etc.)
- **Data Management**: Load data from IBKR, Yahoo Finance, or custom CSV
- **Feature Engineering**: Technical indicators, volume data, sentiment analysis
- **Training Pipeline**: Real-time progress tracking and performance metrics
- **Preset Configurations**: Quick start templates for different strategies

### 👁️ Watchlist
- **Real-time Monitoring**: Live price updates for selected symbols
- **Smart Table**: Sortable columns with color-coded changes
- **Symbol Management**: Add/remove symbols with autocomplete
- **Context Menu**: Right-click for quick trading actions
- **Details Panel**: Selected symbol analysis

### 🔍 Scanner
- **Advanced Filtering**: Price, volume, technical, and fundamental criteria
- **Preset Strategies**: Growth stocks, value stocks, momentum, oversold
- **Real-time Scanning**: Progress tracking and live results
- **Results Analysis**: Sortable table with composite scoring
- **Quick Integration**: Add scan results directly to watchlist

### 🤖 ML (Machine Learning) Tab Features:
- **Data Loading**: טעינת נתונים היסטוריים לאימון
- **Model Training**: אימון מודלים שונים (LSTM, Random Forest, Ensemble)
- **Feature Engineering**: יצירת פיצ'רים טכניים ויסודיים
- **Backtesting**: בדיקת ביצועים היסטוריים
- **Real-time Predictions**: תחזיות בזמן אמת
- **Model Comparison**: השוואת מודלים שונים
- **Training Progress**: מעקב התקדמות אימון
- **Model Export/Import**: שמירה וטעינה של מודלים מאומנים

### 👁️ WATCHLIST Tab Features:
- **Multi-Watchlists**: ניהול רשימות מעקב מרובות
- **Real-time Updates**: עדכונים בזמן אמת של מחירים
- **Alerts System**: התראות על מחירים/חדשות
- **Performance Tracking**: מעקב ביצועים
- **Sector Analysis**: ניתוח חלוקה סקטוריאלית
- **Export/Import**: יצוא/יבוא רשימות
- **Notes & Tags**: הערות ותגיות למניות
- **Correlation Analysis**: ניתוח קורלציות בין מניות

### 🔍 SCANNER Tab Features:
- **Technical Screening**: סריקה לפי אינדיקטורים טכניים
- **Fundamental Screening**: סריקה לפי נתונים יסודיים
- **Custom Filters**: מסנני צפיה מותאמים אישית
- **Preset Strategies**: אסטרטגיות סריקה מוכנות מראש
- **Real-time Scanning**: סריקה בזמן אמת
- **Results History**: היסטוריית תוצאות סריקה
- **Filter Builder**: בונה מסננים חזותי
- **Export Results**: יצוא תוצאות סריקה

---

## Troubleshooting

### Bridge fails to start with "Port already in use"

**Symptom:** `IOException: Failed to bind to address http://127.0.0.1:5080`

**Solution:**
1. Another process is using port 5080
2. Kill existing processes: `Get-Process dotnet | Stop-Process -Force`
3. Or change the port in `tools/InterReactBridge/Properties/launchSettings.json`

### Bridge cannot connect to IBKR

**Symptom:** Bridge starts but never connects

**Solution:**
1. Ensure TWS/Gateway is running on the port you selected
2. Go to TWS/Gateway settings > API > Settings:
   - Enable "Enable ActiveX and Socket Clients"
   - Add `127.0.0.1` to Trusted IPs
   - Note the socket port (usually 7496 for TWS Live, 4002 for Gateway Paper)
3. If TWS shows "Accept incoming connection" popup, click "Yes"
4. Restart the bridge

### Python UI cannot connect to Bridge

**Symptom:** UI starts but no data appears

**Solution:**
1. Verify bridge is running: Check for "Now listening on: http://localhost:5080" in bridge window
2. Test manually: `curl http://localhost:5080/health`
3. Check firewall: Ensure port 5080 is not blocked
4. Restart both bridge and UI

### UI Won't Start

**Error:** `Virtual environment not found`

**Solution:**
```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**Error:** `Module not found (PyQt6, requests, etc.)`

**Solution:**
```bash
.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

### IBKR Connection Issues
- Ensure TWS/Gateway is running
- Check API settings are enabled
- Verify correct port (4001 for IB Gateway, 7497 for TWS paper)
- Try different client ID if conflicts occur

### Performance Optimization
- Close unused tabs when not needed
- Limit watchlist to actively monitored symbols
- Use shorter update intervals only when necessary
- Clear old scan results periodically

### Data Issues
- Restart application if data stops updating
- Check internet connection for market data
- Verify IBKR subscription for required data
- Contact broker for data feed issues

---

## Development

### Typical Day

1. **Start Bridge** (once at the beginning of the day):
   ```bash
   start_bridge.bat
   # Select your environment and let it run
   ```

2. **Start UI** (as many times as you need):
   ```bash
   run_app.bat
   # Make changes to UI code
   # Close and restart when ready to test changes
   ```

3. **Stop Bridge** (end of day):
   - Press `Ctrl+C` in the bridge terminal

### Making Changes

- **UI changes**: Modify files in `src/ui/` or `main.py`, then restart the UI
- **Bridge changes**: Modify files in `tools/InterReactBridge/`, then restart the bridge (it will rebuild automatically)
- **Python dependencies**: Update `requirements.txt`, then reinstall with `pip install -r requirements.txt` and restart UI

### Development Roadmap

#### Phase 1: Core Infrastructure
1. Create folder structure
2. Set up base classes and interfaces
3. Implement configuration system
4. Create main window skeleton

#### Phase 2: Basic Modules
1. **Dashboard Widget**: Overview and metrics
2. **Portfolio Widget**: Basic portfolio tracking
3. **Chat Widget**: AI assistant functionality

#### Phase 3: Trading Features
1. **Trading Widget**: Order execution interface
2. **IBKR Service**: Complete broker integration
3. **Market Data Service**: Real-time data feeds

#### Phase 4: Analysis Modules
1. **WATCHLIST**: Stock monitoring and alerts
2. **SCANNER**: Stock screening and filtering
3. **Macro Widget**: Economic indicators

#### Phase 5: Advanced Features
1. **ML Module**: Machine learning and predictions
2. **Advanced Charts**: Technical analysis charts
3. **Backtesting**: Strategy testing

#### Phase 6: Polish & Optimization
1. Performance optimization
2. UI/UX improvements
3. Testing and bug fixes
4. Documentation completion

---

## Configuration

### Environment Variables

Create a `.env` file in the project root if you need custom settings:

```
# IBKR Connection (used by Bridge)
IBKR_HOST=127.0.0.1
IBKR_PORT=7496

# Bridge Server
ASPNETCORE_URLS=http://localhost:5080

# Python UI
UI_THEME=dark
```

### Key Technologies

- **PyQt6**: Modern GUI framework
- **pandas/numpy**: Data manipulation
- **scikit-learn/tensorflow**: Machine learning
- **TA-Lib/pandas-ta**: Technical analysis
- **ib_insync**: Interactive Brokers integration
- **asyncio**: Asynchronous programming

### Design Patterns

- **MVC Pattern**: Model-View-Controller separation
- **Observer Pattern**: Signal/slot communication
- **Strategy Pattern**: For ML models and scanning strategies
- **Factory Pattern**: For creating different types of components
- **Singleton Pattern**: For configuration and app state

---

## Support & Documentation

- **Setup Guide**: See `IBKR-SETUP-RUNBOOK.md` for detailed Hebrew documentation
- **Architecture Overview**: See `ARCHITECTURE.md`
- **Quick Start**: See `QUICKSTART.md`
- **Issues**: Check troubleshooting section above
- **API Reference**: Bridge API endpoints listed in "Bridge API Endpoints" section

---

## License

[Add your license here if applicable]

# PyQt6 Trading Bot - Quick Start Guide

## 🎯 Overview

You now have a complete professional trading application with the following tabs:

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

## 🚀 Getting Started

### 1. Initial Setup
```bash
# Install dependencies
pip install -r requirements-pyqt6.txt

# Start the application
python main.py
```

### 2. Configure IBKR Connection
1. Start TWS or IB Gateway
2. Enable API connections in TWS/Gateway settings
3. Use port 4001 for IB Gateway, 7497 for TWS paper trading
4. Click "Connect IBKR" in the application

### 3. Basic Workflow
1. **Dashboard**: Check account status and market overview
2. **Portfolio**: Monitor existing positions and performance
3. **AI Agent**: Ask questions about your portfolio or market
4. **Watchlist**: Add symbols you want to monitor
5. **Scanner**: Find new trading opportunities
6. **ML Training**: Build predictive models

## 💡 Tips & Best Practices

### AI Agent Usage
- Ask specific questions: "Analyze my AAPL position"
- Request market updates: "What's happening in tech stocks today?"
- Get trading ideas: "What are good momentum plays right now?"
- Risk assessment: "How risky is my current portfolio?"

### ML Training
- Start with simple models (Random Forest) before complex ones
- Use technical indicators for short-term predictions
- Include volume data for better accuracy
- Monitor validation metrics to avoid overfitting

### Watchlist Management
- Use for stocks you're considering trading
- Monitor daily movers and volume spikes
- Right-click for quick trading actions
- Check details panel for additional analysis

### Scanner Usage
- Use presets as starting points
- Combine technical and fundamental filters
- Sort results by composite score
- Add interesting finds to watchlist

## 🔧 Troubleshooting

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

## 📈 Advanced Features

### Custom ML Models
```python
# Configuration example for custom models
config = {
    'model_type': 'Neural Network',
    'epochs': 100,
    'learning_rate': 0.001,
    'features': {
        'technical_indicators': True,
        'volume_data': True,
        'sentiment_data': False
    }
}
```

### Scanner Strategies
- **Growth**: High revenue growth, above moving averages
- **Value**: Low P/E ratios, strong fundamentals
- **Momentum**: Strong price movement with volume
- **Oversold**: Low RSI, recent price decline

### AI Prompts
- "Compare my tech holdings performance"
- "What sectors are outperforming today?"
- "Suggest diversification for my portfolio"
- "Analyze risk-reward for potential trade"

## 🎨 Customization

### Themes
- Toggle between dark and light themes
- Consistent styling across all components
- Professional color schemes
- High contrast for readability

### Layout
- Resizable splitters between sections
- Sortable table columns
- Adjustable update intervals
- Configurable watchlist symbols

## 📊 Key Metrics

### Portfolio Metrics
- **Net Liquidation**: Total account value
- **Buying Power**: Available funds for trading
- **Day P&L**: Today's profit/loss
- **Unrealized P&L**: Open position gains/losses

### ML Metrics
- **Accuracy**: Correct predictions percentage
- **Precision**: True positive rate
- **Recall**: Sensitivity to positive cases
- **F1 Score**: Harmonic mean of precision and recall

### Scanner Scoring
- **Price Momentum**: Recent price movement
- **Volume Strength**: Trading volume analysis
- **Technical Score**: Indicator-based rating
- **Composite Score**: Overall ranking (1-10)

---

**Ready to trade smarter with AI-powered tools! 🚀**

For detailed documentation, see `ARCHITECTURE.md` and the full `README-PyQt6.md`.
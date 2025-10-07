# AI Trading Bot

An AI-powered trading application that connects to Interactive Brokers for automated trading analysis and execution.

## Features

- Real-time portfolio monitoring
- AI-powered market analysis
- Interactive Brokers integration
- Paper and live trading support
- Risk management tools
- **Non-blocking UI**: IBKR connections run in background threads to keep the interface responsive

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd AI-BOT-TRADE

# Install dependencies
install_dependencies.bat
```

### 2. Configure IB Gateway/TWS

**Option A: IB Gateway (Recommended)**
1. Download and install IB Gateway
2. Start IB Gateway
3. Configure API settings:
   - Enable API connections
   - Set port to 4001 (default)
   - Add 127.0.0.1 to trusted IPs
4. Accept connection when prompted

**Option B: TWS (Trader Workstation)**
1. Start TWS
2. Go to Global Configuration > API > Settings
3. Enable "Enable ActiveX and Socket Clients"
4. Set Socket Port to 7497 (paper) or 7496 (live)
5. Add 127.0.0.1 to trusted IPs

### 3. Run the Application

```bash
python main.py
```

### 4. Connect to IBKR

- Click "Connect IBKR" button in the application
- The app will automatically try ports in this order:
  1. 4001 (IB Gateway)
  2. 7496 (TWS Live)
  3. 7497 (TWS Paper)

## Configuration

### Environment Variables

Copy `.env.example` to `.env` and configure:

```bash
# IBKR Configuration
IBKR_HOST=127.0.0.1
IBKR_PORT=4001  # IB Gateway (4001), TWS Paper (7497), TWS Live (7496)
IBKR_CLIENT_ID=1

# API Keys
PERPLEXITY_API_KEY=your_key_here
# ... other API keys
```

### Port Configuration

- **4001**: IB Gateway (default)
- **7497**: TWS Paper Trading
- **7496**: TWS Live Trading

## Testing Connection

Test your IB Gateway connection:

```bash
test_ibgw.bat
```

Or manually:

```bash
python tools/test_ibgw_connection.py
```

## Troubleshooting

### IBKR Connection Issues

1. **Connection Refused**
   - Ensure IB Gateway/TWS is running
   - Check that API is enabled in settings
   - Verify correct port is configured

2. **Connection Timeout**
   - Accept any connection dialogs in IB Gateway/TWS
   - Check firewall settings
   - Verify trusted IP configuration

3. **Authentication Errors**
   - Use a unique client ID (1-999)
   - Ensure account is logged in to IB Gateway/TWS

### Performance Tips

- Use IB Gateway instead of full TWS for better performance
- Close unused application tabs when not needed
- Limit watchlist to actively monitored symbols
- **UI Responsiveness**: All IBKR connections run in background threads - the interface stays responsive during connection attempts

## Architecture

- `src/core/`: Core configuration and services
- `src/services/`: IBKR and AI service integrations
- `src/ui/`: PyQt6 user interface components
- `src/utils/`: Trading helpers and utilities
- `tools/`: Testing and debugging scripts

## Support

For issues and questions, check the troubleshooting section or refer to the QUICKSTART.md guide.
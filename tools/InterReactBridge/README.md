# InterReactBridge REST Server

This .NET 8 minimal API bridges Interactive Brokers (TWS/IB Gateway) to a simple REST interface consumed by the Python app.

- URL: honors ASPNETCORE_URLS or IBKR_BRIDGE_URL; defaults to http://localhost:8080
- Endpoints:
  - POST /connect?host=127.0.0.1&port=4001&clientId=1
  - GET /account
  - GET /portfolio
  - GET /marketdata?symbol=AAPL&secType=STK&exchange=SMART&durationSeconds=5
  - POST /order?symbol=AAPL&secType=STK&exchange=SMART&action=BUY&quantity=1&price=150&orderType=LMT
  - GET /optionschain?underlying=AAPL&exchange=SMART
  - GET /scan?scanType=TOP_PERC_GAIN&numberOfRows=10
  - GET /livedata?symbol=AAPL&secType=STK&exchange=SMART (Server-Sent Events)

## Prerequisites
- .NET SDK 8.0+
- TWS or IB Gateway running locally with API enabled

## Run
- PowerShell:
  - In this folder, run: `dotnet restore; dotnet build -v minimal; $env:ASPNETCORE_URLS='http://localhost:5080'; dotnet run`
  - Or specify directly: `dotnet run --urls http://localhost:8080`
- Batch script:
  - `scripts\build_and_run.bat` (uses IBKR_BRIDGE_URL if set; defaults to http://localhost:5080)

## Python wiring
Set in environment or .env:
- `IBKR_BRIDGE_URL=http://localhost:8080` (or whatever port you choose)
- `IBKR_HOST=127.0.0.1`
- `IBKR_PORT=4001` (IB Gateway) or `7497` (TWS Paper)
- `IBKR_CLIENT_ID=1`

Then in the app UI, use Connect to IBKR. The Python side uses `IBKRAdapterService` which calls this REST server.
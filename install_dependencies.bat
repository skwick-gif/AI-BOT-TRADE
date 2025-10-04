@echo off
echo ========================================
echo Installing AI Trading Bot Dependencies
echo ========================================

REM Check if Python is installed
py --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH!
    echo Please install Python 3.12+ and add it to your PATH
    pause
    exit /b 1
)

echo Python found. Creating virtual environment...

REM Create virtual environment if it doesn't exist
if not exist ".venv" (
    echo Creating virtual environment...
    python -m venv .venv
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment!
        pause
        exit /b 1
    )
    echo Virtual environment created successfully.
) else (
    echo Virtual environment already exists.
)

echo Activating virtual environment...
call .venv\Scripts\activate.bat

echo Upgrading pip...
py -m pip install --upgrade pip

echo Installing core dependencies...
pip install PyQt6>=6.5.0
pip install PyQt6-tools
pip install ib-insync>=0.9.86
pip install pandas>=2.0.0
pip install numpy>=1.24.0
pip install requests>=2.31.0
pip install python-dotenv>=1.0.0
pip install aiohttp>=3.8.0
pip install nest-asyncio>=1.5.0

echo Installing ML dependencies...
pip install scikit-learn>=1.3.0
pip install xgboost>=1.7.0
pip install lightgbm>=4.0.0
pip install joblib>=1.3.0

echo Installing technical analysis...
pip install TA-Lib
if errorlevel 1 (
    echo WARNING: TA-Lib installation failed. Trying alternative...
    pip install talib-binary
)

pip install pandas-ta>=0.3.14b0
if errorlevel 1 (
    echo Installing stable version of pandas-ta...
    pip install pandas-ta
)

echo Installing additional utilities...
pip install plotly>=5.15.0
pip install yfinance>=0.2.0

echo Creating necessary directories...
if not exist "models" mkdir models
if not exist "cache" mkdir cache
if not exist "logs" mkdir logs
if not exist "data" mkdir data

echo ========================================
echo Installation Complete!
echo ========================================
echo.
echo Next steps:
echo 1. Create a .env file with your API keys
echo 2. Run 'run_app.bat' to start the application
echo.
echo Required .env variables:
echo PERPLEXITY_API_KEY=your_key_here
echo IBKR_HOST=127.0.0.1
echo IBKR_PORT=7497
echo IBKR_CLIENT_ID=1
echo.
pause
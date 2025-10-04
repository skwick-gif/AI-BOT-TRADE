@echo off
echo ========================================
echo Installing AI Trading Bot Dependencies
echo ========================================

REM Check if Python launcher is available
py --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH!
    echo Please install Python 3.12+ (via Microsoft Store or python.org) and ensure 'py' launcher is available
    pause
    exit /b 1
)

echo Python found. Creating virtual environment...

REM Create virtual environment if it doesn't exist
if not exist ".venv" (
    echo Creating virtual environment...
    py -m venv .venv
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment!
        echo Tip: If multiple Python versions are installed, run 'py -m venv .venv' manually.
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
python -m pip install --upgrade pip

echo Installing dependencies from requirements files...
if exist requirements.txt (
    python -m pip install -r requirements.txt
) else (
    echo WARNING: requirements.txt not found; installing a minimal core set...
)
if exist requirements-pyqt6.txt (
    python -m pip install -r requirements-pyqt6.txt
) else (
    echo WARNING: requirements-pyqt6.txt not found; ensuring PyQt6 is installed...
    python -m pip install PyQt6
)

echo Ensuring additional core packages are present...
python -m pip install ib_insync pandas numpy requests python-dotenv aiohttp nest-asyncio yfinance pandas_ta

echo Installing ML packages (optional)...
python -m pip install scikit-learn xgboost lightgbm joblib

echo Installing TA-Lib (optional)...
python -m pip install TA-Lib
if errorlevel 1 (
    echo WARNING: TA-Lib native build may fail on Windows; trying prebuilt binary wheel...
    python -m pip install talib-binary
)

echo Installing visualization utilities...
python -m pip install plotly

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
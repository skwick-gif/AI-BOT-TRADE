@echo off
echo ========================================
echo   AI Trading Bot - PyQt6 Application
echo ========================================

REM Check if virtual environment exists
if not exist ".venv" (
    echo ERROR: Virtual environment not found!
    echo Please run 'install_dependencies.bat' first to set up the environment.
    pause
    exit /b 1
)

REM Check if main.py exists
if not exist "main.py" (
    echo ERROR: main.py not found!
    echo Please ensure you're in the correct directory with the PyQt6 application.
    pause
    exit /b 1
)

echo Activating virtual environment...
call .venv\Scripts\activate.bat

echo Checking Python version...
.venv\Scripts\python.exe --version

echo Starting AI Trading Bot...
echo.
echo ========================================
echo Application is starting...
echo ========================================
echo.

REM Run the PyQt6 application
.venv\Scripts\python.exe main.py

REM Check if the application exited with an error
if errorlevel 1 (
    echo.
    echo ========================================
    echo Application exited with an error!
    echo ========================================
    echo.
    echo Showing last 40 lines from today's log:
    for /f "delims=" %%x in ('powershell -NoProfile -Command "Get-Content -Path 'logs\trading_bot_%date:~10,4%%date:~4,2%%date:~7,2%.log' -Tail 40 -ErrorAction SilentlyContinue"') do echo %%x
    echo.
    echo Common issues:
    echo 1. Missing .env file with API keys
    echo 2. IBKR TWS/Gateway not running
    echo 3. Missing dependencies
    echo.
    echo Run 'install_dependencies.bat' if you have missing packages.
    pause
) else (
    echo.
    echo ========================================
    echo Application closed successfully.
    echo ========================================
)

pause
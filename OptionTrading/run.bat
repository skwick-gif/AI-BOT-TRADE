@echo off
echo Starting Options Trading Application...
echo.

REM Check if Python is installed (try both python and py commands)
python --version >nul 2>&1
if errorlevel 1 (
    echo Python not found, trying py command...
    py --version >nul 2>&1
    if errorlevel 1 (
        echo ERROR: Python is not installed or not in PATH.
        echo Please install Python from https://www.python.org/downloads/
        echo Make sure to check "Add Python to PATH" during installation.
        pause
        exit /b 1
    ) else (
        set PYTHON_CMD=py
    )
) else (
    set PYTHON_CMD=python
)

REM Check if .env file exists and has API key
if not exist ".env" (
    echo ERROR: .env file not found.
    echo Please run install.bat first or create .env file manually.
    pause
    exit /b 1
)

REM Check if API key is set
findstr /C:"PERPLEXITY_API_KEY=your_perplexity_api_key_here" ".env" >nul
if not errorlevel 1 (
    echo ERROR: Perplexity API key not configured in .env file.
    echo Please edit .env and add your actual API key.
    pause
    exit /b 1
)

REM Check if dependencies are installed
%PYTHON_CMD% -c "import PyQt6" >nul 2>&1
if errorlevel 1 (
    echo ERROR: PyQt6 not found. Please run install.bat first.
    pause
    exit /b 1
)

echo Launching application...
echo.

REM Run the application
%PYTHON_CMD% src/main.py

echo.
echo Application closed.
pause
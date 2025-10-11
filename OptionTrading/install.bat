@echo off
echo Installing Options Trading dependencies...
echo.

REM Check if Python is installed
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
)

echo Python found. Installing dependencies...
echo.

REM Install requirements
%PYTHON_CMD% -m pip install -r requirements.txt

if errorlevel 1 (
    echo ERROR: Failed to install dependencies.
    echo Please check your internet connection and try again.
    pause
    exit /b 1
)

REM Create .env file if it doesn't exist
if not exist ".env" (
    if exist ".env.example" (
        copy ".env.example" ".env"
        echo.
        echo Created .env file from .env.example
        echo IMPORTANT: Edit .env file and add your Perplexity API key!
    ) else (
        echo.
        echo WARNING: .env.example not found. Please create .env file manually.
    )
)

echo.
echo Dependencies installed successfully!
echo.
echo Next steps:
echo 1. Run setup.bat to configure your API keys
echo 2. Run run.bat to start the application
echo.
pause
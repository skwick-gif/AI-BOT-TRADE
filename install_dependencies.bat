@echo off
echo ========================================
echo Installing AI Trading Bot Dependencies
echo ========================================

REM Check if Python is available (try venv first, then system python)
if exist ".venv\Scripts\python.exe" (
    set PYTHON_EXE=.venv\Scripts\python.exe
    echo Using virtual environment Python...
) else (
    python --version >nul 2>&1
    if errorlevel 1 (
        echo ERROR: Python is not installed or not in PATH!
        echo Please install Python 3.12+ and ensure it's in PATH
        pause
        exit /b 1
    )
    set PYTHON_EXE=python
)

echo Python found. Creating virtual environment...

REM Create virtual environment if it doesn't exist
if not exist ".venv" (
    echo Creating virtual environment...
    python -m venv .venv
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment!
        echo Tip: If multiple Python versions are installed, run 'python -m venv .venv' manually.
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
%PYTHON_EXE% -m pip install --upgrade pip setuptools wheel

echo Installing base dependencies...
if exist requirements.base.txt (
    %PYTHON_EXE% -m pip install -r requirements.base.txt
    if errorlevel 1 (
        echo ERROR: Failed installing base requirements!
        echo Tip: Ensure you have Microsoft C++ Build Tools installed for packages requiring wheels.
        echo       https://visualstudio.microsoft.com/visual-cpp-build-tools/
        pause
        exit /b 1
    )
) else (
    echo Base file not found, falling back to requirements.txt
    if exist requirements.txt (
        %PYTHON_EXE% -m pip install -r requirements.txt
        if errorlevel 1 (
            echo ERROR: Failed installing from requirements.txt!
            pause
            exit /b 1
        )
    ) else (
        echo ERROR: No requirements file found!
        pause
        exit /b 1
    )
)

echo Ensuring all packages are properly installed...
%PYTHON_EXE% -m pip install --upgrade pip setuptools wheel

echo Verifying critical packages...
%PYTHON_EXE% -c "import PyQt6; print('PyQt6 OK')"
if errorlevel 1 (
    echo PyQt6 install appears incomplete. Retrying explicit install...
    %PYTHON_EXE% -m pip install --force-reinstall --upgrade PyQt6==6.7.1 PyQt6-Qt6==6.7.3
)

REM Common Windows build pain points (lxml)
%PYTHON_EXE% -c "import lxml"
if errorlevel 1 (
    echo lxml not importable. Installing prebuilt wheel if available...
    %PYTHON_EXE% -m pip install --only-binary=:all: lxml || %PYTHON_EXE% -m pip install lxml
)

echo Creating necessary directories...
if not exist "models" mkdir models
if not exist "cache" mkdir cache
if not exist "logs" mkdir logs
if not exist "data" mkdir data

echo.
echo Optional installations (can be run later):
echo   1^)^) ML extras:    .venv\Scripts\pip.exe install -r requirements.extras-ml.txt
echo   2^)^) Dev extras:   .venv\Scripts\pip.exe install -r requirements.extras-dev.txt

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
echo IBKR_PORT=4001
echo IBKR_CLIENT_ID=1
echo.
pause
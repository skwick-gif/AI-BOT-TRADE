@echo off
echo ========================================
echo AI Trading Bot - Run Diagnostics
echo ========================================

echo Current directory: %cd%

echo --- Checking virtual environment (.venv) ---
if exist ".venv\Scripts\python.exe" (
    echo .venv Python: .venv\Scripts\python.exe FOUND
    .venv\Scripts\python.exe --version 2>&1
    .venv\Scripts\python.exe -m pip --version 2>&1
) else (
    echo .venv\Scripts\python.exe NOT FOUND
)

echo.
echo --- 'py' launcher ---
py --version 2>&1 || echo py not available

echo.
echo --- 'where python' ---
where python 2>nul || echo 'python' not found in PATH

echo.
echo --- List .venv\Scripts directory ---
if exist ".venv\Scripts" (
    dir ".venv\Scripts" /b
) else (
    echo .venv\Scripts directory not present
)

echo.
echo --- Check run_app.bat presence ---
if exist "run_app.bat" (echo run_app.bat exists) else (echo run_app.bat NOT found)

echo.
echo --- Tail last 40 lines of latest trading log (if present) ---
powershell -NoProfile -Command "Get-ChildItem -Path .\\logs -Filter 'trading_bot_*.log' | Sort-Object LastWriteTime -Descending | Select-Object -First 1 | ForEach-Object { Get-Content -Path $_.FullName -Tail 40 }" 2>&1

echo.
echo --- Attempting to run the application (captures output to diag_output.txt) ---
if exist ".venv\Scripts\python.exe" (
    echo Running: .venv\Scripts\python.exe main.py > diag_output.txt 2>&1
    .venv\Scripts\python.exe main.py > diag_output.txt 2>&1
    echo Application exit code: %ERRORLEVEL%
    echo ---- Begin diag_output.txt ----
    type diag_output.txt
    echo ---- End diag_output.txt ----
) else (
    echo Skipping run: .venv\Scripts\python.exe not found
)

echo.
echo Diagnostics complete. If the run failed, please attach 'diag_output.txt' and the log lines above.
pause

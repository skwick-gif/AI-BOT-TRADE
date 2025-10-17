@echo off
setlocal

:: ============================================================================
:: InterReactBridge Launcher
::
:: This script builds and launches the .NET InterReactBridge, connects it to
:: Interactive Brokers, and keeps it running in the background.
:: ============================================================================

set BRIDGE_PROJECT_PATH=tools\InterReactBridge\InterReactBridge.csproj
set BRIDGE_URL=http://localhost:5080
set CLIENT_ID=101

:: Allow passing choice as first argument
if not "%1"=="" (
    set CHOICE=%1
    goto PROCESS_CHOICE
)

:MENU
cls
echo.
echo ========================================
echo  Select IBKR Connection Target
echo ========================================
echo.
echo  1. IB Gateway (Live)   - Port 4001
echo  2. IB Gateway (Paper)  - Port 4002
echo  3. TWS (Paper)         - Port 7497
echo  4. TWS (Live)          - Port 7496
echo  5. Custom Port
echo.
set /p "CHOICE=Enter your choice (1-5): "

:PROCESS_CHOICE
if "%CHOICE%"=="1" set PORT=4001& goto BUILD
if "%CHOICE%"=="2" set PORT=4002& goto BUILD
if "%CHOICE%"=="3" set PORT=7497& goto BUILD
if "%CHOICE%"=="4" set PORT=7496& goto BUILD
if "%CHOICE%"=="5" goto CUSTOM_PORT
echo Invalid choice. Please try again.
timeout /t 2 >nul
goto MENU

:CUSTOM_PORT
echo.
set /p "PORT=Enter Port Number: "
set /p "CLIENT_ID=Enter Client ID (default is 101): "
if not defined CLIENT_ID set CLIENT_ID=101
goto BUILD

:BUILD
echo.
echo ========================================
echo  Building InterReactBridge...
echo ========================================
dotnet build "%~dp0tools\InterReactBridge\InterReactBridge.csproj" -c Release
if %errorlevel% neq 0 (
    echo Build failed.
    pause
    exit /b 1
)

:LAUNCH
echo.
echo ========================================
echo  Launching InterReactBridge...
echo ========================================
echo Bridge will run at: %BRIDGE_URL%
echo.
echo Press Ctrl+C to stop the bridge.
echo.

:: Run in a separate window so the bridge stays running
start "InterReactBridge" cmd /k "cd /d %~dp0 && dotnet run --project "%BRIDGE_PROJECT_PATH%" --no-build --urls %BRIDGE_URL%"

echo Waiting for bridge to start...
timeout /t 3 /nobreak >nul

echo.
echo ========================================
echo  Verifying Bridge Health...
echo ========================================
powershell -Command "$i=0; while($i -lt 5) { try { $r = Invoke-WebRequest -Uri %BRIDGE_URL%/health -UseBasicParsing -ErrorAction Stop; Write-Host 'Bridge is healthy! (Status: ' $r.StatusCode ')'; break } catch { $i++; if($i -lt 5) { Start-Sleep -Milliseconds 500 } } }; if($i -eq 5) { Write-Error 'Bridge is not responding' }"

echo.
echo ========================================
echo  Attempting to Connect to IBKR...
echo ========================================
echo Connecting to host 127.0.0.1 on Port %PORT% with Client ID %CLIENT_ID%...
powershell -Command "try { $r = Invoke-RestMethod -Method Post -Uri '%BRIDGE_URL%/connect?host=127.0.0.1&port=%PORT%&clientId=%CLIENT_ID%' -ErrorAction Stop; Write-Host 'Connection attempt sent.' } catch { Write-Error $_ }"

echo.
echo ===================================================================================
echo  Bridge is running and attempting to connect.
echo  You can now run 'run_app.bat' in a separate terminal to start the UI.
echo ===================================================================================
echo.
pause
endlocal

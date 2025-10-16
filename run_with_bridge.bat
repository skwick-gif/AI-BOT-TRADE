@echo off
setlocal EnableDelayedExpansion
echo ========================================
echo   Start InterReactBridge + PyQt App
echo ========================================

setlocal
set IBKR_HOST=127.0.0.1
set IBKR_PORT=4002
set IBKR_CLIENT_ID=1

REM If IBKR_BRIDGE_URL already provided, use it as-is
if not "%IBKR_BRIDGE_URL%"=="" goto HAS_URL

REM Auto-detect existing bridge on 8080; if healthy, reuse it
for /f "usebackq delims=" %%A in (`powershell -NoProfile -Command "try { (Invoke-RestMethod -TimeoutSec 2 http://localhost:8080/health) | ConvertTo-Json -Compress } catch { '' }"`) do set HEALTH8080=%%A
if not "%HEALTH8080%"=="" (
	echo Detected existing InterReactBridge on http://localhost:8080
		set IBKR_BRIDGE_URL=http://localhost:8080
		goto START_APP
)

REM Otherwise default to 5080 and start a new bridge
set IBKR_BRIDGE_URL=http://localhost:5080
echo Starting InterReactBridge on %IBKR_BRIDGE_URL% ...
pushd tools\InterReactBridge
start "InterReactBridge" dotnet run --no-build --urls %IBKR_BRIDGE_URL%
popd
goto START_APP

:HAS_URL
echo Using provided IBKR_BRIDGE_URL=%IBKR_BRIDGE_URL%

:START_APP
timeout /t 2 /nobreak >nul
call run_app.bat

endlocal

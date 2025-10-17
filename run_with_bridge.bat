@echo off
REM ============================================================================
REM DEPRECATED: This script is no longer used.
REM
REM Please use the new separate scripts instead:
REM   1. start_bridge.bat  (Terminal 1 - runs the .NET Bridge)
REM   2. run_app.bat       (Terminal 2 - runs the Python UI)
REM
REM See README.md for detailed instructions.
REM ============================================================================
echo.
echo *** DEPRECATED SCRIPT ***
echo This script is no longer used. Please run:
echo   - start_bridge.bat (Terminal 1)
echo   - run_app.bat      (Terminal 2)
echo.
pause
exit /b 1


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

@echo on
setlocal EnableDelayedExpansion
echo ========================================
echo   DEBUG: Start InterReactBridge + PyQt App
echo ========================================

setlocal
if not exist logs mkdir logs

REM Control whether to open a live log tail window for the bridge (1=on, 0=off)
if "%BRIDGE_TAIL%"=="" set BRIDGE_TAIL=1

REM Configure IB (Paper defaults)
if "%IBKR_HOST%"=="" set IBKR_HOST=127.0.0.1
if "%IBKR_PORT%"=="" set IBKR_PORT=4002
if "%IBKR_CLIENT_ID%"=="" set IBKR_CLIENT_ID=1
set IBKR_AUTO_CONNECT=1

REM Detect existing bridge on 8080
for /f "usebackq delims=" %%A in (`powershell -NoProfile -Command "try { (Invoke-RestMethod -TimeoutSec 2 http://localhost:8080/health) | ConvertTo-Json -Compress } catch { '' }"`) do set HEALTH8080=%%A
if not "%HEALTH8080%"=="" (
  echo Detected existing InterReactBridge on http://localhost:8080
  set IBKR_BRIDGE_URL=http://localhost:8080
)
if "%IBKR_BRIDGE_URL%"=="" set IBKR_BRIDGE_URL=http://localhost:5080
echo Launching InterReactBridge at %IBKR_BRIDGE_URL%
pushd tools\InterReactBridge
if exist bridge_run.log del /q bridge_run.log
set ASPNETCORE_URLS=%IBKR_BRIDGE_URL%
REM Ensure launch profiles are not used and force binding via --urls
set DOTNET_LAUNCH_PROFILE=
REM Build Release and prefer published EXE
dotnet publish -c Release -r win-x64 -p:PublishSingleFile=true -p:PublishTrimmed=false --self-contained true > bridge_build.log 2>&1
set BRIDGE_EXE=bin\Release\net8.0\win-x64\publish\InterReactBridge.exe
if exist "%BRIDGE_EXE%" (
  echo Running published bridge EXE: %BRIDGE_EXE%
  start "InterReactBridge-Debug" cmd /c "set ASPNETCORE_URLS=%ASPNETCORE_URLS% && set DOTNET_LAUNCH_PROFILE= && set IBKR_HOST=%IBKR_HOST% && set IBKR_PORT=%IBKR_PORT% && set IBKR_CLIENT_ID=%IBKR_CLIENT_ID% && set IBKR_AUTO_CONNECT=%IBKR_AUTO_CONNECT% && "%BRIDGE_EXE%" --urls %IBKR_BRIDGE_URL% > bridge_run.log 2>&1"
) else (
  echo Published EXE not found, falling back to dotnet run
  start "InterReactBridge-Debug" cmd /c "set ASPNETCORE_URLS=%ASPNETCORE_URLS% && set DOTNET_LAUNCH_PROFILE= && set IBKR_HOST=%IBKR_HOST% && set IBKR_PORT=%IBKR_PORT% && set IBKR_CLIENT_ID=%IBKR_CLIENT_ID% && set IBKR_AUTO_CONNECT=%IBKR_AUTO_CONNECT% && dotnet run -c Release --no-launch-profile --urls %IBKR_BRIDGE_URL% > bridge_run.log 2>&1"
)
popd

REM Optionally open a live tail of the bridge logs in a separate PowerShell window
if "%BRIDGE_TAIL%"=="1" (
  start "InterReactBridge-Logs" powershell -NoProfile -Command "Write-Host 'Tailing tools/InterReactBridge/bridge_run.log'; Get-Content -Path 'tools/InterReactBridge/bridge_run.log' -Tail 120 -Wait -ErrorAction SilentlyContinue"
)

echo Waiting 3 seconds for bridge...
timeout /t 3 /nobreak >nul

echo Bridge health check at %IBKR_BRIDGE_URL%/health
for /L %%i in (1,1,5) do (
  powershell -NoProfile -Command "try { (Invoke-RestMethod -TimeoutSec 2 '%IBKR_BRIDGE_URL%/health') | ConvertTo-Json -Compress } catch { 'RETRY' }" | find /I "status"
  if not errorlevel 1 goto BRIDGE_OK
  timeout /t 1 /nobreak >nul
)
echo BRIDGE did not respond in time.
:BRIDGE_OK

REM Activate venv and run app with captured output
if not exist .venv\Scripts\python.exe (
  echo ERROR: Missing virtualenv. Run install_dependencies.bat first.
  goto :EOF
)

call .venv\Scripts\activate.bat
echo Starting app (logs\app_run.log)...
if exist logs\app_run.log del /q logs\app_run.log
.venv\Scripts\python.exe main.py > logs\app_run.log 2>&1

echo App exited with code %errorlevel%
echo --- Tail app_run.log ---
powershell -NoProfile -Command "Get-Content -Path 'logs/app_run.log' -Tail 80 -ErrorAction SilentlyContinue"

echo --- Tail bridge_run.log (if any) ---
if exist tools\InterReactBridge\bridge_run.log (
  powershell -NoProfile -Command "Get-Content -Path 'tools/InterReactBridge/bridge_run.log' -Tail 120 -ErrorAction SilentlyContinue"
)

endlocal
@echo off
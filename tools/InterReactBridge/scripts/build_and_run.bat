@echo off
REM Build and run the InterReactBridge; honor IBKR_BRIDGE_URL if set, default to http://localhost:5080
setlocal
cd /d "%~dp0.."

set "BRIDGE_URL=%IBKR_BRIDGE_URL%"
if "%BRIDGE_URL%"=="" set "BRIDGE_URL=http://localhost:5080"

echo Using URL: %BRIDGE_URL%

echo Restoring packages...
dotnet restore
if errorlevel 1 (
  echo dotnet restore failed
  exit /b 1
)

echo Building project...
dotnet build -v minimal
if errorlevel 1 (
  echo dotnet build failed
  exit /b 1
)

echo Starting InterReactBridge on %BRIDGE_URL%
dotnet run --urls %BRIDGE_URL%

endlocal

@echo off
cd /d D:\Projects\AI-BOT-TRADE\tools\InterReactBridge
echo Cleaning previous builds...
dotnet clean > nul 2>&1
echo Building fresh...
dotnet build --no-incremental
echo.
echo Starting server...
echo.
dotnet run --no-build
pause

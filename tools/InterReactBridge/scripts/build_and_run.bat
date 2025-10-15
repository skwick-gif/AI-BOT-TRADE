@echo off
REM Build and run the original InterReactBridge on http://localhost:5000
setlocal
cd /d "%~dp0.."
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

echo Starting InterReactBridge on http://localhost:5000
dotnet run --urls http://localhost:5000

endlocal
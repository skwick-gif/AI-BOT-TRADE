@echo off
echo Options Trading Setup Helper
echo ============================
echo.

REM Check if .env exists
if not exist ".env" (
    echo Creating .env file from template...
    if exist ".env.example" (
        copy ".env.example" ".env"
        echo .env file created successfully.
    ) else (
        echo ERROR: .env.example not found. Cannot create .env file.
        pause
        exit /b 1
    )
) else (
    echo .env file already exists.
)

echo.
echo Current .env configuration:
echo ---------------------------
type .env
echo.

echo.
echo Current .env configuration:
echo ---------------------------
type .env
echo.

set /p api_key="Enter your Perplexity API key (or press Enter to skip): "

if defined api_key (
    REM Update the API key in .env file
    powershell -Command "(Get-Content .env) -replace 'PERPLEXITY_API_KEY=.*', 'PERPLEXITY_API_KEY=%api_key%' | Set-Content .env"
    echo.
    echo API key updated successfully!
    echo The application will use Perplexity Finance model for analysis.
) else (
    echo.
    echo No API key entered. You can edit .env manually later.
)

echo.
echo Setup complete! You can now run the application with run.bat
echo.
echo Remember to:
echo - Make sure Trader Workstation (TWS) is running for IBKR connection
echo - Test the application with run.bat
echo.
pause
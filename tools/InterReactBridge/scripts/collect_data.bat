@echo off
REM Collect data from a running InterReactBridge instance at http://localhost:5000
setlocal
set BASE_URL=http://localhost:5000

echo Calling /connect to request connection to IBKR (TWS/Gateway)
powershell -Command "Invoke-RestMethod -Uri '%BASE_URL%/connect?host=127.0.0.1&port=7497&clientId=1' -Method Post | ConvertTo-Json | Out-File connect_response.json -Encoding utf8"

echo Waiting 10 seconds for IBKR responses...
timeout /t 10 /nobreak >nul

echo Fetching portfolio
powershell -Command "Invoke-RestMethod -Uri '%BASE_URL%/portfolio' -Method Get | ConvertTo-Json | Out-File portfolio.json -Encoding utf8"

echo Fetching account summary
powershell -Command "Invoke-RestMethod -Uri '%BASE_URL%/account' -Method Get | ConvertTo-Json | Out-File account.json -Encoding utf8"

echo Saved: connect_response.json, portfolio.json, account.json
endlocal
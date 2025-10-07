@echo off
echo Testing IB Gateway connection on port 4001...
echo.
echo Make sure:
echo 1. IB Gateway is running
echo 2. API connections are enabled in IB Gateway settings
echo 3. Port is set to 4001 in IB Gateway configuration
echo 4. Accept any connection confirmation dialogs
echo.
pause
python tools\test_ibgw_connection.py
pause
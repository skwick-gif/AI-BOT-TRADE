# run_bridge.ps1 - Persistent Bridge Runner
# This script runs the InterReactBridge persistently, connects to IBKR, and monitors the connection.

param(
    [string]$IbHost = "127.0.0.1",
    [int]$IbPort = 7497,
    [int]$ClientId = 101,
    [string]$BridgeUrl = "http://localhost:5000"
)

$ErrorActionPreference = "Stop"

# Function to check if process is running
function Test-ProcessRunning {
    param([string]$ProcessName)
    return Get-Process -Name $ProcessName -ErrorAction SilentlyContinue
}

# Function to check bridge health
function Test-BridgeHealth {
    try {
        $response = Invoke-WebRequest -Uri "${BridgeUrl}/health" -TimeoutSec 5 -ErrorAction Stop
        return $response.StatusCode -eq 200
    } catch {
        return $false
    }
}

# Function to connect to IBKR
function Connect-Ibkr {
    try {
        $body = @{
            host = $IbHost
            port = $IbPort
            clientId = $ClientId
        } | ConvertTo-Json

        $response = Invoke-WebRequest -Method Post -Uri "${BridgeUrl}/connect" -Body $body -ContentType "application/json" -TimeoutSec 10 -ErrorAction Stop
        return $response.StatusCode -eq 200
    } catch {
        Write-Host "Connection failed: $_"
        return $false
    }
}

# Main script
Write-Host "Starting InterReactBridge persistently..."
Write-Host "IBKR Connection: ${IbHost}:${IbPort}, ClientId: ${ClientId}"
Write-Host "Bridge URL: ${BridgeUrl}"
Write-Host "Press Ctrl+C to stop."

# Build the project
# No build needed for Python Flask bridge

# Start the bridge in background
$pythonPath = "$PSScriptRoot\.venv\Scripts\python.exe"
$bridgeJob = Start-Job -ScriptBlock {
    param($pythonPath)
    & $pythonPath tools/flask_bridge.py
} -ArgumentList $pythonPath

Start-Sleep -Seconds 5

# Monitor loop
$connected = $false
while ($true) {
    if (-not (Test-BridgeHealth)) {
        Write-Host "$(Get-Date): Bridge not healthy, restarting..."
        $bridgeJob | Stop-Job -ErrorAction SilentlyContinue
        $bridgeJob | Remove-Job -ErrorAction SilentlyContinue

        $bridgeJob = Start-Job -ScriptBlock {
            param($BridgeUrl)
            dotnet run --project tools\InterReactBridge\InterReactBridge.csproj --no-build --urls $BridgeUrl
        } -ArgumentList $BridgeUrl

        Start-Sleep -Seconds 5
        $connected = $false
    }

    if (-not $connected -and (Test-BridgeHealth)) {
        Write-Host "$(Get-Date): Connecting to IBKR..."
        if (Connect-Ibkr) {
            Write-Host "$(Get-Date): Connected to IBKR successfully"
            $connected = $true
        } else {
            Write-Host "$(Get-Date): Failed to connect to IBKR"
        }
    }

    Start-Sleep -Seconds 60  # Check every minute
}

# Cleanup on exit
$bridgeJob | Stop-Job -ErrorAction SilentlyContinue
$bridgeJob | Remove-Job -ErrorAction SilentlyContinue
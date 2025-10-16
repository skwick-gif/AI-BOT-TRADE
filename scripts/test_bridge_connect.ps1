# Usage:
#   powershell -ExecutionPolicy Bypass -File scripts/test_bridge_connect.ps1 -Host 127.0.0.1 -Port 4002 -ClientId 101 -BaseUrl http://localhost:5080

param(
    [string]$ApiHost = "127.0.0.1",
    [int]$Port = 4002,
    [int]$ClientId = 101,
    [string]$BaseUrl = "http://localhost:5080"
)

Write-Host "Bridge Health:" -ForegroundColor Cyan
try {
    $health = Invoke-WebRequest -UseBasicParsing -Uri "$BaseUrl/health" -TimeoutSec 5
    Write-Host $health.Content
} catch {
    Write-Warning $_.Exception.Message
}

Write-Host ("`nProbe {0}:{1}:" -f $ApiHost, $Port) -ForegroundColor Cyan
try {
    $probe = Invoke-WebRequest -UseBasicParsing -Uri "$BaseUrl/probe?host=$ApiHost&port=$Port&timeoutMs=800" -TimeoutSec 5
    Write-Host $probe.Content
} catch {
    Write-Warning $_.Exception.Message
}

Write-Host "\nConnect attempt:" -ForegroundColor Cyan
try {
    $connect = Invoke-WebRequest -UseBasicParsing -Method Post -Uri "$BaseUrl/connect?host=$ApiHost&port=$Port&clientId=$ClientId" -TimeoutSec 25
    Write-Host $connect.Content
} catch {
    Write-Warning $_.Exception.Message
}

Start-Sleep -Seconds 1

Write-Host "\nStatus:" -ForegroundColor Cyan
try {
    $status = Invoke-WebRequest -UseBasicParsing -Uri "$BaseUrl/connect/status" -TimeoutSec 5
    Write-Host $status.Content
} catch {
    Write-Warning $_.Exception.Message
}

Write-Host "\nDiagnostics:" -ForegroundColor Cyan
try {
    $diag = Invoke-WebRequest -UseBasicParsing -Uri "$BaseUrl/diagnostics" -TimeoutSec 5
    Write-Host $diag.Content
} catch {
    Write-Warning $_.Exception.Message
}

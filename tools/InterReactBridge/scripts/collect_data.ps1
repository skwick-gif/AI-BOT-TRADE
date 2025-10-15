param()
$base = 'http://localhost:5000'
Write-Output "Calling /connect to request connection to IBKR"
Invoke-RestMethod -Uri "$base/connect?host=127.0.0.1&port=7497&clientId=1" -Method Post | ConvertTo-Json | Out-File -FilePath connect_response.json -Encoding utf8
Start-Sleep -Seconds 10
Write-Output "Fetching portfolio"
Invoke-RestMethod -Uri "$base/portfolio" -Method Get | ConvertTo-Json | Out-File -FilePath portfolio.json -Encoding utf8
Write-Output "Fetching account summary"
Invoke-RestMethod -Uri "$base/account" -Method Get | ConvertTo-Json | Out-File -FilePath account.json -Encoding utf8
Write-Output "Saved: connect_response.json, portfolio.json, account.json"
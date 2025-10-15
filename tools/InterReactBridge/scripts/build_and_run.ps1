param()
Write-Output "Restoring packages..."
Push-Location -Path (Join-Path $PSScriptRoot "..")
dotnet restore
if ($LASTEXITCODE -ne 0) { Write-Error "dotnet restore failed"; exit 1 }
dotnet build -v minimal
if ($LASTEXITCODE -ne 0) { Write-Error "dotnet build failed"; exit 1 }
Write-Output "Starting InterReactBridge on http://localhost:5000"
dotnet run --urls http://localhost:5000
Pop-Location
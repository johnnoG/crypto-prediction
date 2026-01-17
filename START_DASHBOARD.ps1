# START_DASHBOARD.ps1
# Simple script to start the Crypto Dashboard

Write-Host "🚀 Starting Crypto Dashboard..." -ForegroundColor Cyan
Write-Host ""

# Check if virtual environment exists
if (Test-Path ".venv\Scripts\python.exe") {
    Write-Host "✅ Virtual environment found" -ForegroundColor Green
    $pythonCmd = ".\.venv\Scripts\python.exe"
} else {
    Write-Host "⚠️  Virtual environment not found, using system Python" -ForegroundColor Yellow
    $pythonCmd = "python"
}

# Start the dashboard
Write-Host "Starting main launcher..." -ForegroundColor Cyan
& $pythonCmd main.py

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "❌ Dashboard failed to start. Check the error messages above." -ForegroundColor Red
    Write-Host ""
    Write-Host "Common fixes:" -ForegroundColor Yellow
    Write-Host "1. Make sure virtual environment is set up: py -3 -m venv .venv" -ForegroundColor White
    Write-Host "2. Install dependencies: .\.venv\Scripts\Activate.ps1; pip install -r requirements.txt" -ForegroundColor White
    Write-Host "3. Install frontend dependencies: cd frontend; npm install" -ForegroundColor White
    Write-Host "4. Check if ports 8000 and 5173 are available" -ForegroundColor White
    Write-Host ""
    exit 1
}

Write-Host ""
Write-Host "✅ Dashboard started successfully!" -ForegroundColor Green

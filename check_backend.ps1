# Backend Status Checker
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Backend Status Check" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if port 8000 is in use
$portCheck = netstat -ano | Select-String ":8000.*LISTENING"
if ($portCheck) {
    Write-Host "✅ Port 8000 is LISTENING" -ForegroundColor Green
    Write-Host $portCheck
} else {
    Write-Host "❌ Port 8000 is NOT listening" -ForegroundColor Red
}

Write-Host ""

# Test health endpoint
Write-Host "Testing /health/quick endpoint..." -ForegroundColor Yellow
try {
    $response = Invoke-RestMethod -Uri "http://127.0.0.1:8000/health/quick" -TimeoutSec 5 -ErrorAction Stop
    Write-Host "✅ Backend is RUNNING and responding!" -ForegroundColor Green
    Write-Host "Response: $($response | ConvertTo-Json)" -ForegroundColor Green
    Write-Host ""
    Write-Host "🎉 Your frontend should work now!" -ForegroundColor Cyan
} catch {
    Write-Host "❌ Backend is NOT responding" -ForegroundColor Red
    Write-Host "Error: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host ""
    Write-Host "To start the backend, run:" -ForegroundColor Yellow
    Write-Host "  .\start_backend.ps1" -ForegroundColor White
    Write-Host ""
    Write-Host "Or manually:" -ForegroundColor Yellow
    Write-Host "  cd backend\app" -ForegroundColor White
    Write-Host "  python -m uvicorn main:app --host 127.0.0.1 --port 8000" -ForegroundColor White
}

Write-Host ""

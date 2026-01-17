# Start Frontend Server for Crypto Dashboard
# Simple PowerShell script to start the Vite dev server

Write-Host "Starting Crypto Dashboard Frontend..." -ForegroundColor Cyan
Write-Host "=======================================" -ForegroundColor Cyan

# Navigate to frontend directory
Set-Location -Path "frontend"

# Start Vite dev server
Write-Host "`nStarting Vite dev server on http://127.0.0.1:5173" -ForegroundColor Green
Write-Host "Press Ctrl+C to stop the server`n" -ForegroundColor Yellow

npm run dev


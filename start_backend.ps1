# Start Backend Server for Crypto Dashboard
# Simple PowerShell script to start the FastAPI backend

Write-Host "Starting Crypto Dashboard Backend..." -ForegroundColor Cyan
Write-Host "======================================" -ForegroundColor Cyan

# Activate virtual environment
& ".\.venv\Scripts\Activate.ps1"

# Navigate to backend directory
Set-Location -Path "backend\app"

# Start uvicorn server
Write-Host "`nStarting FastAPI server on http://127.0.0.1:8000" -ForegroundColor Green
Write-Host "Press Ctrl+C to stop the server`n" -ForegroundColor Yellow

uvicorn main:app --host 127.0.0.1 --port 8000 --reload


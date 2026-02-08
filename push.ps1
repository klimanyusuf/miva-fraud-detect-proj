# Git push script
Push-Location 'c:\Users\Kyle\Desktop\miva-fraud-detect-proj'

Write-Host "Checking git status..." -ForegroundColor Cyan
git status

Write-Host "`nConfiguring git user..." -ForegroundColor Cyan
git config user.email "support@fraud-detect.local"
git config user.name "Fraud Detection Bot"

Write-Host "`nPushing to GitHub..." -ForegroundColor Cyan
git push origin main -f

Write-Host "`nDone!" -ForegroundColor Green
Pop-Location

@echo off
cd c:\Users\Kyle\Desktop\miva-fraud-detect-proj

echo Checking git status...
git status

echo.
echo Adding changes...
git add app.py requirements.txt setup.sh

echo.
echo Committing changes...
git commit -m "Update: Fix emoji display issue in Streamlit app and resolve merge conflicts"

echo.
echo Pushing to GitHub...
git push origin main

echo.
echo Done!
pause

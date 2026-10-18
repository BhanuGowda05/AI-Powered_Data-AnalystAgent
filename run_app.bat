@echo off
title Data Analyst Agent
echo.
echo  =============================================
echo    Data Analyst Agent - Starting...
echo  =============================================
echo.

:: Always run from the folder where this .bat file sits
cd /d "%~dp0"

:: Use "python -m streamlit" — most reliable on Windows, no PATH issues
:: "%~dp0app.py" = absolute path to app.py next to this bat file
python -m streamlit run "%~dp0app.py"
pause

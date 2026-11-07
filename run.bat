@echo off
cd /d "%~dp0"
"%~dp0venv\Scripts\python.exe" -u "%~dp0app.py" %*
pause

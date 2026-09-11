@echo off
setlocal
cd /d "%~dp0"
if not exist venv (
  echo Creating Python virtual environment...
  python -m venv venv
)
call venv\Scripts\activate.bat
python -m pip install -r requirements.txt
python app.py
pause

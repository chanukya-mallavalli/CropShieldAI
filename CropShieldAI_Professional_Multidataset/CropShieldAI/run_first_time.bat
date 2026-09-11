@echo off
setlocal
cd /d "%~dp0"
if not exist venv (
  echo Creating virtual environment...
  python -m venv venv
)
call venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
if not exist models\disease_ensemble.joblib (
  echo.
  echo ===== First-time ML setup =====
  python setup_ml.py
) else (
  echo Disease model already exists. Skipping training.
)
echo.
echo Starting CropShield AI...
python app.py
pause

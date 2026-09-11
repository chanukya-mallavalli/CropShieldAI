@echo off
setlocal
call "%~dp0venv\Scripts\activate.bat"
python "%~dp0training\train_disease_ensemble.py" --per-class 200
pause

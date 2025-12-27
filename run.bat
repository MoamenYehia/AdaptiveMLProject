@echo off
REM Batch script to run the Streamlit application
REM Windows users: Double-click this file or run it from command prompt

echo.
echo ====================================
echo Adaptive Fraud Detection System
echo ====================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://www.python.org/
    pause
    exit /b 1
)

REM Check if virtual environment exists
if not exist "venv" (
    echo Virtual environment not found. Creating one...
    python -m venv venv
    call venv\Scripts\activate.bat
    echo Installing dependencies...
    pip install -r requirements.txt
) else (
    call venv\Scripts\activate.bat
)

echo.
echo Starting Streamlit application...
echo Opening in browser at http://localhost:8501
echo.

streamlit run src\main.py

pause

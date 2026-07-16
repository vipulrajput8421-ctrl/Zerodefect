@echo off
REM ZeroDefect — Dashboard Launcher (Windows)
REM Starts the FastAPI dashboard server and opens your browser.
REM ============================================================

echo.
echo  ====================================================
echo   ZeroDefect - Starting Dashboard
echo   Open: http://localhost:8000
echo  ====================================================
echo.

IF EXIST "zerodefect_env\Scripts\activate.bat" (
    call zerodefect_env\Scripts\activate.bat
)

REM Wait a moment then open browser
start "" /min cmd /c "timeout /t 2 /noisy >nul && start http://localhost:8000"

echo [INFO] Starting FastAPI server on http://localhost:8000
echo [INFO] Press Ctrl+C to stop the server.
echo.

python -m uvicorn dashboard.app:app --host 0.0.0.0 --port 8000 --reload

pause

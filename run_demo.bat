@echo off
REM ZeroDefect — Live Demo Launcher (Windows)
REM Double-click this file to start the live webcam demo.
REM ============================================================

echo.
echo  ██████████████████████████████████████████████
echo  ██                                          ██
echo  ██      ZERODEFECT — Live Demo              ██
echo  ██      InnoVent 2026-27                    ██
echo  ██                                          ██
echo  ██████████████████████████████████████████████
echo.

REM Activate virtual environment if it exists
IF EXIST "zerodefect_env\Scripts\activate.bat" (
    echo [INFO] Activating virtual environment...
    call zerodefect_env\Scripts\activate.bat
) ELSE (
    echo [WARN] Virtual environment not found at zerodefect_env\
    echo [WARN] Running with system Python. Consider running:
    echo        python -m venv zerodefect_env
    echo        zerodefect_env\Scripts\activate
    echo        pip install -r requirements.txt
    echo.
)

REM Check if model files exist
IF NOT EXIST "models\memory_bank.pkl" (
    echo [ERROR] models\memory_bank.pkl not found!
    echo         Run training first: python src/train_model.py
    echo.
    pause
    exit /b 1
)

echo [INFO] Starting live demo. Press Q in the webcam window to quit.
echo [INFO] Logs saved to: logs\inspections.csv
echo.

python src/live_demo.py %*

echo.
echo [INFO] Demo ended. Run "python src/view_log.py" to see your audit log.
pause

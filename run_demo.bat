@echo off
REM ZeroDefect — Live Demo Launcher (Windows)
REM Double-click this file to start the live webcam demo.
REM ============================================================

echo.
echo  ██████████████████████████████████████████████
echo  ██                                          ██
echo  ██      ZERODEFECT — Live Demo              ██
echo  ██      YOLOv5n ONNX Detection              ██
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

REM Check if YOLO model exists
IF NOT EXIST "models\best.onnx" (
    echo [ERROR] models\best.onnx not found!
    echo         Download: hf sync hf://buckets/prath0029/Zerodefect-1.0-bucket ./local
    echo         Then:     copy local\best.onnx models\best.onnx
    echo.
    pause
    exit /b 1
)

echo [INFO] Starting live YOLOv5n detection demo. Press Q in the webcam window to quit.
echo [INFO] Model: models\best.onnx (7 defect classes)
echo [INFO] Logs saved to: logs\inspections.csv
echo.

python src/live_demo.py %*

echo.
echo [INFO] Demo ended. Run "python src/view_log.py" to see your audit log.
pause

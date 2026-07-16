@echo off
REM ZeroDefect — Data Capture Launcher (Windows)
REM ============================================================

echo.
echo  ██████████████████████████████████████████████
echo  ██                                          ██
echo  ██      ZERODEFECT — Data Capture           ██
echo  ██                                          ██
echo  ██████████████████████████████████████████████
echo.

IF EXIST "zerodefect_env\Scripts\activate.bat" (
    call zerodefect_env\Scripts\activate.bat
)

echo What do you want to capture?
echo.
echo   1. Good parts (for training)
echo   2. Defect: crack
echo   3. Defect: scratch
echo   4. Defect: dent
echo   5. Exit
echo.
set /p CHOICE="Enter choice (1-5): "

IF "%CHOICE%"=="1" (
    echo.
    echo [INFO] Starting good-part capture. Press SPACE to save, Q to quit.
    python src/capture_good.py --target-count 120
)
IF "%CHOICE%"=="2" (
    echo.
    echo [INFO] Starting CRACK defect capture. Press SPACE to save, Q to quit.
    python src/capture_defects.py --defect-type crack
)
IF "%CHOICE%"=="3" (
    echo.
    echo [INFO] Starting SCRATCH defect capture. Press SPACE to save, Q to quit.
    python src/capture_defects.py --defect-type scratch
)
IF "%CHOICE%"=="4" (
    echo.
    echo [INFO] Starting DENT defect capture. Press SPACE to save, Q to quit.
    python src/capture_defects.py --defect-type dent
)
IF "%CHOICE%"=="5" (
    exit /b 0
)

echo.
pause

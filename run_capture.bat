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
echo   5. Defect: corrosion
echo   6. Defect: paint-peel
echo   7. Defect: missing-head
echo   8. Exit
echo.
set /p CHOICE="Enter choice (1-8): "

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
    echo.
    echo [INFO] Starting CORROSION defect capture. Press SPACE to save, Q to quit.
    python src/capture_defects.py --defect-type corrosion
)
IF "%CHOICE%"=="6" (
    echo.
    echo [INFO] Starting PAINT-PEEL defect capture. Press SPACE to save, Q to quit.
    python src/capture_defects.py --defect-type paint-peel
)
IF "%CHOICE%"=="7" (
    echo.
    echo [INFO] Starting MISSING-HEAD defect capture. Press SPACE to save, Q to quit.
    python src/capture_defects.py --defect-type missing-head
)
IF "%CHOICE%"=="8" (
    exit /b 0
)

echo.
pause

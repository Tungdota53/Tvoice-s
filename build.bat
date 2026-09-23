@echo off
echo ===================================================
echo   Building Tvoice-s Windows Executable Package
echo ===================================================

if not exist ".venv\Scripts\pyinstaller.exe" (
    echo [INFO] Installing PyInstaller into virtualenv...
    uv pip install pyinstaller
)

echo [INFO] Running PyInstaller build...
".venv\Scripts\pyinstaller.exe" --noconfirm Tvoice.spec

if %ERRORLEVEL% equ 0 (
    echo ===================================================
    echo   BUILD SUCCESSFUL!
    echo   Output directory: dist\Tvoice-s\
    echo   Executable: dist\Tvoice-s\Tvoice-s.exe
    echo ===================================================
) else (
    echo [ERROR] Build failed with code %ERRORLEVEL%
)

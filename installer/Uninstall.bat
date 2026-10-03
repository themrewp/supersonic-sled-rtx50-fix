@echo off
setlocal EnableExtensions
title Supersonic Sled RTX 50-series fix - Uninstall
echo.
echo  Supersonic Sled RTX 50-series fix - Uninstaller
echo  ================================================
echo.
set "HERE=%~dp0"
set "BIN="
call :try "%HERE%."
call :try "%HERE%bin"
call :try "%HERE%.."
call :try "%HERE%..\bin"
call :try "%ProgramFiles(x86)%\NVIDIA Corporation\Supersonic Sled\bin"
call :try "%ProgramFiles%\NVIDIA Corporation\Supersonic Sled\bin"
for %%D in (C D E F G H) do call :try "%%D:\SuperSonicSled\bin"
for %%D in (C D E F G H) do call :try "%%D:\Supersonic Sled\bin"
if defined BIN goto found

:ask
echo  Drag the Supersonic Sled folder onto this window (or paste its path), then press Enter.
echo.
set "USERPATH="
set /p "USERPATH=  Folder: "
if not defined USERPATH goto ask
set "USERPATH=%USERPATH:"=%"
call :try "%USERPATH%"
call :try "%USERPATH%\bin"
for %%F in ("%USERPATH%") do call :try "%%~dpF."
if defined BIN goto found
echo  SupersonicSled.exe isn't in that folder. Try again.
goto ask

:found
if not exist "%BIN%\original_cuda_dlls\cudart32_30_11.dll" (
    echo  No backup found in %BIN%\original_cuda_dlls
    echo  Nothing to restore. If you need the original files, reinstall the demo.
    echo.
    pause
    exit /b 1
)
set "FAIL="
for %%F in (cudart32_30_11.dll cufft32_30_11.dll) do (
    copy /y "%BIN%\original_cuda_dlls\%%F" "%BIN%\%%F" >nul || set "FAIL=1"
)
if exist "%BIN%\nvcuda.dll" del "%BIN%\nvcuda.dll" || set "FAIL=1"
if exist "%BIN%\cpu_cuda_log.txt" del "%BIN%\cpu_cuda_log.txt"
if defined FAIL (
    echo  Restoring failed. Close the demo, then right-click Uninstall.bat and pick "Run as administrator".
    echo.
    pause
    exit /b 1
)
echo  Done. The original NVIDIA files are back in place.
echo.
pause
exit /b 0

:try
if defined BIN goto :eof
if exist "%~1\SupersonicSled.exe" set "BIN=%~f1"
goto :eof

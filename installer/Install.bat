@echo off
setlocal EnableExtensions
title Supersonic Sled RTX 50-series fix - Install
echo.
echo  Supersonic Sled RTX 50-series fix - Installer
echo  ==============================================
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
echo  I couldn't find Supersonic Sled automatically.
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
echo.
echo  SupersonicSled.exe isn't in that folder. Try again.
echo.
goto ask

:found
echo  Found the demo in: %BIN%
echo.
if not exist "%BIN%\original_cuda_dlls" mkdir "%BIN%\original_cuda_dlls" 2>nul
for %%F in (cudart32_30_11.dll cufft32_30_11.dll) do (
    if exist "%BIN%\%%F" if not exist "%BIN%\original_cuda_dlls\%%F" copy /y "%BIN%\%%F" "%BIN%\original_cuda_dlls\%%F" >nul
)
set "FAIL="
for %%F in (cudart32_30_11.dll cufft32_30_11.dll nvcuda.dll) do (
    copy /y "%HERE%fix-files\%%F" "%BIN%\%%F" >nul || set "FAIL=1"
)
if defined FAIL (
    echo  Copying failed. The game folder is probably protected.
    echo  Close the demo if it's open, then right-click Install.bat and pick "Run as administrator".
    echo.
    pause
    exit /b 1
)
echo  Done! The fix is installed.
echo  Your original files were backed up to:
echo    %BIN%\original_cuda_dlls
echo.
echo  Start Supersonic Sled the usual way.
echo  To remove the fix later, double-click Uninstall.bat.
echo.
pause
exit /b 0

:try
if defined BIN goto :eof
if exist "%~1\SupersonicSled.exe" set "BIN=%~f1"
goto :eof

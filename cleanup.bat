@echo off
REM Set the script directory
set SCRIPT_DIR=%~dp0
set STATIC_DIR=%SCRIPT_DIR%static

REM Check if the static directory exists
if not exist "%STATIC_DIR%" (
    echo ERROR: The static directory does not exist.
    exit /b 1
)

REM Check for .mp4 files older than 1 hour and delete them
forfiles /p "%STATIC_DIR%" /s /m *.mp4 /d -1 /c "cmd /c del @path" >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo No .mp4 files older than 1 hour found.
) else (
    echo Deleted .mp4 files older than 1 hour.
)

REM Check for .mp3 files older than 1 hour and delete them
forfiles /p "%STATIC_DIR%" /s /m *.mp3 /d -1 /c "cmd /c del @path" >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo No .mp3 files older than 1 hour found.
) else (
    echo Deleted .mp3 files older than 1 hour.
)

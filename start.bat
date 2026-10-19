@echo off
REM Starts Metis in this window. Ctrl+C (or closing the window) stops everything cleanly.
REM Options: --open (open the dashboard), --dev (live-reload dashboard), --verbose (all log lines)
cd /d "%~dp0"
set "PYTHON=python"
if exist "venv\Scripts\python.exe" set "PYTHON=venv\Scripts\python.exe"
"%PYTHON%" metis.py %*

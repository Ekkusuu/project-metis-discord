@echo off
REM Stops a Metis that was left running (e.g. after a crash). Normally Ctrl+C in the Metis window is enough.
cd /d "%~dp0"
set "PYTHON=python"
if exist "venv\Scripts\python.exe" set "PYTHON=venv\Scripts\python.exe"
"%PYTHON%" metis.py --stop

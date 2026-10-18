@echo off
setlocal
cd /d "%~dp0"
title Metis Launcher

echo ============================================================
echo  Project Metis - Discord bot, local LLM and dashboard
echo ============================================================
echo.

if not exist ".env" (
    echo [!] No .env file found. Copy .env.example to .env and add your DISCORD_BOT_TOKEN.
    echo.
)

REM Prefer the project's virtual environment if there is one
set "PYTHON=python"
if exist "venv\Scripts\python.exe" set "PYTHON=venv\Scripts\python.exe"

REM Ports come from config.yaml (llm_service.port / backend.port)
set "LLM_PORT="
set "BACKEND_PORT="
for /f "usebackq tokens=1,2" %%a in (`%PYTHON% -c "from backend.config import get_config as g; c = g(); print(c['llm_service']['port'], c['backend']['port'])"`) do (
    set "LLM_PORT=%%a"
    set "BACKEND_PORT=%%b"
)
if not defined BACKEND_PORT (
    echo [!] Could not read the ports from config.yaml - is Python set up? See README.md.
    pause
    exit /b 1
)

REM Fail early, with a clear message, if another program already uses one of the ports
call :check_port %LLM_PORT% llm_service.port || exit /b 1
call :check_port %BACKEND_PORT% backend.port || exit /b 1

if not exist "backend\llm_service\node_modules" (
    echo Installing LLM service dependencies...
    pushd backend\llm_service && call npm install && popd
)
if not exist "frontend\node_modules" (
    echo Installing dashboard dependencies...
    pushd frontend && call npm install && popd
)

echo Starting LLM service (Node.js) on port %LLM_PORT%...
start "Metis LLM Service" cmd /k "cd /d backend\llm_service && npm start"

echo Waiting for the model to load...
set /a tries=0
:wait_llm
set /a tries+=1
powershell -NoProfile -Command "try { $r = Invoke-RestMethod -TimeoutSec 2 http://127.0.0.1:%LLM_PORT%/health; if ($r.model_loaded) { exit 0 } else { exit 1 } } catch { exit 1 }" >nul 2>&1
if %errorlevel%==0 goto llm_ready
if %tries% geq 90 (
    echo [!] LLM service did not come up after 3 minutes - check the "Metis LLM Service" window.
    goto llm_ready
)
timeout /t 2 /nobreak >nul
goto wait_llm
:llm_ready

echo Starting backend + Discord bot (Python) on port %BACKEND_PORT%...
start "Metis Backend" cmd /k "%PYTHON% -m backend"

echo Starting dashboard (Vite)...
start "Metis Dashboard" cmd /k "cd /d frontend && set VITE_API_URL=http://127.0.0.1:%BACKEND_PORT%&& npm run dev"

echo.
echo ============================================================
echo  LLM service : http://localhost:%LLM_PORT%/health
echo  Backend API : http://localhost:%BACKEND_PORT%/docs
echo  Dashboard   : http://localhost:5173
echo ============================================================
echo.
echo Press any key to stop all Metis services...
pause >nul

echo Stopping services...
for %%T in ("Metis LLM Service*" "Metis Backend*" "Metis Dashboard*") do (
    taskkill /FI "WINDOWTITLE eq %%~T" /T /F >nul 2>&1
)
echo Done.
endlocal
exit /b 0

:check_port
%PYTHON% -c "import sys; from backend.config import port_in_use; sys.exit(1 if port_in_use('127.0.0.1', %1) else 0)"
if errorlevel 1 (
    echo [!] Port %1 is already in use by another program or another copy of Metis.
    echo     Close that program, or set a free port under %2 in config.yaml.
    echo.
    pause
    exit /b 1
)
exit /b 0

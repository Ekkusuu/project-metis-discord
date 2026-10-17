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

if not exist "backend\llm_service\node_modules" (
    echo Installing LLM service dependencies...
    pushd backend\llm_service && call npm install && popd
)
if not exist "frontend\node_modules" (
    echo Installing dashboard dependencies...
    pushd frontend && call npm install && popd
)

echo Starting LLM service (Node.js)...
start "Metis LLM Service" cmd /k "cd /d backend\llm_service && npm start"

echo Waiting for the model to load...
set /a tries=0
:wait_llm
set /a tries+=1
powershell -NoProfile -Command "try { $r = Invoke-RestMethod -TimeoutSec 2 http://localhost:3000/health; if ($r.model_loaded) { exit 0 } else { exit 1 } } catch { exit 1 }" >nul 2>&1
if %errorlevel%==0 goto llm_ready
if %tries% geq 90 (
    echo [!] LLM service did not come up after 3 minutes - starting the backend anyway.
    goto llm_ready
)
timeout /t 2 /nobreak >nul
goto wait_llm
:llm_ready

echo Starting backend + Discord bot (Python)...
start "Metis Backend" cmd /k "%PYTHON% -m uvicorn backend.main:app --host 127.0.0.1 --port 8000"

echo Starting dashboard (Vite)...
start "Metis Dashboard" cmd /k "cd /d frontend && npm run dev"

echo.
echo ============================================================
echo  LLM service : http://localhost:3000/health
echo  Backend API : http://localhost:8000/docs
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

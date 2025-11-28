@echo off
echo ============================================================
echo Starting Project Metis Services (Discord Bot Edition)
echo ============================================================
echo.

REM Start the Node.js LLM service in a new window
echo Starting LLM Service (Node.js)...
start "Metis LLM Service" cmd /k "cd backend\llm_service && npm start"

REM Wait a bit for the LLM service to start
timeout /t 5 /nobreak

REM Start the Python backend
echo Starting FastAPI Backend (Python)...
start "Metis Backend" cmd /k "uvicorn backend.main:app --reload"

REM Wait a bit for the backend to start
timeout /t 3 /nobreak

REM Discord bot will be started automatically by backend
REM Wait a bit for the backend to initialize
timeout /t 2 /nobreak

REM Start the frontend dev server
echo Starting Frontend (Vite)...
start "Metis Frontend" cmd /k "cd frontend && npm run dev"

echo.
echo ============================================================
echo All services started!
echo ============================================================
echo LLM Service: http://localhost:3000
echo Backend API: http://localhost:8000
echo Discord Bot: Connected to Discord
echo Frontend UI: http://localhost:5173
echo ============================================================
echo The UI will display Discord messages in real-time.
echo Input box is disabled - all messages come from Discord!
echo ============================================================
echo.
echo Press any key to stop all services...
echo (Note: Close the service windows manually or use Ctrl+C in each)
pause >nul

REM Kill Node.js and Python processes (more targeted)
echo Stopping services...
wmic process where "commandline like '%%npm start%%' and commandline like '%%llm_service%%'" delete 2>nul
wmic process where "commandline like '%%uvicorn%%' and commandline like '%%backend.main%%'" delete 2>nul
wmic process where "commandline like '%%vite%%' and commandline like '%%frontend%%'" delete 2>nul
echo.
echo Services stopped. You can now close the terminal windows.
pause

@echo off
echo ============================================================
echo Starting Project Metis Services
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

REM Start the frontend dev server
echo Starting Frontend (Vite)...
start "Metis Frontend" cmd /k "cd frontend && npm run dev"

echo.
echo ============================================================
echo All services started!
echo ============================================================
echo LLM Service: http://localhost:3000
echo Backend API: http://localhost:8000
echo Frontend: http://localhost:5173
echo ============================================================
echo.
echo Press any key to stop all services...
pause >nul

REM Kill all service windows
taskkill /FI "WindowTitle eq Metis*" /F

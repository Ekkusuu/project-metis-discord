@echo off
REM install_all_node_deps.bat — install npm packages for frontend and backend LLM service
REM Usage: run from project root: install_all_node_deps.bat

setlocal


echo Installing frontend dependencies (npm install first)...
pushd "%~dp0frontend"

echo Frontend: running npm install
npm install
if errorlevel 1 set FRONTEND_ERR=%ERRORLEVEL%

IF EXIST package-lock.json (
    echo Frontend: Found package-lock.json — running npm ci
    npm ci
    if errorlevel 1 if not defined FRONTEND_ERR set FRONTEND_ERR=%ERRORLEVEL%
) ELSE (
    echo Frontend: No package-lock.json — skipped npm ci
)
popd

echo.
echo Installing backend LLM service dependencies (npm install first)...
pushd "%~dp0backend\llm_service"

echo Backend LLM service: running npm install
npm install
if errorlevel 1 set BACKEND_ERR=%ERRORLEVEL%

IF EXIST package-lock.json (
    echo Backend LLM service: Found package-lock.json — running npm ci
    npm ci
    if errorlevel 1 if not defined BACKEND_ERR set BACKEND_ERR=%ERRORLEVEL%
) ELSE (
    echo Backend LLM service: No package-lock.json — skipped npm ci
)
popd

echo.
echo Summary:
if defined FRONTEND_ERR (
    echo Frontend install failed with exit code %FRONTEND_ERR%
) ELSE (
    echo Frontend install succeeded
)

if defined BACKEND_ERR (
    echo Backend LLM service install failed with exit code %BACKEND_ERR%
) ELSE (
    echo Backend LLM service install succeeded
)

set EXITCODE=0
if defined FRONTEND_ERR set EXITCODE=%FRONTEND_ERR%
if defined BACKEND_ERR if %EXITCODE% EQU 0 set EXITCODE=%BACKEND_ERR%

endlocal & exit /b %EXITCODE%

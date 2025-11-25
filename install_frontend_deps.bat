@echo off
REM install_frontend_deps.bat — install npm packages for the frontend
REM Usage: double-click or run from cmd in the project root: install_frontend_deps.bat

REM Change directory to the frontend folder relative to this script
pushd "%~dp0frontend"

IF EXIST package-lock.json (
    echo Found package-lock.json — running npm ci
    npm ci
) ELSE (
    echo No package-lock.json — running npm install
    npm install
)

set EXITCODE=%ERRORLEVEL%
popd
exit /b %EXITCODE%
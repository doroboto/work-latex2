@echo off
chcp 65001 >nul
REM ============================================================
REM  TeX Live Portable - VS Code 실행 스크립트
REM  이 스크립트는 PATH 설정 후 VS Code를 실행합니다.
REM  VS Code 내부 터미널에서 latexmk, xelatex 등을 사용할 수 있습니다.
REM ============================================================

REM 현재 스크립트의 위치를 기준으로 경로 설정
set "SCRIPT_DIR=%~dp0"
set "TEXLIVE_BIN=%SCRIPT_DIR%texlive-portable\bin\windows"
set "PROJECT_ROOT=%SCRIPT_DIR%.."

REM PATH에 TeX Live 추가
set "PATH=%TEXLIVE_BIN%;%PATH%"

echo.
echo ========================================
echo  TeX Live Portable + VS Code 실행
echo ========================================
echo.
echo  TeX Live 경로: %TEXLIVE_BIN%
echo  프로젝트 경로: %PROJECT_ROOT%
echo.

REM VS Code 실행 (프로젝트 루트 폴더를 열기)
echo VS Code를 실행합니다...
start "" code "%PROJECT_ROOT%"

echo.
echo VS Code가 실행되었습니다.
echo LaTeX Workshop 확장 프로그램이 자동으로 빌드를 수행합니다.
echo.
echo 수동 빌드 명령어 (VS Code 터미널에서):
echo   latexmk -xelatex -outdir=build main.tex
echo.
pause

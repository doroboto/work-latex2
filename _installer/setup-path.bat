@echo off
chcp 65001 >nul
REM ============================================================
REM  TeX Live Portable - PATH 설정 스크립트
REM  이 스크립트는 현재 CMD 세션에만 PATH를 추가합니다.
REM  VS Code를 이 CMD에서 실행하면 LaTeX Workshop이 작동합니다.
REM ============================================================

REM 현재 스크립트의 위치를 기준으로 texlive-portable 경로 설정
set "SCRIPT_DIR=%~dp0"
set "TEXLIVE_BIN=%SCRIPT_DIR%texlive-portable\bin\windows"

REM PATH에 TeX Live 추가
set "PATH=%TEXLIVE_BIN%;%PATH%"

echo.
echo ========================================
echo  TeX Live Portable PATH 설정 완료
echo ========================================
echo.
echo  TeX Live 경로: %TEXLIVE_BIN%
echo.
echo  확인 명령어:
echo    xelatex --version
echo    latexmk --version
echo    biber --version
echo.
echo  VS Code 실행:
echo    code .
echo.
echo ========================================
echo.

REM 버전 확인
xelatex --version 2>nul | findstr /i "XeTeX"
if errorlevel 1 (
    echo [오류] xelatex를 찾을 수 없습니다.
    echo        texlive-portable 폴더가 올바른 위치에 있는지 확인하세요.
) else (
    echo [성공] xelatex 사용 가능
)
echo.

@echo off
chcp 65001 >nul
REM ============================================================
REM  TeX Live Portable - 단일 파일 빌드 스크립트
REM  사용법: build.bat [폴더명] [파일명]
REM  예시:   build.bat paper main
REM          build.bat thesis main
REM ============================================================

REM 현재 스크립트의 위치를 기준으로 경로 설정
set "SCRIPT_DIR=%~dp0"
set "TEXLIVE_BIN=%SCRIPT_DIR%texlive-portable\bin\windows"
set "PROJECT_ROOT=%SCRIPT_DIR%.."

REM PATH에 TeX Live 추가
set "PATH=%TEXLIVE_BIN%;%PATH%"

REM 인자 확인
if "%~1"=="" (
    echo.
    echo 사용법: build.bat [폴더명] [파일명]
    echo.
    echo 예시:
    echo   build.bat paper main      - paper/main.tex 빌드
    echo   build.bat report main     - report/main.tex 빌드
    echo   build.bat thesis main     - thesis/main.tex 빌드
    echo.
    goto :eof
)

set "TARGET_FOLDER=%~1"
set "TARGET_FILE=%~2"
if "%TARGET_FILE%"=="" set "TARGET_FILE=main"

set "WORK_DIR=%PROJECT_ROOT%\%TARGET_FOLDER%"

if not exist "%WORK_DIR%\%TARGET_FILE%.tex" (
    echo [오류] 파일을 찾을 수 없습니다: %WORK_DIR%\%TARGET_FILE%.tex
    goto :eof
)

echo.
echo ========================================
echo  빌드 시작: %TARGET_FOLDER%/%TARGET_FILE%.tex
echo ========================================
echo.

REM build 폴더 생성
if not exist "%WORK_DIR%\build" mkdir "%WORK_DIR%\build"

REM latexmk로 빌드
cd /d "%WORK_DIR%"
"%TEXLIVE_BIN%\latexmk.exe" -xelatex -interaction=nonstopmode -outdir=build "%TARGET_FILE%.tex"

if errorlevel 1 (
    echo.
    echo [오류] 빌드 실패. 로그 파일을 확인하세요: build/%TARGET_FILE%.log
) else (
    echo.
    echo [성공] 빌드 완료: build/%TARGET_FILE%.pdf
)
echo.

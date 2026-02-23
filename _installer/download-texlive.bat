@echo off
chcp 65001 >nul
REM ============================================================
REM  TeX Live Portable 다운로드/설치 스크립트
REM  GitHub Releases에서 texlive-portable.zip을 다운로드합니다.
REM ============================================================

set "SCRIPT_DIR=%~dp0"
set "TEXLIVE_DIR=%SCRIPT_DIR%texlive-portable"

echo.
echo ========================================
echo  TeX Live Portable 설치 스크립트
echo ========================================
echo.

REM 이미 설치되어 있는지 확인
if exist "%TEXLIVE_DIR%\bin\windows\xelatex.exe" (
    echo [정보] TeX Live Portable이 이미 설치되어 있습니다.
    echo       경로: %TEXLIVE_DIR%
    echo.
    set /p REINSTALL="다시 설치하시겠습니까? (y/N): "
    if /i not "%REINSTALL%"=="y" goto :eof
    echo.
    echo 기존 설치를 삭제합니다...
    rmdir /s /q "%TEXLIVE_DIR%" 2>nul
)

echo.
echo ========================================
echo  다운로드 방법을 선택하세요
echo ========================================
echo.
echo  [1] GitHub Releases에서 다운로드 (자동)
echo  [2] 수동으로 zip 파일 지정
echo  [3] 취소
echo.
set /p CHOICE="선택 (1-3): "

if "%CHOICE%"=="1" goto :download_github
if "%CHOICE%"=="2" goto :manual_install
if "%CHOICE%"=="3" goto :eof
goto :eof

:download_github
echo.
echo GitHub Releases에서 다운로드 중...
echo.
echo [주의] 회사 보안 정책으로 다운로드가 차단될 수 있습니다.
echo       차단되면 수동 다운로드(옵션 2)를 사용하세요.
echo.

REM GitHub 저장소 정보
set "GITHUB_REPO=doroboto/work-latex2"
set "RELEASE_TAG=v1.0"
set "ZIP_NAME=texlive-portable.zip"

REM 다운로드 URL 구성
set "DOWNLOAD_URL=https://github.com/%GITHUB_REPO%/releases/download/%RELEASE_TAG%/%ZIP_NAME%"

echo 다운로드 URL: %DOWNLOAD_URL%
echo.

REM PowerShell로 다운로드
powershell -Command "Invoke-WebRequest -Uri '%DOWNLOAD_URL%' -OutFile '%SCRIPT_DIR%%ZIP_NAME%'" 2>nul
if errorlevel 1 (
    echo.
    echo [오류] 다운로드 실패. 수동 다운로드를 진행하세요.
    echo.
    echo  1. 브라우저에서 다음 URL 접속:
    echo     https://github.com/%GITHUB_REPO%/releases
    echo.
    echo  2. texlive-portable.zip 다운로드
    echo.
    echo  3. 이 스크립트를 다시 실행하고 옵션 2 선택
    echo.
    pause
    goto :eof
)

echo 다운로드 완료. 압축 해제 중...
goto :extract

:manual_install
echo.
echo texlive-portable.zip 파일 경로를 입력하세요.
echo (예: C:\Users\user\Downloads\texlive-portable.zip)
echo.
set /p ZIP_PATH="ZIP 파일 경로: "

if not exist "%ZIP_PATH%" (
    echo [오류] 파일을 찾을 수 없습니다: %ZIP_PATH%
    pause
    goto :eof
)

echo 파일 복사 중...
copy "%ZIP_PATH%" "%SCRIPT_DIR%texlive-portable.zip" >nul
goto :extract

:extract
echo.
echo 압축 해제 중... (약 1-2분 소요)
echo.

powershell -Command "Expand-Archive -Path '%SCRIPT_DIR%texlive-portable.zip' -DestinationPath '%SCRIPT_DIR%' -Force"
if errorlevel 1 (
    echo [오류] 압축 해제 실패.
    pause
    goto :eof
)

REM 다운로드한 zip 삭제
del "%SCRIPT_DIR%texlive-portable.zip" 2>nul

echo.
echo ========================================
echo  설치 완료!
echo ========================================
echo.

REM 설치 확인
if exist "%TEXLIVE_DIR%\bin\windows\xelatex.exe" (
    echo [성공] TeX Live Portable 설치 완료
    echo        경로: %TEXLIVE_DIR%
    echo.
    echo 버전 확인:
    "%TEXLIVE_DIR%\bin\windows\xelatex.exe" --version 2>nul | findstr /i "XeTeX"
) else (
    echo [오류] 설치가 올바르게 완료되지 않았습니다.
    echo        texlive-portable.zip 파일이 올바른지 확인하세요.
)

echo.
echo 이제 VS Code에서 LaTeX 문서를 빌드할 수 있습니다.
echo.
pause

# TeX Live Portable 사용 가이드

이 문서는 **보안 환경**에서 MiKTeX/TeX Live 설치가 차단된 경우, 포터블 버전으로 LaTeX 문서를 빌드하는 방법을 안내합니다.

---

## 최초 설치 (처음 사용 시)

GitHub에서 clone/다운로드 후, TeX Live Portable을 별도로 설치해야 합니다.

### 방법 1: 자동 설치 스크립트 (권장)

```batch
_installer\download-texlive.bat
```

스크립트가 GitHub Releases에서 자동으로 다운로드합니다.
회사 보안으로 차단되면 수동 설치(방법 2)를 사용하세요.

### 방법 2: 수동 설치

1. GitHub Releases 페이지에서 `texlive-portable.zip` 다운로드
2. `_installer\download-texlive.bat` 실행
3. 옵션 2 선택 후 다운로드한 zip 파일 경로 입력

### 방법 3: OneDrive/USB로 복사

이미 설치된 `_installer/texlive-portable/` 폴더를 통째로 복사해도 됩니다.

---

## 빠른 시작

### 방법 1: VS Code에서 바로 사용 (권장)

1. 이 프로젝트 폴더를 VS Code로 엽니다
2. `.tex` 파일을 열고 저장하면 자동으로 빌드됩니다
3. 빌드된 PDF는 각 폴더의 `build/` 디렉토리에 생성됩니다

> **참고**: `.vscode/settings.json`에 포터블 TeX Live 경로가 이미 설정되어 있습니다.

### 방법 2: 스크립트로 VS Code 실행

```
_installer\launch-vscode.bat
```

이 스크립트는 PATH를 설정한 후 VS Code를 실행합니다. VS Code 터미널에서 `latexmk`, `xelatex` 등을 직접 사용할 수 있습니다.

### 방법 3: 명령줄에서 직접 빌드

```batch
REM PATH 설정
_installer\setup-path.bat

REM 빌드 (paper 폴더의 main.tex)
_installer\build.bat paper main

REM 또는 직접 latexmk 사용
cd paper
..\\_installer\\texlive-portable\\bin\\windows\\latexmk.exe -xelatex -outdir=build main.tex
```

---

## 포함된 파일

```
_installer/
├── texlive-portable/     ← TeX Live Portable (별도 설치 필요, ~1.1GB)
├── download-texlive.bat  ← TeX Live 다운로드/설치 스크립트
├── setup-path.bat        ← CMD에서 PATH 설정
├── launch-vscode.bat     ← PATH 설정 + VS Code 실행
├── build.bat             ← 단일 파일 빌드 스크립트
└── README_PORTABLE.md    ← 이 문서
```

> **참고**: `texlive-portable/` 폴더는 git에 포함되지 않습니다.
> 처음 사용 시 `download-texlive.bat`으로 설치하거나 Releases에서 다운로드하세요.

---

## VS Code LaTeX Workshop 사용법

### 자동 빌드
- `.tex` 파일을 저장하면 자동으로 빌드됩니다
- 빌드 결과는 `build/` 폴더에 생성됩니다

### 수동 빌드
- `Ctrl + Alt + B` : 빌드 실행
- `Ctrl + Alt + V` : PDF 미리보기

### 빌드 레시피 선택
1. `Ctrl + Shift + P` 열기
2. "LaTeX Workshop: Build with recipe" 입력
3. 원하는 레시피 선택:
   - **latexmk (xelatex) - Portable**: 일반적인 경우 (권장)
   - **xelatex -> biber -> xelatex*2 - Portable**: 참고문헌(biblatex) 포함 시
   - **xelatex only - Portable**: 단순 빌드

---

## 터미널에서 직접 빌드

VS Code 터미널 또는 CMD에서:

```batch
REM 기본 빌드 (latexmk 사용)
latexmk -xelatex -outdir=build main.tex

REM xelatex만 사용
xelatex -output-directory=build main.tex

REM 참고문헌 포함 빌드
xelatex -output-directory=build main.tex
biber --input-directory=build --output-directory=build main
xelatex -output-directory=build main.tex
xelatex -output-directory=build main.tex

REM 빌드 파일 정리
latexmk -c
```

---

## 문제 해결

### "latexmk를 찾을 수 없습니다" 오류

**원인**: PATH가 설정되지 않음

**해결**:
1. `_installer\setup-path.bat`을 실행하여 PATH 설정
2. 또는 `_installer\launch-vscode.bat`으로 VS Code 실행

### VS Code에서 빌드가 안 됨

**원인**: LaTeX Workshop 확장 프로그램 미설치 또는 설정 문제

**해결**:
1. LaTeX Workshop 확장 프로그램 설치 확인
2. `.vscode/settings.json` 파일 존재 확인
3. VS Code 재시작

### 한글이 깨짐

**원인**: 폰트 경로 문제 또는 kotex 패키지 누락

**해결**:
1. `fonts/` 폴더에 나눔 폰트 파일 확인
2. `.tex` 파일에서 폰트 경로 확인:
   ```latex
   \setmainfont{NanumMyeongjo-Regular.ttf}[
     Path=../fonts/,
     BoldFont=NanumMyeongjo-Bold.ttf
   ]
   ```

### 보안 프로그램이 실행을 차단함

**원인**: 보안 프로그램이 .exe 실행 차단

**해결**:
1. 보안 팀에 `texlive-portable/bin/windows/` 폴더의 실행 파일 허용 요청
2. 필요한 파일: `xelatex.exe`, `latexmk.exe`, `biber.exe`

---

## 포함된 패키지

이 포터블 TeX Live에는 다음 패키지가 포함되어 있습니다:

- **기본**: collection-basic, collection-latex
- **XeLaTeX**: collection-xetex
- **한글**: collection-langkorean (kotex, xetexko 등)
- **수학**: collection-mathscience (amsmath, amssymb 등)
- **폰트**: collection-fontsrecommended
- **참고문헌**: biber, biblatex
- **유틸리티**: latexmk, csquotes

추가 패키지가 필요한 경우:
```batch
_installer\texlive-portable\bin\windows\tlmgr.bat install [패키지명]
```

---

## 폴더 구조

```
work-latex-main/
├── _installer/           ← 포터블 TeX Live 및 스크립트
├── fonts/                ← 나눔 폰트 (프로젝트 내장)
├── paper/                ← 논문 템플릿
│   ├── main.tex
│   ├── references.bib
│   └── build/            ← 빌드 결과물
├── report/               ← 보고서 템플릿
├── thesis/               ← 학위논문 템플릿
├── work1/                ← 실제 작업 프로젝트
├── .vscode/
│   └── settings.json     ← LaTeX Workshop 설정 (포터블 경로)
└── README.md
```

---

## 버전 정보

- TeX Live 2025
- XeTeX 3.141592653-2.6-0.999997
- latexmk 4.87
- biber 2.20
- 전체 용량: 약 1.1GB

---

## 배포 방법 (관리자용)

### GitHub Releases에 TeX Live 업로드

`texlive-portable/` 폴더는 용량이 커서(~1.1GB) git에 포함되지 않습니다.
GitHub Releases를 통해 배포합니다.

**1. texlive-portable 폴더 압축**

```powershell
# PowerShell에서 실행
cd C:\Users\kakha\OneDrive\_Project\work\work-latex-main\_installer
Compress-Archive -Path "texlive-portable" -DestinationPath "texlive-portable.zip" -CompressionLevel Optimal
```

**2. GitHub Release 생성**

1. GitHub Repository > Releases > "Create new release"
2. Tag: `v1.0` (또는 원하는 버전)
3. Title: `TeX Live Portable v1.0`
4. Assets에 `texlive-portable.zip` 첨부 (Attach binaries)
5. Publish release

**3. download-texlive.bat 수정**

`download-texlive.bat` 파일에서 GitHub 저장소 정보를 수정:

```batch
set "GITHUB_REPO=doroboto/work-latex2"
set "RELEASE_TAG=v1.0"
```

### 사용자 안내

사용자는 다음 순서로 설치합니다:

1. `git clone https://github.com/doroboto/work-latex2.git`
2. `cd work-latex-main`
3. `_installer\download-texlive.bat` 실행
4. VS Code로 프로젝트 열기

---

## 참고

- 이 포터블 버전은 시스템 설치 없이 동작합니다
- 레지스트리 수정이나 시스템 PATH 변경이 없습니다
- 폴더 전체를 복사하면 다른 PC에서도 사용 가능합니다

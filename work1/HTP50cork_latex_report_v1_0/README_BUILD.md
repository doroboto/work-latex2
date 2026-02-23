# HTP50cork LaTeX Report (Local Build Package)

이 압축파일은 로컬(PC)에서 바로 편집/컴파일 가능한 LaTeX 프로젝트입니다.

## 1) 폴더 구성
- main.tex                 : 메인 문서
- sections/                : 장(Chapter)별 본문 tex
- figures/                 : 보고서 삽입용 이미지(벡터 PDF)
- code/                    : 부록용 코드 발췌 + 원본(app.py) + 참고자료(ADD/README)

## 2) 권장 컴파일 방법 (XeLaTeX)
한글 문서이므로 XeLaTeX 또는 LuaLaTeX을 권장합니다.

### (A) latexmk 사용 (권장)
터미널에서:
    latexmk -xelatex -outdir=build main.tex

### (B) xelatex만 사용
    xelatex -output-directory=build main.tex
    xelatex -output-directory=build main.tex

## 3) 필요 패키지
- TeX Live (Full 권장)
- kotex (한글)
- amsmath, graphicx, listings, siunitx 등 표준 패키지

## 4) 편집 팁
- 본문은 sections/*.tex에서 수정
- 그림은 figures/에 PDF/PNG 등을 추가한 뒤 \includegraphics로 삽입
- 코드 부록은 code/ 폴더의 파일을 \lstinputlisting으로 포함

문서 버전/표지 항목(작성자/제출처/문서번호)은 main.tex의 Title Page에서 수정하세요.


## 5) PATH 점검 (Windows / VS Code / Git Bash)
TeX Live 설치 후 기존 터미널은 PATH가 갱신되지 않아 `xelatex`/`latexmk`가 안 잡힐 수 있습니다.

### (A) 먼저 재시작
- VS Code를 완전히 종료 후 다시 실행
- 터미널(CMD/PowerShell/Git Bash)도 새 창으로 다시 열기

### (B) Windows CMD/PowerShell 확인
    where xelatex
    where latexmk
    where tlmgr

정상이라면 아래 경로가 보여야 합니다:
    C:/Users/<사용자명>/texlive/2025/bin/windows

### (C) Git Bash에서 임시 PATH 추가
    export PATH="$PATH:/c/Users/<사용자명>/texlive/2025/bin/windows"
    xelatex --version
    latexmk -v
    tlmgr.bat --version

### (D) 영구 PATH (권장)
사용자 환경 변수 `Path`에 아래를 추가:
    C:/Users/<사용자명>/texlive/2025/bin/windows

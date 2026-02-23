# TeX Live 한글 논문/보고서/학위논문 템플릿 (XeLaTeX)

이 문서는 `TeX Live + XeLaTeX + kotex` 기준으로 작성되었고,
Windows/macOS/Linux/VS Code/CLI 환경에서 같은 빌드 흐름으로 쓰도록 구성했다.

## 1) 공통 준비

- TeX Live 설치
- 필수 패키지 설치:

```bash
tlmgr update --self --all
tlmgr install collection-langkorean latexmk biber
```

- 한글 문서는 `xelatex` 엔진 사용 권장

---

## 2) 글씨체(폰트) 다운로드 및 프로젝트 포함

이 프로젝트는 로컬 폰트 번들 방식으로 맞췄다.
즉, 시스템 폰트 설치 없이 `fonts/` 폴더의 파일만으로 빌드할 수 있다.

현재 기준 포함 폰트(다운로드 완료):

- `fonts/NanumMyeongjo-Regular.ttf`
- `fonts/NanumMyeongjo-Bold.ttf`
- `fonts/NanumGothic-Regular.ttf`
- `fonts/NanumGothic-Bold.ttf`

### Windows PowerShell에서 다시 받기

```powershell
$fontDir = "./fonts"
New-Item -ItemType Directory -Path $fontDir -Force | Out-Null

Invoke-WebRequest "https://github.com/google/fonts/raw/main/ofl/nanummyeongjo/NanumMyeongjo-Regular.ttf" -OutFile "$fontDir/NanumMyeongjo-Regular.ttf"
Invoke-WebRequest "https://github.com/google/fonts/raw/main/ofl/nanummyeongjo/NanumMyeongjo-Bold.ttf" -OutFile "$fontDir/NanumMyeongjo-Bold.ttf"
Invoke-WebRequest "https://github.com/google/fonts/raw/main/ofl/nanumgothic/NanumGothic-Regular.ttf" -OutFile "$fontDir/NanumGothic-Regular.ttf"
Invoke-WebRequest "https://github.com/google/fonts/raw/main/ofl/nanumgothic/NanumGothic-Bold.ttf" -OutFile "$fontDir/NanumGothic-Bold.ttf"
```

### macOS/Linux에서 다시 받기

```bash
mkdir -p fonts
curl -L "https://github.com/google/fonts/raw/main/ofl/nanummyeongjo/NanumMyeongjo-Regular.ttf" -o "fonts/NanumMyeongjo-Regular.ttf"
curl -L "https://github.com/google/fonts/raw/main/ofl/nanummyeongjo/NanumMyeongjo-Bold.ttf" -o "fonts/NanumMyeongjo-Bold.ttf"
curl -L "https://github.com/google/fonts/raw/main/ofl/nanumgothic/NanumGothic-Regular.ttf" -o "fonts/NanumGothic-Regular.ttf"
curl -L "https://github.com/google/fonts/raw/main/ofl/nanumgothic/NanumGothic-Bold.ttf" -o "fonts/NanumGothic-Bold.ttf"
```

---

## 3) 권장 폴더 구조

```text
kr-latex-template/
├─ fonts/
│  ├─ NanumMyeongjo-Regular.ttf
│  ├─ NanumMyeongjo-Bold.ttf
│  ├─ NanumGothic-Regular.ttf
│  └─ NanumGothic-Bold.ttf
├─ paper/
│  ├─ main.tex
│  ├─ references.bib
│  └─ figures/
├─ report/
│  ├─ main.tex
│  └─ figures/
├─ thesis/
│  ├─ main.tex
│  ├─ references.bib
│  └─ figures/
└─ .vscode/
   └─ settings.json
```

---

## 4) 논문용 템플릿 (`paper/main.tex`)

```tex
\documentclass[12pt,a4paper]{article}

\usepackage{kotex}
\usepackage{fontspec}
\setmainfont{NanumMyeongjo-Regular.ttf}[
  Path=../fonts/,
  BoldFont=NanumMyeongjo-Bold.ttf
]
\setsansfont{NanumGothic-Regular.ttf}[
  Path=../fonts/,
  BoldFont=NanumGothic-Bold.ttf
]

\usepackage[margin=25mm]{geometry}
\usepackage{setspace}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{amsmath,amssymb}
\usepackage{csquotes}
\usepackage[hidelinks]{hyperref}
\usepackage[backend=biber,style=authoryear,sorting=nyt]{biblatex}
\addbibresource{references.bib}

\onehalfspacing

\title{한글 논문 제목\\\large English Title (Optional)}
\author{홍길동}
\date{\today}

\begin{document}
\maketitle

\begin{abstract}
이 문서는 XeLaTeX + kotex 기반 한글 논문 템플릿이다.
인용 예시: \parencite{knuth1984texbook}.
\end{abstract}

\tableofcontents
\newpage

\section{서론}
연구 배경과 목적을 작성한다.

\section{관련 연구}
관련 연구를 정리한다.

\section{방법}
수식 예시:
\begin{equation}
f(x)=\int_0^x t^2\,dt
\end{equation}

\section{결과}
그림 예시:
\begin{figure}[h]
  \centering
  \includegraphics[width=.6\linewidth]{figures/example.png}
  \caption{예시 그림}
\end{figure}

\section{결론}
결론과 향후 과제를 작성한다.

\printbibliography

\end{document}
```

---

## 5) 보고서용 템플릿 (`report/main.tex`)

```tex
\documentclass[12pt,a4paper]{report}

\usepackage{kotex}
\usepackage{fontspec}
\setmainfont{NanumMyeongjo-Regular.ttf}[
  Path=../fonts/,
  BoldFont=NanumMyeongjo-Bold.ttf
]
\setsansfont{NanumGothic-Regular.ttf}[
  Path=../fonts/,
  BoldFont=NanumGothic-Bold.ttf
]

\usepackage[margin=25mm]{geometry}
\usepackage{setspace}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{amsmath,amssymb}
\usepackage[hidelinks]{hyperref}

\onehalfspacing

\title{한글 보고서 제목}
\author{홍길동}
\date{\today}

\begin{document}
\maketitle
\tableofcontents
\listoffigures
\listoftables
\clearpage

\chapter{개요}
보고서 목적, 범위, 일정을 작성한다.

\chapter{진행 내용}
\section{요구사항}
요구사항을 정리한다.

\section{구현}
구현 내용을 작성한다.

\chapter{결과 및 회고}
성과, 이슈, 개선안을 작성한다.

\appendix
\chapter{부록}
부가 자료를 작성한다.

\end{document}
```

---

## 6) 학위논문 스타일 템플릿 (`thesis/main.tex`)

아래 템플릿은 기본적으로 다음 순서를 포함한다.

- 표지
- 심사위원 페이지
- 국문초록
- 영문초록
- 목차/그림목차/표목차
- 본문(장/절)
- 부록
- 참고문헌

```tex
\documentclass[12pt,a4paper,oneside]{book}

\usepackage{kotex}
\usepackage{fontspec}
\setmainfont{NanumMyeongjo-Regular.ttf}[
  Path=../fonts/,
  BoldFont=NanumMyeongjo-Bold.ttf
]
\setsansfont{NanumGothic-Regular.ttf}[
  Path=../fonts/,
  BoldFont=NanumGothic-Bold.ttf
]

\usepackage[margin=30mm]{geometry}
\usepackage{setspace}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{amsmath,amssymb}
\usepackage{csquotes}
\usepackage[hidelinks]{hyperref}
\usepackage[backend=biber,style=authoryear,sorting=nyt]{biblatex}
\addbibresource{references.bib}

\onehalfspacing

% ===== 메타데이터 =====
\newcommand{\university}{OO대학교}
\newcommand{\graduateSchool}{대학원}
\newcommand{\department}{OO학과}
\newcommand{\degreeName}{OO석사}
\newcommand{\thesisTitleKr}{학위논문 한글 제목}
\newcommand{\thesisTitleEn}{English Thesis Title}
\newcommand{\authorName}{홍길동}
\newcommand{\advisorName}{지도교수 OOO}
\newcommand{\submitDate}{2026년 2월}
\newcommand{\defenseDate}{2026년 1월 15일}

\begin{document}

% ===== 표지 =====
\begin{titlepage}
  \thispagestyle{empty}
  \begin{center}
    {\Large \university \par}
    \vspace{0.5cm}
    {\Large \graduateSchool \par}
    \vspace{2.5cm}

    {\LARGE \bfseries \thesisTitleKr \par}
    \vspace{0.7cm}
    {\large \thesisTitleEn \par}
    \vspace{2.0cm}

    {\Large \department \par}
    \vspace{0.6cm}
    {\Large \authorName \par}
    \vfill
    {\Large \submitDate \par}
  \end{center}
\end{titlepage}

% ===== 심사위원 페이지 =====
\clearpage
\thispagestyle{empty}
\begin{center}
  {\Large 학위논문 심사위원 명단 \par}
  \vspace{2cm}
  {\large 논문제목: \thesisTitleKr \par}
  \vspace{1.2cm}
  {\large \degreeName\ 학위논문을 제출함 \par}
  \vspace{1.5cm}

  \begin{tabular}{ll}
    지도교수 & \advisorName \\
    심사위원장 & OOO \\
    심사위원 & OOO \\
    심사위원 & OOO \\
  \end{tabular}

  \vfill
  {\large 심사일: \defenseDate \par}
\end{center}

% ===== 앞부분 =====
\frontmatter

\chapter*{국문초록}
\addcontentsline{toc}{chapter}{국문초록}
이곳에 연구 배경, 목적, 방법, 핵심 결과, 의의를 1쪽 이내로 작성한다.

\bigskip
\noindent\textbf{주요어:} 키워드1, 키워드2, 키워드3

\chapter*{Abstract}
\addcontentsline{toc}{chapter}{Abstract}
Write your English abstract here.

\bigskip
\noindent\textbf{Keywords:} keyword1, keyword2, keyword3

\tableofcontents
\listoffigures
\listoftables

% ===== 본문 =====
\mainmatter

\chapter{서론}
연구의 배경과 목적을 작성한다.

\chapter{이론적 배경}
선행연구와 이론을 정리한다.

\chapter{연구 방법}
데이터, 실험 설계, 분석 방법을 작성한다.

\chapter{연구 결과}
결과와 해석을 작성한다.

\chapter{결론}
요약, 기여, 한계, 향후 과제를 작성한다.

% ===== 부록 =====
\appendix
\chapter{부록 A}
부가 표, 실험 설정, 코드 설명 등을 작성한다.

% ===== 참고문헌 =====
\backmatter
\printbibliography

\end{document}
```

---

## 7) 참고문헌 예시 (`paper/references.bib`, `thesis/references.bib`)

```bibtex
@book{knuth1984texbook,
  author    = {Knuth, Donald E.},
  title     = {The TeXbook},
  year      = {1984},
  publisher = {Addison-Wesley}
}
```

---

## 8) VS Code 설정 (`.vscode/settings.json`)

```json
{
  "latex-workshop.latex.tools": [
    {
      "name": "latexmk-xelatex",
      "command": "latexmk",
      "args": [
        "-xelatex",
        "-synctex=1",
        "-interaction=nonstopmode",
        "-file-line-error",
        "-outdir=build",
        "%DOC%"
      ]
    }
  ],
  "latex-workshop.latex.recipes": [
    {
      "name": "latexmk-xelatex",
      "tools": ["latexmk-xelatex"]
    }
  ],
  "latex-workshop.latex.autoBuild.run": "onSave",
  "latex-workshop.view.pdf.viewer": "tab"
}
```

---

## 9) 빌드 명령 (CLI 공통)

각 폴더(`paper`, `report`, `thesis`)에서 실행:

```bash
latexmk -xelatex -outdir=build main.tex
```

참고문헌 포함 문서 정리(clean):

```bash
latexmk -c
```

---

## 10) 이식성 팁 (다른 작업환경 대응)

- 폰트를 프로젝트 상대경로(`../fonts/`)로 고정하면 OS가 달라도 결과가 거의 동일함
- 경로는 상대경로만 사용 (`figures/...`)
- 빌드는 항상 `latexmk -xelatex`로 통일
- CI/서버에서는 TeX Live + `collection-langkorean` + `latexmk` + `biber`만 맞추면 재현성 높음
- 학교 양식이 별도로 있으면 `thesis/main.tex`의 표지/심사위원 페이지 문구와 레이아웃만 교체하면 됨

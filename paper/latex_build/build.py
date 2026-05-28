"""Convert hermes_dqn markdown papers to NeurIPS-style PDF.

Pipeline:
  1. pandoc markdown -> LaTeX body (fragment, no preamble)
  2. wrap body in NeurIPS 2024 template
  3. compile (pdflatex for EN, xelatex for ZH)
  4. run twice to resolve refs

Requirements (all already installed):
  - pypandoc (bundled pandoc.exe)
  - MiKTeX (auto-installs missing packages on first use)
"""
from __future__ import annotations
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(r"C:\Users\Mao\Desktop\DRL\Final Project\paper")
BUILD = ROOT / "latex_build"
PANDOC = Path(r"C:\Users\Mao\AppData\Local\Programs\Python\Python311\Lib\site-packages\pypandoc\files\pandoc.exe")
MIKTEX_BIN = Path(r"C:\Users\Mao\AppData\Local\Programs\MiKTeX\miktex\bin\x64")
PDFLATEX = MIKTEX_BIN / "pdflatex.exe"
XELATEX = MIKTEX_BIN / "xelatex.exe"


# ----------------------- pandoc -> LaTeX body fragment ----------------------

def md_to_tex_body(md_path: Path, out_path: Path) -> None:
    """Run pandoc to convert markdown into LaTeX fragment (no preamble)."""
    cmd = [
        str(PANDOC),
        str(md_path),
        "-f", "gfm",  # github-flavored markdown (tables)
        "-t", "latex",
        "--wrap=preserve",
        "-o", str(out_path),
    ]
    subprocess.run(cmd, check=True)


# ----------------------- template wrapper ----------------------

PREAMBLE_EN = r"""
\documentclass{article}

% NeurIPS 2024 style (preprint mode = author names visible)
\usepackage[preprint, nonatbib]{neurips_2024}

\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{hyperref}
\usepackage{url}
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{array}
\usepackage{amsfonts}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{nicefrac}
\usepackage{microtype}
\usepackage{xcolor}
\usepackage{graphicx}
\usepackage{pifont}
% pandoc-emitted helpers
\providecommand{\tightlist}{\setlength{\itemsep}{0pt}\setlength{\parskip}{0pt}}
\providecommand{\passthrough}[1]{#1}
% pandoc 3.x wraps figures in \pandocbounded{...}; we preprocess them into
% \PaperFig{path}{caption} via build.py, so just provide that macro here.
% Compact figures so we keep within the typical 6-10 page conference range.
\providecommand{\PaperFig}[2]{\begin{figure}[!htb]\centering\includegraphics[width=0.68\linewidth,height=0.20\textheight,keepaspectratio]{#1}\caption{\small #2}\end{figure}}
% Tighten vertical spacing
\setlength{\textfloatsep}{6pt plus 1pt minus 2pt}
\setlength{\intextsep}{4pt plus 1pt minus 2pt}
\setlength{\abovecaptionskip}{2pt}
% pandoc emits "\def\LTcaptype{none}" for caption-less longtables; define a
% throwaway counter so this compiles.
\newcounter{none}
% Where to find figure files (relative to this .tex file)
\graphicspath{{figures/}{./}{../figures/}}
% allow unicode dashes / minus sign without errors
\DeclareUnicodeCharacter{2212}{-}
\DeclareUnicodeCharacter{2009}{\,}
\DeclareUnicodeCharacter{2248}{$\approx$}
\DeclareUnicodeCharacter{2264}{$\leq$}
\DeclareUnicodeCharacter{2265}{$\geq$}
\DeclareUnicodeCharacter{2192}{$\rightarrow$}
\DeclareUnicodeCharacter{00D7}{$\times$}
\DeclareUnicodeCharacter{2013}{--}
\DeclareUnicodeCharacter{2014}{---}
\DeclareUnicodeCharacter{2019}{'}
\DeclareUnicodeCharacter{2018}{`}
\DeclareUnicodeCharacter{201C}{``}
\DeclareUnicodeCharacter{201D}{''}
\DeclareUnicodeCharacter{2026}{\ldots}
\DeclareUnicodeCharacter{03B5}{$\varepsilon$}
\DeclareUnicodeCharacter{03B3}{$\gamma$}
\DeclareUnicodeCharacter{0394}{$\Delta$}
\DeclareUnicodeCharacter{2205}{$\varnothing$}
\DeclareUnicodeCharacter{2713}{\ding{51}}
\DeclareUnicodeCharacter{00B1}{$\pm$}
\DeclareUnicodeCharacter{03B1}{$\alpha$}
\DeclareUnicodeCharacter{03B2}{$\beta$}
\DeclareUnicodeCharacter{03B4}{$\delta$}
\DeclareUnicodeCharacter{03BC}{$\mu$}
\DeclareUnicodeCharacter{03C3}{$\sigma$}

\title{Hermes-DQN: When Does Memory-Augmented LLM Reward Design Help DQN? A 4-Environment Analysis}

\author{%
  ShengMao Chen, Hsienan Lin, YuJou Hsin, KuanYu Chen \\[3pt]
  \small Department of Management Information Systems \\
  \small National Chung Hsing University
}

\begin{document}
\maketitle
"""

POSTAMBLE = r"""
\end{document}
"""


# Chinese version: use ctex + xelatex. fontset=none prevents ctex from auto-
# loading SimHei/SimSun (which aren't installed on Trad-Chinese Windows). We
# set Traditional-Chinese fonts manually via xeCJK below.
PREAMBLE_ZH = r"""
\documentclass[UTF8,fontset=none,a4paper]{ctexart}

% ── Font size: 20pt body text ─────────────────────────────────────────────────
% ctexart does not natively support 20pt in the documentclass option; we use
% the fontsize package (or \fontsize) to set the base size after loading.
\usepackage{fontsize}
\changefontsize[30pt]{20pt}   % {baselineskip}{fontsize}

% ── Page geometry (A4, comfortable margins for 20pt text) ─────────────────────
\usepackage[a4paper, top=2.5cm, bottom=2.5cm, left=2.8cm, right=2.8cm]{geometry}
\linespread{1.5}

% ── Fonts ─────────────────────────────────────────────────────────────────────
\usepackage{xeCJK}
% Latin / numbers: Times New Roman
\setmainfont{Times New Roman}
\setsansfont{Times New Roman}
% Chinese: 標楷體 (DFKai-SB)
\setCJKmainfont[BoldFont=DFKai-SB, ItalicFont=DFKai-SB]{DFKai-SB}
\setCJKsansfont{DFKai-SB}
\setCJKmonofont{DFKai-SB}

\usepackage{amsfonts}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{pifont}
\usepackage{hyperref}
\usepackage{url}
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{array}
\usepackage{nicefrac}
\usepackage{microtype}
\usepackage{xcolor}
\usepackage{graphicx}
\usepackage{titlesec}
% Section title formatting (mimic NeurIPS)
\titleformat{\section}{\normalfont\Large\bfseries}{\thesection}{1em}{}
\titleformat{\subsection}{\normalfont\large\bfseries}{\thesubsection}{1em}{}
\titleformat{\subsubsection}{\normalfont\normalsize\bfseries}{\thesubsubsection}{1em}{}

\providecommand{\tightlist}{\setlength{\itemsep}{0pt}\setlength{\parskip}{0pt}}
\providecommand{\passthrough}[1]{#1}
\providecommand{\PaperFig}[2]{\begin{figure}[!htb]\centering\includegraphics[width=0.68\linewidth,height=0.20\textheight,keepaspectratio]{#1}\caption{\small #2}\end{figure}}
\newcounter{none}
\graphicspath{{figures/}{./}{../figures/}}
% Tighten vertical spacing for compactness
\setlength{\textfloatsep}{6pt plus 1pt minus 2pt}
\setlength{\intextsep}{4pt plus 1pt minus 2pt}
\setlength{\abovecaptionskip}{2pt}
\setlength{\parskip}{0pt}

% xelatex needs glyphs to come from a font that contains them. Microsoft
% JhengHei does NOT include Greek letters, empty-set, check-mark, or many
% math symbols, so we map those Unicode codepoints to math/pifont commands
% (similar to \DeclareUnicodeCharacter for pdflatex).
\usepackage{newunicodechar}
\newunicodechar{∅}{$\varnothing$}
\newunicodechar{✓}{\ding{51}}
\newunicodechar{≥}{$\geq$}
\newunicodechar{≤}{$\leq$}
\newunicodechar{≈}{$\approx$}
\newunicodechar{→}{$\rightarrow$}
\newunicodechar{×}{$\times$}
\newunicodechar{±}{$\pm$}
\newunicodechar{α}{$\alpha$}
\newunicodechar{β}{$\beta$}
\newunicodechar{γ}{$\gamma$}
\newunicodechar{δ}{$\delta$}
\newunicodechar{ε}{$\varepsilon$}
\newunicodechar{μ}{$\mu$}
\newunicodechar{σ}{$\sigma$}
\newunicodechar{Δ}{$\Delta$}

% ctex uses Simplified-Chinese figure/table labels by default ("图" / "表"); the
% paper body is Traditional Chinese, so override to "圖" / "表" for consistency.
\renewcommand{\figurename}{圖}
\renewcommand{\tablename}{表}

\title{Hermes-DQN:記憶擴增之大型語言模型獎勵設計何時對 DQN 有效?四環境分析}
\author{%
  陳盛茂、林仙安、辛語柔、陳冠宇 \\[3pt]
  \small 國立中興大學　資訊管理學研究所
}
\date{}

\begin{document}
\maketitle
"""


# ----------------------- body cleanup ----------------------

def cleanup_body(tex: str, lang: str) -> str:
    """Clean up pandoc-emitted LaTeX.

    The markdown is structured as:
       H1 = paper title           -> pandoc emits \section{...}
       H2 = Abstract / 1. Intro / 2. Related Work / ... / References / Appendix
                                  -> pandoc emits \subsection{...}
       H3 = 3.1, 3.2, ...          -> pandoc emits \subsubsection{...}

    We want:
       paper title -> handled by \title in the preamble, drop it
       Abstract    -> \begin{abstract} ... \end{abstract}
       1. Intro    -> \section{Introduction}     (LaTeX adds the "1.")
       3.1         -> \subsection{...}           (LaTeX adds "3.1")
    """
    # Remove pandoc \label{...} attachments
    tex = re.sub(r"\\label\{[^}]*\}", "", tex)
    # Remove \rule lines (markdown ---)
    tex = re.sub(r"\\begin\{center\}\\rule\{[^}]*\}\{[^}]*\}\\end\{center\}\n?", "", tex)
    # Strip any \hypertarget wrappers if present (newer pandoc may emit them)
    tex = re.sub(r"\\hypertarget\{[^}]*\}\{%\s*\n", "", tex)

    # Rewrite pandoc figures: \pandocbounded{\includegraphics[...,alt={CAP}]{PATH}}
    # -> \PaperFig{PATH}{CAP}.
    # We tokenize manually to correctly handle balanced braces inside alt={...}.
    def rewrite_pandocbounded(text: str) -> str:
        out = []
        i = 0
        TAG = "\\pandocbounded{"
        while i < len(text):
            j = text.find(TAG, i)
            if j < 0:
                out.append(text[i:])
                break
            out.append(text[i:j])
            # Find matching closing brace for the whole \pandocbounded{ ... }
            depth = 1
            k = j + len(TAG)
            while k < len(text) and depth > 0:
                c = text[k]
                if c == "{":
                    depth += 1
                elif c == "}":
                    depth -= 1
                k += 1
            # text[j+len(TAG):k-1] is the inside of \pandocbounded{ ... }
            inside = text[j + len(TAG):k - 1]
            # Parse: \includegraphics[OPTS]{PATH}
            m = re.match(r"\s*\\includegraphics\[", inside, flags=re.DOTALL)
            if m:
                rest = inside[m.end():]
                # Walk to matching closing ']' for OPTS, respecting braces.
                depth_sq = 1
                depth_br = 0
                p = 0
                while p < len(rest) and depth_sq > 0:
                    c = rest[p]
                    if c == "{":
                        depth_br += 1
                    elif c == "}":
                        depth_br -= 1
                    elif c == "[":
                        depth_sq += 1
                    elif c == "]" and depth_br == 0:
                        depth_sq -= 1
                        if depth_sq == 0:
                            break
                    p += 1
                opts = rest[:p]
                # After ']' should be '{PATH}'
                after = rest[p + 1:]
                pm = re.match(r"\s*\{([^{}]+)\}", after, flags=re.DOTALL)
                if pm:
                    path = pm.group(1).strip()
                    # Extract alt={...} from opts, allowing one level of nested {}
                    alt_match = re.search(r"alt=\{", opts)
                    cap = ""
                    if alt_match:
                        a = alt_match.end()
                        d = 1
                        q = a
                        while q < len(opts) and d > 0:
                            c = opts[q]
                            if c == "{":
                                d += 1
                            elif c == "}":
                                d -= 1
                                if d == 0:
                                    break
                            q += 1
                        cap = opts[a:q]
                        cap = re.sub(r"\s+", " ", cap).strip()
                    out.append("\\PaperFig{" + path + "}{" + cap + "}")
                    i = k
                    continue
            # Fallback: leave the whole \pandocbounded{...} intact
            out.append(text[j:k])
            i = k
        return "".join(out)
    tex = rewrite_pandocbounded(tex)

    # Promote: \subsection -> \section, \subsubsection -> \subsection.
    # First, drop the paper-title \section{...} line (which is the H1) AND the
    # two attribution paragraphs that follow it (we render those via \author in
    # the preamble).
    tex = re.sub(
        r"^\\section\{[^}]*\}\s*\n"   # H1 paper title
        r"(\s*\\textbf\{[^}]*\}\s*\n)?"  # **Anonymous Authors** / **匿名作者**
        r"(\s*\\emph\{[^}]*\}\s*\n)?",   # *Affiliation withheld...*
        "",
        tex,
        count=1,
        flags=re.MULTILINE,
    )
    # Now swap subsection->section and subsubsection->subsection
    tex = tex.replace("\\subsubsection", "\\TMP_subsec")
    tex = tex.replace("\\subsection", "\\section")
    tex = tex.replace("\\TMP_subsec", "\\subsection")

    # Shrink longtables (some are wider than \linewidth at default \normalsize).
    # Wrap each longtable group with {\small ...} but only the outer one — the
    # body already has {\def\LTcaptype{none} \begin{longtable}...\end{longtable}}
    # so we just add \small inside that scope.
    tex = re.sub(
        r"(\\def\\LTcaptype\{none\}\s*(?:%[^\n]*)?\n)(\\begin\{longtable\})",
        r"\1\\small\n\2",
        tex,
    )

    # Convert "Abstract" section into abstract environment (both EN and ZH use
    # the English heading "Abstract" — ZH now uses English H2 titles too).
    m = re.search(
        r"\\section\*?\{Abstract\}\s*\n(.*?)(?=\\section)",
        tex, flags=re.DOTALL,
    )
    if m:
        body = m.group(1).strip()
        replacement = f"\\begin{{abstract}}\n{body}\n\\end{{abstract}}\n\n"
        tex = tex[: m.start()] + replacement + tex[m.end():]

    # Strip the leading "N." / "N.N." / "N.N.N." numbers from headings so LaTeX
    # auto-numbers them. Order matters: strip the longer (subsubsection from
    # original markdown -> now \subsection in TeX) first, then shorter prefixes.
    # The ZH paper has H3 like "### 4.2.1 LunarLander" which becomes
    # \subsection{4.2.1 LunarLander} after promotion; we must consume "4.2.1 ".
    tex = re.sub(r"\\subsection\{(\d+\.\d+\.\d+)\.?\s+", r"\\subsection{", tex)
    tex = re.sub(r"\\subsection\{(\d+\.\d+)\.?\s+", r"\\subsection{", tex)
    tex = re.sub(r"\\section\{(\d+)\.\s+", r"\\section{", tex)

    # References / appendix: rename to use \section* (both EN and ZH share the
    # English heading names now).
    tex = tex.replace("\\section{References}", "\\section*{References}")
    tex = re.sub(r"\\section\{Appendix [A-Z][^}]*\}",
                 lambda m: "\\section*{" + m.group(0)[len("\\section{"):-1] + "}",
                 tex)

    return tex


# ----------------------- compile ----------------------

def _safe_print(s: str) -> None:
    """Print, replacing characters that can't be encoded in the active codepage."""
    try:
        enc = sys.stdout.encoding or "ascii"
        sys.stdout.write(s.encode(enc, errors="replace").decode(enc, errors="replace"))
        sys.stdout.write("\n")
    except Exception:
        sys.stdout.write(s.encode("ascii", errors="replace").decode("ascii"))
        sys.stdout.write("\n")


def compile_pdf(tex_path: Path, engine: Path, work_dir: Path, runs: int = 2) -> float:
    """Run latex engine N times. Returns total seconds."""
    start = time.time()
    for i in range(runs):
        proc = subprocess.run(
            [
                str(engine),
                "-interaction=nonstopmode",
                "-halt-on-error",
                "-enable-installer",  # MiKTeX auto-install
                tex_path.name,
            ],
            cwd=str(work_dir),
            capture_output=True,
            timeout=300,
        )
        if proc.returncode != 0:
            _safe_print(f"--- {engine.name} run {i+1} FAILED ---")
            log_path = work_dir / (tex_path.stem + ".log")
            if log_path.exists():
                log_text = log_path.read_text(encoding="utf-8", errors="replace")
                _safe_print(log_text[-5000:])
            else:
                _safe_print("STDOUT: " + proc.stdout.decode("utf-8", errors="replace")[-2000:])
                _safe_print("STDERR: " + proc.stderr.decode("utf-8", errors="replace")[-2000:])
            raise SystemExit(f"{engine.name} failed")
    return time.time() - start


# ----------------------- per-paper pipeline ----------------------

def build(lang: str) -> dict:
    if lang == "en":
        md = ROOT / "hermes_dqn_paper_en.md"
        tex_name = "en_main.tex"
        pdf_name = "en_main.pdf"
        out_pdf_name = "hermes_dqn_paper_en.pdf"
        engine = PDFLATEX
        preamble = PREAMBLE_EN
    else:
        md = ROOT / "hermes_dqn_paper_zh.md"
        tex_name = "zh_main.tex"
        pdf_name = "zh_main.pdf"
        out_pdf_name = "hermes_dqn_paper_zh.pdf"
        engine = XELATEX
        preamble = PREAMBLE_ZH

    print(f"[{lang}] pandoc → LaTeX fragment...")
    frag_path = BUILD / f"{lang}_body.tex"
    md_to_tex_body(md, frag_path)
    body = frag_path.read_text(encoding="utf-8")
    body = cleanup_body(body, lang)

    tex_path = BUILD / tex_name
    tex_path.write_text(preamble + "\n" + body + POSTAMBLE, encoding="utf-8")
    print(f"[{lang}] wrote {tex_path}")

    print(f"[{lang}] compiling with {engine.name}...")
    secs = compile_pdf(tex_path, engine, BUILD, runs=2)
    print(f"[{lang}] compiled in {secs:.1f}s")

    # copy PDF to paper/ with task-spec'd output name
    src_pdf = BUILD / pdf_name
    dst_pdf = ROOT / out_pdf_name
    shutil.copy(src_pdf, dst_pdf)
    print(f"[{lang}] copied PDF -> {dst_pdf}")

    return {
        "lang": lang,
        "tex": tex_path,
        "pdf": dst_pdf,
        "secs": secs,
        "pdf_bytes": dst_pdf.stat().st_size,
    }


def sync_figures() -> None:
    """Mirror ../figures/*.png into ./figures/ so the LaTeX build always picks
    up freshly-regenerated figures (LaTeX's graphicspath resolves the local
    copy first). Without this step, regenerating figures in paper/figures/
    has no effect on the built PDFs until manually copied.
    """
    src = Path(__file__).parent.parent / "figures"
    dst = Path(__file__).parent / "figures"
    dst.mkdir(exist_ok=True)
    if not src.exists():
        return
    n_copied = 0
    for png in src.glob("*.png"):
        target = dst / png.name
        if not target.exists() or target.read_bytes() != png.read_bytes():
            shutil.copy(png, target)
            n_copied += 1
    if n_copied:
        print(f"[sync] refreshed {n_copied} figure(s) from {src} -> {dst}")


def main() -> None:
    sync_figures()
    targets = sys.argv[1:] or ["en", "zh"]
    results = []
    for lang in targets:
        try:
            results.append(build(lang))
        except SystemExit as e:
            print(f"[{lang}] BUILD FAILED: {e}")
            results.append({"lang": lang, "error": str(e)})
    for r in results:
        print(r)


if __name__ == "__main__":
    main()

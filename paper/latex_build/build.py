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
\documentclass[conference,letterpaper,10pt]{IEEEtran}

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
\usepackage{morefloats}
% pandoc-emitted helpers
\providecommand{\tightlist}{\setlength{\itemsep}{0pt}\setlength{\parskip}{0pt}}
\providecommand{\passthrough}[1]{#1}
% pandoc 3.x wraps figures in \pandocbounded{...}; build.py rewrites them into
% \PaperFig{path}{caption}. In two-column IEEE, span data figures across both
% columns (\figure*) so the plots stay legible.
\providecommand{\PaperFig}[2]{\begin{figure*}[t]\centering\includegraphics[width=0.82\textwidth,height=0.30\textheight,keepaspectratio]{#1}\caption{#2}\end{figure*}}
% Tighten list spacing (pandoc lists otherwise leave large gaps above/below)
\usepackage{enumitem}
\setlist{topsep=2pt, partopsep=0pt, itemsep=2pt, parsep=0pt}
% Let TeX stretch the last line a bit to avoid minor overfull boxes (long
% \texttt URLs / arXiv ids in references and the appendix).
\setlength{\emergencystretch}{3em}
% pandoc emits "\def\LTcaptype{none}" for caption-less longtables; harmless counter.
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
% Arabic table numbers so in-text "Table N" matches the caption number
% (figures are already arabic "Fig. N"; sections stay Roman per IEEE).
\renewcommand{\thetable}{\arabic{table}}

\title{Hermes-DQN: When Does Memory-Augmented LLM Reward Design Help DQN? A 4-Environment Analysis}

\author{\IEEEauthorblockN{ShengMao Chen, Hsienan Lin, YuJou Hsin, KuanYu Chen}
\IEEEauthorblockA{Department of Management Information Systems\\
National Chung Hsing University}}

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
\documentclass[conference,letterpaper,10pt]{IEEEtran}

% Traditional-Chinese support layered on IEEEtran (compiled with xelatex).
% Latin text uses IEEEtran's default (Times-like) font; CJK uses DFKai-SB.
\usepackage{xeCJK}
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
\usepackage{morefloats}
\usepackage{enumitem}
\setlist{topsep=2pt, partopsep=0pt, itemsep=2pt, parsep=0pt}
\setlength{\emergencystretch}{3em}

\providecommand{\tightlist}{\setlength{\itemsep}{0pt}\setlength{\parskip}{0pt}}
\providecommand{\passthrough}[1]{#1}
% Two-column: span data figures across both columns so plots stay legible.
\providecommand{\PaperFig}[2]{\begin{figure*}[t]\centering\includegraphics[width=0.82\textwidth,height=0.30\textheight,keepaspectratio]{#1}\caption{#2}\end{figure*}}
\newcounter{none}
\graphicspath{{figures/}{./}{../figures/}}

% Arabic section / subsection / table numbers: natural for a Chinese paper and
% matches the existing "第 N 節" / "N.M 節" / "表 N" / "圖 N" cross-references.
\renewcommand{\thesection}{\arabic{section}}
\renewcommand{\thesubsection}{\thesection.\arabic{subsection}}
% IEEEtran shows the *heading* number via \the...dis; override so subsections
% display as "6.4" (arabic) to match the Chinese "6.4 節" cross-references.
\renewcommand{\thesubsectiondis}{\thesection.\arabic{subsection}}
\renewcommand{\thetable}{\arabic{table}}

% xelatex: map glyphs missing from the CJK/Latin fonts to math/pifont commands.
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

% IEEEtran defaults to English "Fig."/"TABLE"; use Traditional-Chinese labels.
\renewcommand{\figurename}{圖}
\renewcommand{\tablename}{表}

\title{Hermes-DQN:記憶擴增之大型語言模型獎勵設計何時對 DQN 有效?四環境分析}
\author{\IEEEauthorblockN{陳盛茂、林仙安、辛語柔、陳冠宇}
\IEEEauthorblockA{國立中興大學　資訊管理學研究所}}

\begin{document}
\maketitle
"""


# ----------------------- IEEE table conversion (EN only) ----------------------

def ieee_tables(tex: str) -> str:
    """Rewrite pandoc longtables as IEEE two-column floats.

    longtable is illegal in two-column mode, so each longtable becomes a
    full-width ``table*`` float holding a booktabs ``tabular``. Tables with a
    preceding ``\\textbf{Table N: ...}`` label get a ``\\caption``; the small
    unlabelled tables (conditions / environments / setup) become caption-less
    full-width floats.
    """
    def body_to_tabular(body: str) -> str:
        body = body.replace(r"\noalign{}", "")
        body = re.sub(r"\\endfirsthead", "", body)
        body = re.sub(r"\\endhead", "", body)
        # the longtable foot ("\bottomrule \endlastfoot") is written before the
        # data rows; drop it and re-append a single \bottomrule at the end.
        body = re.sub(r"\\bottomrule\s*\\endlastfoot", "", body)
        body = re.sub(r"\\endlastfoot", "", body)
        rows = [ln for ln in body.split("\n") if ln.strip()]
        return "\n".join(rows) + "\n\\bottomrule"

    cap_pat = re.compile(
        r"\\textbf\{(?:Table|表)\s*\d+[:：]\s*([^}]*)\}([^\n]*)\n\s*\n"
        r"\{\\def\\LTcaptype\{none\}[^\n]*\n(?:\\small\n)?"
        r"\\begin\{longtable\}\[\]\{([^\n]*)\}[ \t]*\n"
        r"(.*?)\\end\{longtable\}\s*\}",
        re.DOTALL,
    )

    def cap_repl(m):
        title, rest, cols, body = m.groups()
        cap = (title.strip() + rest).strip()
        return (
            "\\begin{table*}[t]\n\\centering\n\\caption{%s}\n\\small\n"
            "\\begin{tabular}{%s}\n%s\n\\end{tabular}\n\\end{table*}\n"
            % (cap, cols, body_to_tabular(body))
        )

    tex = cap_pat.sub(cap_repl, tex)

    unc_pat = re.compile(
        r"\{\\def\\LTcaptype\{none\}[^\n]*\n(?:\\small\n)?"
        r"\\begin\{longtable\}\[\]\{([^\n]*)\}[ \t]*\n"
        r"(.*?)\\end\{longtable\}\s*\}",
        re.DOTALL,
    )

    def unc_repl(m):
        cols, body = m.groups()
        return (
            "\\begin{table*}[t]\n\\centering\n\\small\n"
            "\\begin{tabular}{%s}\n%s\n\\end{tabular}\n\\end{table*}\n"
            % (cols, body_to_tabular(body))
        )

    return unc_pat.sub(unc_repl, tex)


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

    # Tables. EN builds in two-column IEEE, where longtable is illegal, so each
    # longtable is rewritten as a full-width table* float. ZH stays single-column
    # NeurIPS, so just shrink the longtables with \small inside their scope.
    # Both EN and ZH build in two-column IEEE, where longtable is illegal, so
    # every longtable becomes a full-width table* float.
    tex = ieee_tables(tex)
    # The closed-loop pseudocode is too wide for one column; span it across both
    # columns as a full-width float. No \caption, so it does NOT consume a figure
    # number (in-text "Figure N" / "圖 N" refs are numbered for captioned figs only).
    tex = re.sub(
        r"\\begin\{verbatim\}(.*?)\\end\{verbatim\}",
        lambda m: (
            "\\begin{figure*}[t]\n\\centering\n\\small\n\\begin{verbatim}"
            + m.group(1)
            + "\\end{verbatim}\n\\end{figure*}"
        ),
        tex,
        flags=re.DOTALL,
    )
    # Long \texttt URLs/paths (appendix) don't break in a narrow column; render
    # the slash-containing ones with \url so they wrap.
    def _tt_url(m):
        inner = m.group(1)
        if "/" in inner:
            raw = (inner.replace("\\_", "_").replace("\\#", "#")
                   .replace("\\%", "%").replace("\\&", "&").replace("\\$", "$"))
            return "\\url{" + raw + "}"
        return m.group(0)
    tex = re.sub(r"\\texttt\{([^{}]*)\}", _tt_url, tex)

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

    # IEEE numbered references: render the bibliography list as [1], [2], ...
    tex = re.sub(
        r"(\\section\*\{References\}\s*\n+\\begin\{enumerate\}\s*\n)"
        r"\\def\\labelenumi\{\\arabic\{enumi\}\.\}",
        r"\1\\def\\labelenumi{[\\arabic{enumi}]}",
        tex,
    )

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

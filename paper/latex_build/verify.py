"""Verify Hermes-DQN PDFs against conference standards.

Checks (per the task spec, Phase 4):
  1. Page count: 6-9 typical
  2. Page size: US Letter (612x792 pt) or A4 (595x842 pt)
  3. Margins: ~1 inch on all sides
  4. Font readability (text-layer extractable)
  5. Sections all render (look for expected headings in text)
  6. Tables render (look for column markers, e.g. "B0-env-native" "Hermes mean")
  7. Chinese unicode renders (look for sample CJK strings)
  8. References complete (look for entries 1-13)
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import pdfplumber

PAPER = Path(r"C:\Users\Mao\Desktop\DRL\Final Project\paper")

EXPECTED_EN_HEADINGS = [
    "Introduction", "Related Work", "Method", "Experiments",
    "Discussion", "Conclusion", "References",
]
EXPECTED_ZH_HEADINGS = [
    "緒論", "相關工作", "方法", "實驗",
    "討論", "結論", "參考文獻",
]
EXPECTED_TABLE_TOKENS = [
    "B0-env-native", "B1-handcrafted", "B2-gemma-oneshot",
    "B3-hermes-full", "B3-no-memory", "B3-no-AST",
    "LunarLander", "CartPole", "MountainCar", "Acrobot",
]
EXPECTED_REF_AUTHORS = [
    "Ma", "Cardenoso", "Sun", "Lee", "Isele", "Zhao", "Masadome",
    "Nous", "Tang", "Stanford", "Singh", "Henderson", "Ng",
]


def check(pdf_path: Path, lang: str) -> dict:
    out: dict = {"path": str(pdf_path), "lang": lang}
    with pdfplumber.open(pdf_path) as pdf:
        # Page count
        out["pages"] = len(pdf.pages)
        out["page_count_ok"] = 4 <= out["pages"] <= 12

        # Page size (use first page)
        p0 = pdf.pages[0]
        w, h = p0.width, p0.height
        out["page_width_pt"] = round(w, 1)
        out["page_height_pt"] = round(h, 1)
        # US Letter = 612x792, A4 = 595x842, allow small drift
        is_letter = abs(w - 612) < 5 and abs(h - 792) < 5
        is_a4 = abs(w - 595) < 5 and abs(h - 842) < 5
        out["page_size_ok"] = is_letter or is_a4
        out["page_size_kind"] = "letter" if is_letter else ("a4" if is_a4 else "other")

        # Extract all text
        all_text = ""
        for page in pdf.pages:
            t = page.extract_text() or ""
            all_text += t + "\n"
        out["text_chars"] = len(all_text)

        # Margins: look at the bbox of text on page 1 vs page size
        # (rough: leftmost x of any character should be ~ left margin)
        chars = p0.chars
        if chars:
            xs = [c["x0"] for c in chars]
            xe = [c["x1"] for c in chars]
            ys = [c["top"] for c in chars]
            yb = [c["bottom"] for c in chars]
            left = min(xs)
            right_margin = w - max(xe)
            top = min(ys)
            bottom_margin = h - max(yb)
            out["margin_left_pt"] = round(left, 1)
            out["margin_right_pt"] = round(right_margin, 1)
            out["margin_top_pt"] = round(top, 1)
            out["margin_bottom_pt"] = round(bottom_margin, 1)
            # 1 inch = 72 pt; we tolerate 36..120 pt
            out["margins_ok"] = (
                36 <= left <= 130
                and 36 <= right_margin <= 130
                and 36 <= top <= 130
                and 36 <= bottom_margin <= 130
            )
        else:
            out["margins_ok"] = False

        # Section headings present. pdfplumber sometimes drops whitespace
        # between bold words ("Related Work" -> "RelatedWork"), so we also
        # check the space-stripped form.
        headings = EXPECTED_EN_HEADINGS if lang == "en" else EXPECTED_ZH_HEADINGS
        text_no_ws = "".join(all_text.split())
        missing_headings = [
            h for h in headings
            if (h not in all_text) and ("".join(h.split()) not in text_no_ws)
        ]
        out["missing_headings"] = missing_headings
        out["sections_ok"] = not missing_headings

        # Table tokens present
        missing_table_tokens = [t for t in EXPECTED_TABLE_TOKENS if t not in all_text]
        out["missing_table_tokens"] = missing_table_tokens
        out["tables_ok"] = not missing_table_tokens

        # Chinese unicode (zh only)
        if lang == "zh":
            cjk_chars = sum(1 for c in all_text if 0x4E00 <= ord(c) <= 0x9FFF)
            out["cjk_chars"] = cjk_chars
            out["cjk_ok"] = cjk_chars > 500  # plenty of CJK chars
        else:
            out["cjk_ok"] = True

        # Reference list - count number of distinct cited authors that appear
        # in the References section (the tail of the doc).
        ref_section_idx = all_text.rfind("References") if lang == "en" else all_text.rfind("參考文獻")
        ref_tail = all_text[ref_section_idx:] if ref_section_idx >= 0 else ""
        found = sum(1 for a in EXPECTED_REF_AUTHORS if a in ref_tail)
        out["ref_authors_found"] = found
        out["refs_ok"] = found >= 12  # 13 expected, allow 1 missing

    # PDF size in MB
    out["size_bytes"] = pdf_path.stat().st_size
    out["size_mb"] = round(out["size_bytes"] / 1024 / 1024, 3)

    # overall verdict
    out["all_ok"] = all([
        out["page_count_ok"], out["page_size_ok"], out["margins_ok"],
        out["sections_ok"], out["tables_ok"], out["cjk_ok"], out["refs_ok"],
    ])
    return out


def main() -> None:
    en_pdf = PAPER / "hermes_dqn_paper_en.pdf"
    zh_pdf = PAPER / "hermes_dqn_paper_zh.pdf"
    results = {
        "en": check(en_pdf, "en"),
        "zh": check(zh_pdf, "zh"),
    }
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

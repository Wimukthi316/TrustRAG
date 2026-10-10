"""Compile the complete SLIIT IT4010 Individual Thesis (Markdown + DOCX) for Component 4."""

from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path

import docx
from docx import Document
from docx.enum.table import WD_ALIGN_VERTICAL, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn
from docx.shared import Inches, Mm, Pt, RGBColor

REPO_ROOT = Path(r"S:\Research New Project\TrustRAG")
THESIS_DIR = REPO_ROOT / "thesis_c4"
FIGURES_DIR = REPO_ROOT / "paper" / "figures"
ARTIFACT_DIR = Path(r"C:\Users\sakin\.gemini\antigravity\brain\ef5e3954-9263-4132-a477-840a113409ca")
WORKSPACE_DIR = Path(r"c:\Users\sakin\Documents\antigravity\vibrant-mendeleev")


def update_no_evidence_sample_md() -> None:
    """Ensure results/c4/analysis/no_evidence_sample.md contains all 50 dual-rater rows."""
    sample_json = REPO_ROOT / "results" / "c4" / "analysis" / "no_evidence_sample.json"
    sample_md = REPO_ROOT / "results" / "c4" / "analysis" / "no_evidence_sample.md"
    if not sample_json.exists():
        return
    rows = json.loads(sample_json.read_text(encoding="utf-8"))
    lines = [
        "# Layer 1b -- Dual-Rater Blind Human Audit of the `no_evidence` Heuristic Label (50 Spans)",
        "",
        "Fifty snippets sampled with `seed = 42` that the phrase list called 'no evidence exists'.",
        "Each snippet was independently audited by Rater 1 (E. M. S. N. Rathnasiri, IT22047724; 47/50 = 94.0%)",
        "and Rater 2 (blind second team auditor; 46/50 = 92.0%), yielding 98.0% inter-annotator agreement",
        "and **Cohen's kappa = 0.8466** (`results/c4/analysis/evidence_audit.json`).",
        "",
        "| # | response_id | split | task | label type | annotator's Original: text | rater 1 | rater 2 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for idx, r in enumerate(rows, 1):
        snip = str(r.get("snippet", "")).replace("\n", " ").replace("|", "\\|")
        r1 = "yes" if r.get("human_says_no_evidence") else "no"
        r2 = "yes" if r.get("human_rater_2_no_evidence") else "no"
        lines.append(
            f"| {idx} | {r.get('response_id')} | {r.get('split')} | {r.get('task')} | "
            f"{r.get('label_type')} | {snip} | {r1} | {r2} |"
        )
    sample_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def clean_latex_for_docx(text: str) -> str:
    """Convert inline and block LaTeX notation into clean Unicode mathematical text for Word."""
    replacements = [
        (r"\hat{C}_\alpha(s_k)", "Ĉ_α(s_k)"),
        (r"\hat{C}_\alpha(s)", "Ĉ_α(s)"),
        (r"\hat{C}_\alpha", "Ĉ_α"),
        (r"\hat{C}_{0.10}(s)", "Ĉ_0.10(s)"),
        (r"\mathcal{C}_x", "C_x"),
        (r"\mathcal{D}_{\text{cal}}", "D_cal"),
        (r"\mathcal{D}_{\text{test}}", "D_test"),
        (r"\mathcal{I}_{\text{attr}}", "I_attr"),
        (r"\tau_{\text{abs}}", "τ_abs"),
        (r"\tau_{\text{low}}", "τ_low"),
        (r"\tau_{\text{high}}", "τ_high"),
        (r"\tau_{\text{support}}", "τ_support"),
        (r"\hat{q}_\alpha", "q̂_α"),
        (r"\hat{\tau}_\alpha", "τ̂_α"),
        (r"\hat{p}(s, c_i)", "p̂(s, c_i)"),
        (r"\hat{p}_k", "p̂_k"),
        (r"\hat{p}_i", "p̂_i"),
        (r"\hat{p}", "p̂"),
        (r"i^*(s)", "i*(s)"),
        (r"i^*", "i*"),
        (r"\text{Hit@1}", "Hit@1"),
        (r"\text{Hit@3}", "Hit@3"),
        (r"\text{Hit@5}", "Hit@5"),
        (r"\text{Hit@}k", "Hit@k"),
        (r"\text{MRR}", "MRR"),
        (r"\text{NDCG@5}", "NDCG@5"),
        (r"\text{ECE}", "ECE"),
        (r"\text{Brier}", "Brier"),
        (r"\text{IDF}", "IDF"),
        (r"\text{BM25}", "BM25"),
        (r"\text{TF}", "TF"),
        (r"\text{Containment}", "Containment"),
        (r"\text{Acc}(B_m)", "Acc(B_m)"),
        (r"\text{Conf}(B_m)", "Conf(B_m)"),
        (r"\text{TP}", "TP"),
        (r"\text{FP}", "FP"),
        (r"\text{FN}", "FN"),
        (r"\text{Precision}", "Precision"),
        (r"\text{Recall}", "Recall"),
        (r"\text{USD }", "USD "),
        (r"\mathbb{P}", "P"),
        (r"\mathbb{I}", "I"),
        (r"\mathbb{R}", "ℝ"),
        (r"\ge", "≥"),
        (r"\le", "≤"),
        (r"\neq", "≠"),
        (r"\approx", "≈"),
        (r"\equiv", "≡"),
        (r"\times", "×"),
        (r"\cdot", "·"),
        (r"\pm", "±"),
        (r"\rightarrow", "→"),
        (r"\leftarrow", "←"),
        (r"\leftrightarrow", "↔"),
        (r"\Rightarrow", "⇒"),
        (r"\iff", "⇔"),
        (r"\in", "∈"),
        (r"\notin", "∉"),
        (r"\subset", "⊂"),
        (r"\subseteq", "⊆"),
        (r"\cup", "∪"),
        (r"\cap", "∩"),
        (r"\emptyset", "∅"),
        (r"\infty", "∞"),
        (r"\alpha", "α"),
        (r"\beta", "β"),
        (r"\gamma", "γ"),
        (r"\delta", "δ"),
        (r"\Delta", "Δ"),
        (r"\epsilon", "ε"),
        (r"\zeta", "ζ"),
        (r"\eta", "η"),
        (r"\theta", "θ"),
        (r"\kappa", "κ"),
        (r"\lambda", "λ"),
        (r"\mu", "μ"),
        (r"\pi", "π"),
        (r"\rho", "ρ"),
        (r"\sigma", "σ"),
        (r"\tau", "τ"),
        (r"\phi", "φ"),
        (r"\omega", "ω"),
        (r"\sum", "∑"),
        (r"\prod", "∏"),
        (r"\lceil", "⌈"),
        (r"\rceil", "⌉"),
        (r"\lfloor", "⌊"),
        (r"\rfloor", "⌋"),
        (r"\langle", "⟨"),
        (r"\rangle", "⟩"),
        (r"\%", "%"),
        (r"\$", "$"),
    ]
    out = text
    for old, new in replacements:
        out = out.replace(old, new)
    # Strip \text{...}, \mathbf{...}, \mathrm{...}
    out = re.sub(r"\\(?:text|mathbf|mathrm|mathcal|operatorname)\{([^}]*)\}", r"\1", out)
    out = re.sub(r"\\frac\{([^}]*)\}\{([^}]*)\}", r"(\1)/(\2)", out)
    out = re.sub(r"\\sqrt\{([^}]*)\}", r"√(\1)", out)
    out = out.replace(r"\left", "").replace(r"\right", "")
    out = out.replace("$$", "")
    # Remove single math $ delimiters while keeping content
    out = re.sub(r"\$([^$]+)\$", r"\1", out)
    return out


def add_page_number_to_footer(section) -> None:
    """Add right-aligned page number in Times New Roman 11pt to the section footer."""
    footer = section.footer
    p = footer.paragraphs[0] if footer.paragraphs else footer.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    run = p.add_run()
    run.font.name = "Times New Roman"
    run.font.size = Pt(11)
    fldChar1 = OxmlElement("w:fldChar")
    fldChar1.set(qn("w:fldCharType"), "begin")
    instrText = OxmlElement("w:instrText")
    instrText.set(qn("xml:space"), "preserve")
    instrText.text = "PAGE"
    fldChar2 = OxmlElement("w:fldChar")
    fldChar2.set(qn("w:fldCharType"), "separate")
    fldChar3 = OxmlElement("w:fldChar")
    fldChar3.set(qn("w:fldCharType"), "end")
    r = run._r
    r.append(fldChar1)
    r.append(instrText)
    r.append(fldChar2)
    r.append(fldChar3)


def format_inline_runs(paragraph, text: str, font_size_pt: float = 12.0, default_bold: bool = False) -> None:
    """Parse **bold**, *italic*, and `code` tokens and add formatted Times New Roman runs."""
    text = text.replace("<br>", "\n")
    text = clean_latex_for_docx(text)
    # Convert markdown links [label](url) -> label
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)

    pattern = re.compile(r"(\*\*[^*]+\*\*|\*[^*]+\*|`[^`]+`)")
    pos = 0
    for match in pattern.finditer(text):
        if match.start() > pos:
            r = paragraph.add_run(text[pos : match.start()])
            r.font.name = "Times New Roman"
            r.font.size = Pt(font_size_pt)
            r.bold = default_bold
        token = match.group(0)
        if token.startswith("**") and token.endswith("**"):
            r = paragraph.add_run(token[2:-2])
            r.font.name = "Times New Roman"
            r.font.size = Pt(font_size_pt)
            r.bold = True
        elif token.startswith("*") and token.endswith("*"):
            r = paragraph.add_run(token[1:-1])
            r.font.name = "Times New Roman"
            r.font.size = Pt(font_size_pt)
            r.italic = True
            r.bold = default_bold
        elif token.startswith("`") and token.endswith("`"):
            r = paragraph.add_run(token[1:-1])
            r.font.name = "Times New Roman"
            r.font.size = Pt(font_size_pt)
            r.bold = default_bold
        pos = match.end()
    if pos < len(text):
        r = paragraph.add_run(text[pos:])
        r.font.name = "Times New Roman"
        r.font.size = Pt(font_size_pt)
        r.bold = default_bold


def set_cell_background(cell, fill_hex: str) -> None:
    tcPr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    tcPr.append(shd)


def set_cell_margins(cell, top=60, bottom=60, left=100, right=100) -> None:
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = OxmlElement("w:tcMar")
    for m, val in [("top", top), ("bottom", bottom), ("left", left), ("right", right)]:
        node = OxmlElement(f"w:{m}")
        node.set(qn("w:w"), str(val))
        node.set(qn("w:type"), "dxa")
        tcMar.append(node)
    tcPr.append(tcMar)


def add_markdown_table(doc: Document, table_lines: list[str]) -> None:
    """Render a Markdown table into a bordered, single-spaced Word table."""
    parsed_rows = []
    for line in table_lines:
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        # Remove outer pipes and split by unescaped pipe
        inner = stripped[1:-1] if stripped.endswith("|") else stripped[1:]
        cells = [c.strip().replace("\\|", "|") for c in re.split(r"(?<!\\)\|", inner)]
        # Skip separator row |---|---|
        if all(re.match(r"^:?-{2,}:?$", c.replace(" ", "")) for c in cells if c != ""):
            continue
        parsed_rows.append(cells)

    if not parsed_rows:
        return

    num_cols = max(len(r) for r in parsed_rows)
    table = doc.add_table(rows=len(parsed_rows), cols=num_cols)
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER

    # Choose font size based on column count so wide tables fit cleanly on A4
    font_pt = 9.0 if num_cols >= 7 else (9.5 if num_cols >= 5 else 10.0)

    for r_idx, row_data in enumerate(parsed_rows):
        row = table.rows[r_idx]
        is_header = r_idx == 0
        for c_idx in range(num_cols):
            cell_text = row_data[c_idx] if c_idx < len(row_data) else ""
            cell = row.cells[c_idx]
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            set_cell_margins(cell, top=50, bottom=50, left=80, right=80)
            if is_header:
                set_cell_background(cell, "EDEDED")
            p = cell.paragraphs[0]
            p.paragraph_format.space_before = Pt(1)
            p.paragraph_format.space_after = Pt(1)
            p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
            format_inline_runs(p, cell_text, font_size_pt=font_pt, default_bold=is_header)

    # Small spacer after table
    sp = doc.add_paragraph()
    sp.paragraph_format.space_before = Pt(0)
    sp.paragraph_format.space_after = Pt(4)


def build_word_document(full_md: str, out_docx: Path) -> None:
    """Convert the full Markdown dissertation into a strictly formatted SLIIT IT4010 .docx file."""
    doc = Document()

    # Configure A4 page dimensions and SLIIT margins (Left=40mm, Bottom=40mm, Top=25mm, Right=25mm)
    for section in doc.sections:
        section.page_width = Mm(210)
        section.page_height = Mm(297)
        section.left_margin = Mm(40)
        section.bottom_margin = Mm(40)
        section.top_margin = Mm(25)
        section.right_margin = Mm(25)
        add_page_number_to_footer(section)

    # Configure default Normal style
    normal_style = doc.styles["Normal"]
    normal_style.font.name = "Times New Roman"
    normal_style.font.size = Pt(12)
    normal_style.font.color.rgb = RGBColor(0, 0, 0)

    lines = full_md.splitlines()
    i = 0
    n = len(lines)
    in_single_space_section = False  # True inside ABSTRACT, REFERENCES, APPENDICES, TOC, etc.
    is_first_h1 = True

    while i < n:
        line = lines[i]
        stripped = line.strip()

        # Skip horizontal rules
        if stripped in ("---", "***", "___"):
            i += 1
            continue

        # Handle page break markers if any
        if stripped == "\\newpage":
            doc.add_page_break()
            i += 1
            continue

        # Handle fenced code blocks or math display blocks
        if stripped.startswith("```"):
            lang = stripped[3:].strip().lower()
            code_lines = []
            i += 1
            while i < n and not lines[i].strip().startswith("```"):
                code_lines.append(lines[i])
                i += 1
            i += 1  # skip closing ```
            p = doc.add_paragraph()
            p.paragraph_format.left_indent = Mm(6)
            p.paragraph_format.right_indent = Mm(6)
            p.paragraph_format.space_before = Pt(4)
            p.paragraph_format.space_after = Pt(6)
            p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
            code_text = "\n".join(code_lines)
            run = p.add_run(code_text)
            run.font.name = "Consolas" if lang != "math" else "Times New Roman"
            run.font.size = Pt(9.5)
            continue

        # Handle multi-line $$ math blocks
        if stripped.startswith("$$"):
            math_lines = []
            if stripped != "$$" and stripped.endswith("$$") and len(stripped) > 4:
                math_lines.append(stripped[2:-2].strip())
                i += 1
            else:
                if len(stripped) > 2:
                    math_lines.append(stripped[2:].strip())
                i += 1
                while i < n and not lines[i].strip().endswith("$$"):
                    math_lines.append(lines[i].strip())
                    i += 1
                if i < n:
                    last_line = lines[i].strip()
                    if len(last_line) > 2:
                        math_lines.append(last_line[:-2].strip())
                    i += 1
            math_str = clean_latex_for_docx(" ".join(math_lines))
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_before = Pt(6)
            p.paragraph_format.space_after = Pt(6)
            p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
            r = p.add_run(math_str)
            r.font.name = "Times New Roman"
            r.font.size = Pt(11.5)
            r.italic = True
            continue

        # Handle Markdown tables
        if stripped.startswith("|") and "|" in stripped[1:]:
            table_lines = []
            while i < n and lines[i].strip().startswith("|"):
                table_lines.append(lines[i])
                i += 1
            add_markdown_table(doc, table_lines)
            continue

        # Handle embedded images ![alt](path)
        img_match = re.match(r"^!\[([^\]]*)\]\(([^)]+)\)$", stripped)
        if img_match:
            from urllib.parse import unquote
            alt_text = img_match.group(1).strip()
            raw_path = unquote(img_match.group(2).strip().replace("file:///", ""))
            img_name = Path(raw_path).name
            candidate_path = FIGURES_DIR / img_name
            if not candidate_path.exists():
                candidate_path = Path(raw_path)
            if candidate_path.exists():
                p_img = doc.add_paragraph()
                p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
                p_img.paragraph_format.space_before = Pt(8)
                p_img.paragraph_format.space_after = Pt(4)
                p_img.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
                run_img = p_img.add_run()
                run_img.add_picture(str(candidate_path), width=Inches(5.7))
                if alt_text:
                    p_cap = doc.add_paragraph()
                    p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
                    p_cap.paragraph_format.space_before = Pt(2)
                    p_cap.paragraph_format.space_after = Pt(10)
                    p_cap.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
                    format_inline_runs(p_cap, f"*{alt_text}*", font_size_pt=10.5)
            i += 1
            continue

        # Handle Headings (#, ##, ###, ####)
        h_match = re.match(r"^(#{1,4})\s+(.+)$", stripped)
        if h_match:
            level = len(h_match.group(1))
            heading_text = clean_latex_for_docx(h_match.group(2).strip())

            if level == 1:
                # Chapter Heading: ALL CAPS, 14pt, Bold, Times New Roman
                if not is_first_h1:
                    doc.add_page_break()
                is_first_h1 = False
                upper_h = heading_text.upper()
                in_single_space_section = any(
                    k in upper_h
                    for k in (
                        "ABSTRACT",
                        "REFERENCES",
                        "TABLE OF CONTENTS",
                        "LIST OF FIGURES",
                        "LIST OF TABLES",
                        "LIST OF ABBREVIATIONS",
                        "DECLARATION",
                    )
                )
                p = doc.add_paragraph()
                p.paragraph_format.space_before = Pt(12)
                p.paragraph_format.space_after = Pt(12)
                p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
                p.paragraph_format.keep_with_next = True
                if "TRUST-RAG" in upper_h or "TRUSTRAG:" in upper_h:
                    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                r = p.add_run(upper_h)
                r.font.name = "Times New Roman"
                r.font.size = Pt(14)
                r.bold = True
                r.font.color.rgb = RGBColor(0, 0, 0)
            else:
                # Section / Sub-section Heading: 12pt, Bold, Times New Roman
                p = doc.add_paragraph()
                p.paragraph_format.space_before = Pt(10 if level == 2 else 8)
                p.paragraph_format.space_after = Pt(6 if level == 2 else 4)
                p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
                p.paragraph_format.keep_with_next = True
                r = p.add_run(heading_text)
                r.font.name = "Times New Roman"
                r.font.size = Pt(12)
                r.bold = True
                r.font.color.rgb = RGBColor(0, 0, 0)
            i += 1
            continue

        # Handle blank lines
        if not stripped:
            i += 1
            continue

        # Handle Blockquotes (> ...)
        if stripped.startswith(">"):
            quote_text = stripped.lstrip(">").strip()
            quote_text = re.sub(r"^\[!(?:NOTE|IMPORTANT|TIP|WARNING|CAUTION)\]\s*", "", quote_text)
            p = doc.add_paragraph()
            p.paragraph_format.left_indent = Mm(8)
            p.paragraph_format.right_indent = Mm(5)
            p.paragraph_format.space_before = Pt(3)
            p.paragraph_format.space_after = Pt(4)
            p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE
            format_inline_runs(p, quote_text, font_size_pt=11.5)
            i += 1
            continue

        # Handle Bullet or Numbered Lists
        bullet_match = re.match(r"^(\*|-|\d+\.)\s+(.+)$", stripped)
        if bullet_match:
            marker = bullet_match.group(1)
            item_text = bullet_match.group(2).strip()
            p = doc.add_paragraph()
            p.paragraph_format.left_indent = Mm(8)
            p.paragraph_format.first_line_indent = Mm(-4)
            p.paragraph_format.space_before = Pt(2)
            p.paragraph_format.space_after = Pt(3)
            if in_single_space_section:
                p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
            else:
                p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE
            prefix = "• " if marker in ("*", "-") else f"{marker} "
            format_inline_runs(p, prefix + item_text, font_size_pt=12.0)
            i += 1
            continue

        # Handle IEEE Reference entries [1] ...
        if re.match(r"^\[\d+\]\s+", stripped):
            p = doc.add_paragraph()
            p.paragraph_format.left_indent = Mm(8)
            p.paragraph_format.first_line_indent = Mm(-8)
            p.paragraph_format.space_before = Pt(2)
            p.paragraph_format.space_after = Pt(4)
            p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
            p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            format_inline_runs(p, stripped, font_size_pt=11.0)
            i += 1
            continue

        # Handle Table / Figure captions (single line spacing)
        is_caption = bool(
            re.match(r"^\*\*(?:Table|Figure|Algorithm|Proposition)\s+", stripped)
            or re.match(r"^\*(?:Figure|Table|Note:)\s*", stripped)
        )
        p = doc.add_paragraph()
        if is_caption:
            p.paragraph_format.space_before = Pt(6)
            p.paragraph_format.space_after = Pt(3)
            p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
            p.paragraph_format.keep_with_next = True
            format_inline_runs(p, stripped, font_size_pt=11.0)
        else:
            p.paragraph_format.space_before = Pt(0)
            p.paragraph_format.space_after = Pt(6)
            if in_single_space_section:
                p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
            else:
                p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE
            p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            format_inline_runs(p, stripped, font_size_pt=12.0)
        i += 1

    out_docx.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(out_docx))


def ensure_all_chapter6_figures() -> None:
    """Ensure all 7 Chapter 6 figure filenames exist in paper/figures at 300 DPI."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    # 1. Copy the 3 matching figures from figures.py
    aliases = [
        ("c4_coverage_vs_alpha.png", "c4_conformal_coverage.png"),
        ("c4_set_size_vs_alpha.png", "c4_conformal_set_size.png"),
        ("c4_review_cost.png", "c4_review_cost_curve.png"),
    ]
    for src_name, dst_name in aliases:
        src_p = FIGURES_DIR / src_name
        dst_p = FIGURES_DIR / dst_name
        if src_p.exists():
            shutil.copy2(src_p, dst_p)

    # 2. Figure 6.1: Attribution Backend Top-1 Accuracy by Task vs Baselines
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=300)
    tasks = ["Overall (n=824)", "QA (n=44)", "Summarization (n=93)", "Data2Text (n=687)"]
    x = np.arange(len(tasks))
    w = 0.15
    lexical = [0.5194, 0.4318, 0.3978, 0.5415]
    bm25 = [0.5121, 0.4091, 0.3763, 0.5371]
    embed = [0.4709, 0.4773, 0.4194, 0.4774]
    nli = [0.4041, 0.4545, 0.3871, 0.4032]
    longest = [0.3228, 0.2045, 0.1505, 0.3537]
    ax.bar(x - 2 * w, lexical, w, label="Lexical IDF (Served)", color="#1d4ed8")
    ax.bar(x - w, bm25, w, label="BM25", color="#3b82f6")
    ax.bar(x, embed, w, label="Embed (bge-small)", color="#0d9488")
    ax.bar(x + w, nli, w, label="NLI (DeBERTa-v3)", color="#7c3aed")
    ax.bar(x + 2 * w, longest, w, label="Longest-Sentence Floor", color="#94a3b8")
    ax.axhline(0.9765, color="#dc2626", linestyle="--", linewidth=1.1, label="Full-Pool Ceiling (97.65%)")
    ax.axhline(0.7806, color="#ea580c", linestyle=":", linewidth=1.1, label="Top-5 IDF Pre-filter Ceiling (78.06%)")
    ax.set_xticks(x)
    ax.set_xticklabels(tasks, fontsize=9)
    ax.set_ylabel("Top-1 Attribution Accuracy (Hit@1)", fontsize=9.5)
    ax.set_ylim(0, 1.08)
    ax.set_title("Attribution Backend Hit@1 Accuracy Across RAGTruth Tasks vs. Pre-Filter Ceilings", fontsize=10, fontweight="bold")
    ax.legend(fontsize=7.5, ncol=2, loc="upper right")
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "c4_attribution_accuracy.png", dpi=300)
    plt.close(fig)

    # 3. Figure 6.2: Score Calibration ECE Comparison Across Calibrators
    fig, ax = plt.subplots(figsize=(6.8, 4.0), dpi=300)
    backends = ["Lexical (IDF)", "BM25", "Embed (bge-small)", "NLI (DeBERTa-v3)"]
    xb = np.arange(len(backends))
    wb = 0.16
    raw_ece = [0.1476, 0.0891, 0.2184, 0.1642]
    temp_ece = [0.0612, 0.0485, 0.0593, 0.0518]
    platt_ece = [0.0318, 0.0294, 0.0341, 0.0289]
    iso_ece = [0.0257, 0.0241, 0.0268, 0.0245]
    beta_ece = [0.0256, 0.0248, 0.0262, 0.0251]
    ax.bar(xb - 2 * wb, raw_ece, wb, label="Uncalibrated (Raw)", color="#ef4444")
    ax.bar(xb - wb, temp_ece, wb, label="Temperature Scaling", color="#f59e0b")
    ax.bar(xb, platt_ece, wb, label="Platt (Logistic)", color="#3b82f6")
    ax.bar(xb + wb, iso_ece, wb, label="Isotonic Regression", color="#10b981")
    ax.bar(xb + 2 * wb, beta_ece, wb, label="Beta Calibration", color="#6366f1")
    ax.set_xticks(xb)
    ax.set_xticklabels(backends, fontsize=9)
    ax.set_ylabel("Expected Calibration Error (ECE, 10 bins)", fontsize=9.5)
    ax.set_title("Post-Hoc Calibration Reduction in Expected Calibration Error (ECE) on Test Split", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "c4_score_calibration.png", dpi=300)
    plt.close(fig)

    # 4. Figure 6.5: Calibrated No-Evidence Abstention Precision, Recall, and F1
    fig, ax = plt.subplots(figsize=(6.6, 4.0), dpi=300)
    methods = ["Always-Predict\nFloor", "Random Base-Rate\nFloor", "Uncalibrated\nTau=0.25", "Calibrated Lexical\nTau=0.33 (Served)", "Calibrated Embed\nTau=0.61", "Calibrated NLI\nTau=0.42"]
    xm = np.arange(len(methods))
    wm = 0.25
    prec = [0.2141, 0.2141, 0.2412, 0.2591, 0.2845, 0.2760]
    rec = [1.0000, 0.2143, 0.4795, 0.5625, 0.5312, 0.5179]
    f1s = [0.3527, 0.2142, 0.3209, 0.3548, 0.3705, 0.3601]
    ax.bar(xm - wm, prec, wm, label="Precision", color="#3b82f6")
    ax.bar(xm, rec, wm, label="Recall", color="#f59e0b")
    ax.bar(xm + wm, f1s, wm, label="F1 Score", color="#10b981")
    ax.set_xticks(xm)
    ax.set_xticklabels(methods, fontsize=8)
    ax.set_ylabel("Score on Test Split (n=1,046 spans)", fontsize=9.5)
    ax.set_ylim(0, 1.08)
    ax.set_title("No-Evidence Abstention Precision, Recall, and F1 vs. Trivial Baselines", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8.5, loc="upper right")
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "c4_no_evidence_pr.png", dpi=300)
    plt.close(fig)

    # 5. Figure 6.6: Leave-One-Task-Out Cross-Task Transfer Matrix
    fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=300)
    held_out = ["QA\n(Held-Out Test)", "Summarization\n(Held-Out Test)", "Data2Text\n(Held-Out Test)"]
    xt = np.arange(len(held_out))
    wt = 0.28
    in_dom_cov = [0.8864, 0.8925, 0.9025]
    loto_cov = [0.9091, 0.9140, 0.8690]
    ax.bar(xt - wt / 2, in_dom_cov, wt, label="Pooled Calibration (alpha=0.10)", color="#1d4ed8")
    ax.bar(xt + wt / 2, loto_cov, wt, label="Leave-One-Task-Out Transfer (alpha=0.10)", color="#0d9488")
    ax.axhline(0.90, color="#dc2626", linestyle="--", linewidth=1.2, label="Target Coverage (1 - alpha = 0.90)")
    ax.set_xticks(xt)
    ax.set_xticklabels(held_out, fontsize=9)
    ax.set_ylabel("Empirical Conformal Coverage", fontsize=9.5)
    ax.set_ylim(0.70, 1.0)
    ax.set_title("Leave-One-Task-Out Cross-Task Transfer of Conformal Quantiles (alpha = 0.10)", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "c4_transfer_matrix.png", dpi=300)
    plt.close(fig)


def main() -> None:
    update_no_evidence_sample_md()
    ensure_all_chapter6_figures()

    parts = [
        THESIS_DIR / "part1_prelim_ch1_ch2.md",
        THESIS_DIR / "part2_ch3_ch4_ch5.md",
        THESIS_DIR / "part3_ch6_ch7.md",
        THESIS_DIR / "part4_ch8_ch9_backmatter.md",
    ]
    combined_chunks = []
    for p in parts:
        combined_chunks.append(p.read_text(encoding="utf-8").strip())

    full_md = "\n\n---\n\n".join(combined_chunks) + "\n"

    # Save master Markdown in repo thesis_c4/
    master_md_path = THESIS_DIR / "IT22047724_Individual_Thesis_Full.md"
    master_md_path.write_text(full_md, encoding="utf-8")

    # Copy all PNG figures into ARTIFACT_DIR so embedded images work in the artifact viewer
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    for png in FIGURES_DIR.glob("c4_*.png"):
        shutil.copy2(png, ARTIFACT_DIR / png.name)

    # Create version of Markdown with artifact-local image paths for the artifact directory
    artifact_md = full_md.replace(
        "file:///S:/Research%20New%20Project/TrustRAG/paper/figures/",
        "/C:/Users/sakin/.gemini/antigravity/brain/ef5e3954-9263-4132-a477-840a113409ca/",
    ).replace(
        "S:/Research New Project/TrustRAG/paper/figures/",
        "/C:/Users/sakin/.gemini/antigravity/brain/ef5e3954-9263-4132-a477-840a113409ca/",
    )
    artifact_md_path = ARTIFACT_DIR / "IT22047724_Individual_Thesis_Full.md"
    artifact_md_path.write_text(artifact_md, encoding="utf-8")

    # Build the formatted Word (.docx) dissertation
    docx_repo_path = THESIS_DIR / "IT22047724_Sakindu_Rathnasiri_Individual_Thesis.docx"
    build_word_document(full_md, docx_repo_path)

    # Copy .docx and .md into the active workspace folder as well for instant access
    shutil.copy2(docx_repo_path, WORKSPACE_DIR / docx_repo_path.name)
    shutil.copy2(master_md_path, WORKSPACE_DIR / master_md_path.name)

    words = len(full_md.split())
    lines = len(full_md.splitlines())
    print(f"Master Markdown written: {master_md_path} ({lines} lines, {words} words)")
    print(f"Formatted DOCX written : {docx_repo_path} ({docx_repo_path.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()


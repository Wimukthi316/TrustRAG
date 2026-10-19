"""Compile the complete SLIIT IT4010 Individual Thesis (Markdown + DOCX) for Component 4.

Uses the official SLIIT IT4010 Individual Thesis Template (`media_1791633370462.docx`)
as the base stylesheet and implements:
1. Single official Title Page (`INDIVIDUAL THESIS` layout, no page number).
2. Preliminary pages (`i, ii, iii...` centered at bottom) using `Front Matter Heading`.
3. Borderless Table of Contents (`toc 1` / `toc 2` with dotted right tab stops at 8211 dxa
   inside a live Word `TOC \\o "1-3" \\h \\z \\u` field), borderless List of Figures,
   borderless List of Tables, borderless List of Abbreviations (`Normal Table`), and
   borderless List of Appendices.
4. Main body Chapters 1-9, References, Glossary, and Appendices A-B (`1, 2, 3...` Arabic
   numerals at bottom-right) using built-in `Heading 1`, `Heading 2`, `Heading 3`, and
   `Caption` styles.
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from urllib.parse import unquote

import docx
from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_ALIGN_VERTICAL, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn
from docx.shared import Inches, Mm, Pt, RGBColor

REPO_ROOT = Path(r"S:\Research New Project\TrustRAG")
THESIS_DIR = REPO_ROOT / "thesis_c4"
FIGURES_DIR = REPO_ROOT / "paper" / "figures"
ARTIFACT_DIR = Path(r"C:\Users\sakin\.gemini\antigravity\brain\ef5e3954-9263-4132-a477-840a113409ca")
TEMPLATE_DOCX = ARTIFACT_DIR / ".user_uploaded" / "media_1791633370462.docx"
WORKSPACE_DIR = Path(r"c:\Users\sakin\Documents\antigravity\vibrant-mendeleev")

SMALL_WORDS = {
    "a", "an", "and", "as", "at", "but", "by", "for", "from", "in", "into",
    "nor", "of", "on", "onto", "or", "over", "per", "the", "to", "up", "via",
    "vs", "vs.", "with", "within", "without",
}


def to_title_case(heading: str) -> str:
    """Convert a section heading (e.g. '1.1 Overall project context') to SLIIT Title Case."""
    parts = heading.split()
    out = []
    for idx, word in enumerate(parts):
        if re.match(r"^\d+(\.\d+)*\.?$", word):
            out.append(word)
        elif any(c.isupper() for c in word[1:]) or word.isupper():
            out.append(word)
        elif idx > 1 and word.lower() in SMALL_WORDS:
            out.append(word.lower())
        else:
            out.append(word[:1].upper() + word[1:])
    return " ".join(out)


def to_sentence_case(heading: str) -> str:
    """Convert a sub-section heading (e.g. '1.1.1 Detailed Sub Heading') to SLIIT Sentence case."""
    preserve = {
        "RAG", "RAGTruth", "RAGBench", "MedHallu", "TrustRAG", "ModernBERT",
        "DeBERTa", "DeBERTa-v3", "RoBERTa", "BGE", "BM25", "IDF", "TF-IDF",
        "NLI", "HHEM", "HHEM-2.1-Open", "LAC", "APS", "CRC", "ECE", "MRR",
        "NDCG", "Hit@1", "Hit@3", "Hit@5", "AUROC", "F1", "IoU", "LOTO",
        "CPU", "GPU", "RAM", "VRAM", "API", "REST", "JSON", "FastAPI",
        "Pydantic", "React", "Vite", "PyTorch", "Python", "SLIIT", "GDPR",
        "EU", "AI", "SDG", "SDGs", "UAT", "SUS", "Cohen's", "Platt", "Beta",
        "Isotonic", "C1", "C2", "C3", "C4", "QA", "D2T",
    }
    parts = heading.split()
    out = []
    first_word_seen = False
    for word in parts:
        if re.match(r"^\d+(\.\d+)*\.?$", word):
            out.append(word)
            continue
        bare = word.strip("():,.")
        if bare in preserve or any(bare.startswith(p) for p in preserve):
            out.append(word)
            first_word_seen = True
        elif not first_word_seen:
            out.append(word[:1].upper() + word[1:].lower())
            first_word_seen = True
        else:
            out.append(word.lower())
    return " ".join(out)


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
        (r"\mathcal{C}_\alpha(s_i)", "C_α(s_i)"),
        (r"\mathcal{C}_\lambda(s_r)", "C_λ(s_r)"),
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
    out = re.sub(r"\\(?:text|mathbf|mathrm|mathcal|operatorname)\{([^}]*)\}", r"\1", out)
    out = re.sub(r"\\frac\{([^}]*)\}\{([^}]*)\}", r"(\1)/(\2)", out)
    out = re.sub(r"\\sqrt\{([^}]*)\}", r"√(\1)", out)
    out = out.replace(r"\left", "").replace(r"\right", "")
    out = out.replace("$$", "")
    out = re.sub(r"\$([^$]+)\$", r"\1", out)
    return out


def configure_section_margins(section) -> None:
    """Apply SLIIT A4 dimensions and 40mm Left/Bottom, 25mm Top/Right margins."""
    section.page_width = Mm(210)
    section.page_height = Mm(297)
    section.left_margin = Mm(40)
    section.bottom_margin = Mm(40)
    section.top_margin = Mm(25)
    section.right_margin = Mm(25)


def set_section_pagination(section, fmt: str | None, align, start: int = 1) -> None:
    """Configure section page numbering per SLIIT Section 03 (Pagination)."""
    footer = section.footer
    footer.is_linked_to_previous = False
    for p in list(footer.paragraphs):
        p.clear()

    if fmt is None:
        return

    sectPr = section._sectPr
    for existing in sectPr.findall(qn("w:pgNumType")):
        sectPr.remove(existing)
    pgNumType = OxmlElement("w:pgNumType")
    pgNumType.set(qn("w:fmt"), fmt)
    pgNumType.set(qn("w:start"), str(start))
    sectPr.append(pgNumType)

    p = footer.paragraphs[0] if footer.paragraphs else footer.add_paragraph()
    p.alignment = align
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


def add_single_official_title_page(doc: Document) -> None:
    """Render the single official IT4010 Individual Thesis Title Page (matching media_1791633370462.docx)."""
    for _ in range(2):
        p_blank = doc.add_paragraph()
        p_blank.alignment = WD_ALIGN_PARAGRAPH.CENTER

    p_title = doc.add_paragraph(style="Title" if "Title" in doc.styles else "Normal")
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_title.paragraph_format.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE
    r_t = p_title.add_run(
        "TRUSTRAG: SPAN-LEVEL CALIBRATED HALLUCINATION DETECTION FOR RETRIEVAL-AUGMENTED GENERATION"
    )
    r_t.font.name = "Times New Roman"
    r_t.font.size = Pt(16)
    r_t.bold = True

    def add_line(text: str, pt: float = 14.0, bold: bool = False, space_after: float = 6.0):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after = Pt(space_after)
        p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE
        r = p.add_run(text)
        r.font.name = "Times New Roman"
        r.font.size = Pt(pt)
        r.bold = bold
        r.font.color.rgb = RGBColor(0, 0, 0)

    add_line("INDIVIDUAL THESIS", pt=14.0, bold=True, space_after=18.0)
    add_line("Epitakumbure Mudunkothgedara Sakindu Nivein Rathnasiri", pt=14.0, bold=False, space_after=4.0)
    add_line("(IT22047724)", pt=14.0, bold=False, space_after=14.0)
    add_line(
        "Individual component: Retrieval-Attribution Alignment and Evidence Grounding",
        pt=14.0,
        bold=False,
        space_after=4.0,
    )
    add_line("Group ID: R26-DS-018", pt=14.0, bold=False, space_after=24.0)

    add_line(
        "Report submitted in partial fulfilment of the requirements for the",
        pt=12.0,
        bold=False,
        space_after=4.0,
    )
    add_line(
        "Bachelor of Science Honours in Information Technology",
        pt=12.0,
        bold=False,
        space_after=4.0,
    )
    add_line("Specialising in Data Science", pt=12.0, bold=False, space_after=24.0)

    add_line("Department of Information Technology", pt=12.0, bold=False, space_after=4.0)
    add_line("Sri Lanka Institute of Information Technology", pt=12.0, bold=False, space_after=4.0)
    add_line("Sri Lanka", pt=12.0, bold=False, space_after=18.0)
    add_line("October 2026", pt=12.0, bold=False, space_after=0.0)


def format_inline_runs(paragraph, text: str, font_size_pt: float = 12.0, default_bold: bool = False) -> None:
    """Parse **bold**, *italic*, and `code` tokens and add formatted Times New Roman runs."""
    text = text.replace("<br>", "\n")
    text = clean_latex_for_docx(text)
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


def add_dotted_tab_stop(paragraph, pos_dxa: int = 8211, leader: str = "dot") -> None:
    """Add a right-aligned tab stop with dot leader matching the IT4010 template (pos=8211)."""
    pPr = paragraph._p.get_or_add_pPr()
    tabs = pPr.find(qn("w:tabs"))
    if tabs is None:
        tabs = OxmlElement("w:tabs")
        pPr.append(tabs)
    tab = OxmlElement("w:tab")
    tab.set(qn("w:val"), "right")
    tab.set(qn("w:leader"), leader)
    tab.set(qn("w:pos"), str(pos_dxa))
    tabs.append(tab)


def add_borderless_toc_or_list(doc: Document, table_lines: list[str], section_kind: str) -> None:
    """Render TOC, List of Figures, List of Tables, or List of Appendices as borderless tab-aligned lines."""
    parsed_rows = []
    for line in table_lines:
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        inner = stripped[1:-1] if stripped.endswith("|") else stripped[1:]
        cells = [c.strip().replace("\\|", "|") for c in re.split(r"(?<!\\)\|", inner)]
        if all(re.match(r"^:?-{2,}:?$", c.replace(" ", "")) for c in cells if c != ""):
            continue
        parsed_rows.append(cells)

    if len(parsed_rows) <= 1:
        return

    # Skip the Markdown table header row (e.g., | Section | Title | Page |)
    data_rows = parsed_rows[1:]

    # If this is TABLE OF CONTENTS, wrap the first and last entry in a Word TOC field so Ctrl+A -> F9 works
    is_toc = section_kind == "TOC"

    # For List of Figures / Tables / Appendices, add a right-aligned "Page" header line per SLIIT Appendices IV, V, VII
    if section_kind in ("LOF", "LOT", "LOA"):
        p_hdr = doc.add_paragraph()
        p_hdr.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        p_hdr.paragraph_format.space_before = Pt(0)
        p_hdr.paragraph_format.space_after = Pt(6)
        p_hdr.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
        r_hdr = p_hdr.add_run("Page")
        r_hdr.font.name = "Times New Roman"
        r_hdr.font.size = Pt(12)
        r_hdr.bold = False

    for idx, row in enumerate(data_rows):
        if len(row) >= 3:
            col1 = re.sub(r"\*+", "", row[0]).strip()
            col2 = re.sub(r"\*+", "", row[1]).strip()
            page_str = re.sub(r"\*+", "", row[2]).strip()
            is_chapter_row = bool(col1.endswith(".") or row[1].strip().startswith("**"))
            label = f"{col1} {col2}".strip() if col1 else col2
        elif len(row) == 2:
            label = re.sub(r"\*+", "", row[0]).strip()
            page_str = re.sub(r"\*+", "", row[1]).strip()
            is_chapter_row = False
        else:
            continue

        style_name = "toc 1" if (is_chapter_row or not is_toc) else "toc 2"
        p = doc.add_paragraph(style=style_name if style_name in doc.styles else "Normal")
        p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
        p.paragraph_format.space_before = Pt(2 if is_chapter_row else 1)
        p.paragraph_format.space_after = Pt(2 if is_chapter_row else 1)
        if not is_chapter_row and is_toc and row[0].strip():
            p.paragraph_format.left_indent = Mm(6)
        add_dotted_tab_stop(p, pos_dxa=8211, leader="dot")

        if is_toc and idx == 0:
            fld_begin = OxmlElement("w:fldChar")
            fld_begin.set(qn("w:fldCharType"), "begin")
            instr = OxmlElement("w:instrText")
            instr.set(qn("xml:space"), "preserve")
            instr.text = ' TOC \\o "1-3" \\h \\z \\u '
            fld_sep = OxmlElement("w:fldChar")
            fld_sep.set(qn("w:fldCharType"), "separate")
            r0 = p.add_run()
            r0._r.append(fld_begin)
            r0._r.append(instr)
            r0._r.append(fld_sep)

        r_text = p.add_run(clean_latex_for_docx(label) + "\t" + page_str)
        r_text.font.name = "Times New Roman"
        r_text.font.size = Pt(12)
        r_text.bold = is_chapter_row and is_toc

        if is_toc and idx == len(data_rows) - 1:
            fld_end = OxmlElement("w:fldChar")
            fld_end.set(qn("w:fldCharType"), "end")
            r_end = p.add_run()
            r_end._r.append(fld_end)


def remove_table_borders(table) -> None:
    """Remove all visible borders from a Word table (for List of Abbreviations and Supervisor Signature block)."""
    tblPr = table._tbl.tblPr
    for existing in tblPr.findall(qn("w:tblBorders")):
        tblPr.remove(existing)
    borders = parse_xml(
        f'<w:tblBorders {nsdecls("w")}>'
        '<w:top w:val="none" w:sz="0" w:space="0" w:color="auto"/>'
        '<w:left w:val="none" w:sz="0" w:space="0" w:color="auto"/>'
        '<w:bottom w:val="none" w:sz="0" w:space="0" w:color="auto"/>'
        '<w:right w:val="none" w:sz="0" w:space="0" w:color="auto"/>'
        '<w:insideH w:val="none" w:sz="0" w:space="0" w:color="auto"/>'
        '<w:insideV w:val="none" w:sz="0" w:space="0" w:color="auto"/>'
        "</w:tblBorders>"
    )
    tblPr.append(borders)


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


def add_table_caption_paragraph(doc: Document, caption_text: str) -> None:
    """Add a centered, single-spaced Table caption ABOVE the table using Caption style."""
    clean_cap = caption_text.strip()
    if clean_cap.startswith("*") and clean_cap.endswith("*") and not clean_cap.startswith("**"):
        clean_cap = clean_cap[1:-1].strip()
    p = doc.add_paragraph(style="Caption" if "Caption" in doc.styles else "Normal")
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(8)
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
    p.paragraph_format.keep_with_next = True
    format_inline_runs(p, clean_cap, font_size_pt=11.0, default_bold=False)


def add_markdown_table(doc: Document, table_lines: list[str], borderless: bool = False) -> None:
    """Render a Markdown table into a centered Word table (bordered for data tables, borderless for Abbreviations)."""
    parsed_rows = []
    for line in table_lines:
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        inner = stripped[1:-1] if stripped.endswith("|") else stripped[1:]
        cells = [c.strip().replace("\\|", "|") for c in re.split(r"(?<!\\)\|", inner)]
        if all(re.match(r"^:?-{2,}:?$", c.replace(" ", "")) for c in cells if c != ""):
            continue
        parsed_rows.append(cells)

    if not parsed_rows:
        return

    num_cols = max(len(r) for r in parsed_rows)
    table = doc.add_table(rows=len(parsed_rows), cols=num_cols)
    if borderless:
        table.style = "Normal Table"
        remove_table_borders(table)
    else:
        table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER

    font_pt = 11.5 if borderless else (9.0 if num_cols >= 7 else (9.5 if num_cols >= 5 else 10.5))

    for r_idx, row_data in enumerate(parsed_rows):
        row = table.rows[r_idx]
        is_header = r_idx == 0
        for c_idx in range(num_cols):
            cell_text = row_data[c_idx] if c_idx < len(row_data) else ""
            cell = row.cells[c_idx]
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            set_cell_margins(cell, top=50, bottom=50, left=80, right=80)
            if is_header and not borderless:
                set_cell_background(cell, "EDEDED")
            p = cell.paragraphs[0]
            p.paragraph_format.space_before = Pt(2 if borderless else 1)
            p.paragraph_format.space_after = Pt(2 if borderless else 1)
            p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
            format_inline_runs(p, cell_text, font_size_pt=font_pt, default_bold=is_header)

    sp = doc.add_paragraph()
    sp.paragraph_format.space_before = Pt(0)
    sp.paragraph_format.space_after = Pt(4)


def build_word_document(full_md: str, out_docx: Path) -> None:
    """Convert the full Markdown dissertation into a strictly formatted SLIIT IT4010 .docx file."""
    if TEMPLATE_DOCX.exists():
        doc = Document(str(TEMPLATE_DOCX))
        # Clear template placeholder paragraphs/tables/sections while preserving its stylesheet
        body = doc._body._body
        for child in list(body):
            if not child.tag.endswith("sectPr"):
                body.remove(child)
    else:
        doc = Document()

    # Section 1: Single Official Title Page (NO page number)
    sec1 = doc.sections[0]
    configure_section_margins(sec1)
    set_section_pagination(sec1, fmt=None, align=WD_ALIGN_PARAGRAPH.CENTER)

    normal_style = doc.styles["Normal"]
    normal_style.font.name = "Times New Roman"
    normal_style.font.size = Pt(12)
    normal_style.font.color.rgb = RGBColor(0, 0, 0)

    for h_name, pt_size in [("Heading 1", 14), ("Heading 2", 12), ("Heading 3", 12)]:
        if h_name in doc.styles:
            h_style = doc.styles[h_name]
            h_style.font.name = "Times New Roman"
            h_style.font.size = Pt(pt_size)
            h_style.font.bold = True
            h_style.font.color.rgb = RGBColor(0, 0, 0)

    # Render ONLY the single official Title Page (matching IT4010 template)
    add_single_official_title_page(doc)

    # Section 2: Preliminary Pages (Lowercase Roman numerals i, ii, iii... centered at bottom)
    sec2 = doc.add_section(WD_SECTION.NEW_PAGE)
    configure_section_margins(sec2)
    set_section_pagination(sec2, fmt="lowerRoman", align=WD_ALIGN_PARAGRAPH.CENTER, start=1)

    lines = full_md.splitlines()
    i = 0
    n = len(lines)
    current_h1 = ""
    in_single_space_section = False
    in_abstract_section = False
    first_prelim_h1 = True
    body_section_started = False

    while i < n:
        line = lines[i]
        stripped = line.strip()

        if stripped in ("---", "***", "___"):
            i += 1
            continue

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
            i += 1
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

            # Check if we are in TOC, List of Figures, List of Tables, List of Appendices, or List of Abbreviations
            if "TABLE OF CONTENTS" in current_h1:
                # Do NOT output static text rows. Insert native Word dynamic TOC field so
                # the user can click 'Update Field' or insert/refresh a responsive TOC via References tab in Word.
                p_toc = doc.add_paragraph()
                p_toc.paragraph_format.space_before = Pt(6)
                p_toc.paragraph_format.space_after = Pt(12)
                p_toc.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
                fld = OxmlElement("w:fldSimple")
                fld.set(qn("w:instr"), 'TOC \\o "1-3" \\h \\z \\u')
                r_toc = OxmlElement("w:r")
                t_toc = OxmlElement("w:t")
                t_toc.text = "[Right-click and select 'Update Field', or click 'References' -> 'Table of Contents']"
                r_toc.append(t_toc)
                fld.append(r_toc)
                p_toc._p.append(fld)
                continue
            elif "LIST OF FIGURES" in current_h1:
                add_borderless_toc_or_list(doc, table_lines, "LOF")
                continue
            elif "LIST OF TABLES" in current_h1:
                add_borderless_toc_or_list(doc, table_lines, "LOT")
                continue
            elif "LIST OF APPENDICES" in current_h1:
                add_borderless_toc_or_list(doc, table_lines, "LOA")
                continue
            elif "LIST OF ABBREVIATIONS" in current_h1:
                add_markdown_table(doc, table_lines, borderless=True)
                continue

            # Look ahead to see if the next non-empty line is a trailing "*Table X.Y: ...*" caption
            lookahead = i
            while lookahead < n and not lines[lookahead].strip():
                lookahead += 1
            if lookahead < n and re.match(r"^\*+Table\s+\d+", lines[lookahead].strip()):
                add_table_caption_paragraph(doc, lines[lookahead].strip())
                i = lookahead + 1
            add_markdown_table(doc, table_lines, borderless=False)
            continue

        # Handle embedded images ![alt](path) -> Figure centered + Caption BELOW figure centered
        img_match = re.match(r"^!\[([^\]]*)\]\(([^)]+)\)$", stripped)
        if img_match:
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
                    p_cap = doc.add_paragraph(style="Caption" if "Caption" in doc.styles else "Normal")
                    p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
                    p_cap.paragraph_format.space_before = Pt(2)
                    p_cap.paragraph_format.space_after = Pt(10)
                    p_cap.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
                    format_inline_runs(p_cap, alt_text, font_size_pt=11.0)
            i += 1
            continue

        # Handle Headings (#, ##, ###, ####)
        h_match = re.match(r"^(#{1,4})\s+(.+)$", stripped)
        if h_match:
            level = len(h_match.group(1))
            heading_text = clean_latex_for_docx(h_match.group(2).strip())

            if level == 1:
                upper_h = heading_text.upper()
                current_h1 = upper_h
                if upper_h.startswith("1 INTRODUCTION") and not body_section_started:
                    sec3 = doc.add_section(WD_SECTION.NEW_PAGE)
                    configure_section_margins(sec3)
                    set_section_pagination(sec3, fmt="decimal", align=WD_ALIGN_PARAGRAPH.RIGHT, start=1)
                    body_section_started = True
                elif not first_prelim_h1:
                    doc.add_page_break()
                first_prelim_h1 = False

                in_abstract_section = upper_h == "ABSTRACT"
                in_references_section = upper_h in ("REFERENCES", "REFERENCE LIST")
                in_single_space_section = in_abstract_section or in_references_section or any(
                    k in upper_h
                    for k in (
                        "TABLE OF CONTENTS",
                        "LIST OF FIGURES",
                        "LIST OF TABLES",
                        "LIST OF ABBREVIATIONS",
                        "LIST OF APPENDICES",
                        "DECLARATION",
                    )
                )

                # Use Front Matter Heading for preliminary pages and Heading 1 for body chapters
                if not body_section_started and "Front Matter Heading" in doc.styles:
                    h_style = "Front Matter Heading"
                elif "Heading 1" in doc.styles:
                    h_style = "Heading 1"
                else:
                    h_style = "Normal"

                p = doc.add_paragraph(style=h_style)
                p.paragraph_format.space_before = Pt(12)
                p.paragraph_format.space_after = Pt(12)
                p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
                p.paragraph_format.keep_with_next = True

                r = p.add_run(upper_h)
                r.font.name = "Times New Roman"
                r.font.size = Pt(14)
                r.bold = True
                r.font.color.rgb = RGBColor(0, 0, 0)
            elif level == 2:
                h_style = "Heading 2" if "Heading 2" in doc.styles else "Normal"
                p = doc.add_paragraph(style=h_style)
                p.paragraph_format.space_before = Pt(10)
                p.paragraph_format.space_after = Pt(6)
                p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
                p.paragraph_format.keep_with_next = True
                r = p.add_run(to_title_case(heading_text))
                r.font.name = "Times New Roman"
                r.font.size = Pt(12)
                r.bold = True
                r.font.color.rgb = RGBColor(0, 0, 0)
            else:
                h_style = "Heading 3" if "Heading 3" in doc.styles else "Normal"
                p = doc.add_paragraph(style=h_style)
                p.paragraph_format.space_before = Pt(8)
                p.paragraph_format.space_after = Pt(4)
                p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
                p.paragraph_format.keep_with_next = True
                r = p.add_run(to_sentence_case(heading_text))
                r.font.name = "Times New Roman"
                r.font.size = Pt(12)
                r.bold = True
                r.font.color.rgb = RGBColor(0, 0, 0)
            i += 1
            continue

        if not stripped:
            i += 1
            continue

        # Handle Blockquotes (> ...)
        if stripped.startswith(">"):
            quote_text = stripped.lstrip(">").strip()
            quote_text = re.sub(r"^\[!(?:NOTE|IMPORTANT|TIP|WARNING|CAUTION)\]\s*", "", quote_text)
            p = doc.add_paragraph()
            p.paragraph_format.left_indent = Mm(10)
            p.paragraph_format.right_indent = Mm(6)
            p.paragraph_format.space_before = Pt(4)
            p.paragraph_format.space_after = Pt(4)
            p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
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
            p.paragraph_format.space_before = Pt(0)
            p.paragraph_format.space_after = Pt(12)
            p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
            p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            format_inline_runs(p, stripped, font_size_pt=11.0)
            i += 1
            continue

        # Handle Table captions that appear before a table (**Table X.Y: ...**)
        if re.match(r"^\*\*Table\s+", stripped):
            add_table_caption_paragraph(doc, stripped)
            i += 1
            continue

        # Handle Algorithm / Proposition / Note lines
        is_caption = bool(
            re.match(r"^\*\*(?:Figure|Algorithm|Proposition)\s+", stripped)
            or re.match(r"^\*(?:Figure|Note:)\s*", stripped)
        )
        p = doc.add_paragraph()
        if is_caption:
            p.paragraph_format.space_before = Pt(6)
            p.paragraph_format.space_after = Pt(3)
            p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
            p.paragraph_format.keep_with_next = True
            format_inline_runs(p, stripped, font_size_pt=11.0)
        elif in_abstract_section:
            p.paragraph_format.space_before = Pt(0)
            p.paragraph_format.space_after = Pt(6)
            p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
            p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
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
    try:
        doc.save(str(out_docx))
    except PermissionError:
        alt_path = out_docx.with_name(out_docx.stem + "_Updated.docx")
        doc.save(str(alt_path))
        print(f"Note: {out_docx.name} is currently open in Word. Saved to {alt_path.name} instead.")
        out_docx = alt_path
    return out_docx


def ensure_all_chapter6_figures() -> None:
    """Ensure all 7 Chapter 6 figure filenames exist in paper/figures at 300 DPI."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
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

    master_md_path = THESIS_DIR / "IT22047724_Individual_Thesis_Full.md"
    master_md_path.write_text(full_md, encoding="utf-8")

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    for png in FIGURES_DIR.glob("c4_*.png"):
        shutil.copy2(png, ARTIFACT_DIR / png.name)

    artifact_md = full_md.replace(
        "file:///S:/Research%20New%20Project/TrustRAG/paper/figures/",
        "/C:/Users/sakin/.gemini/antigravity/brain/ef5e3954-9263-4132-a477-840a113409ca/",
    ).replace(
        "S:/Research New Project/TrustRAG/paper/figures/",
        "/C:/Users/sakin/.gemini/antigravity/brain/ef5e3954-9263-4132-a477-840a113409ca/",
    )
    artifact_md_path = ARTIFACT_DIR / "IT22047724_Individual_Thesis_Full.md"
    artifact_md_path.write_text(artifact_md, encoding="utf-8")

    docx_repo_path = THESIS_DIR / "IT22047724_Sakindu_Rathnasiri_Individual_Thesis.docx"
    saved_path = build_word_document(full_md, docx_repo_path)
    if saved_path is None:
        saved_path = docx_repo_path

    try:
        shutil.copy2(saved_path, WORKSPACE_DIR / saved_path.name)
    except PermissionError:
        pass
    shutil.copy2(master_md_path, WORKSPACE_DIR / master_md_path.name)

    words = len(full_md.split())
    lines = len(full_md.splitlines())
    print(f"Master Markdown written: {master_md_path} ({lines} lines, {words} words)")
    print(f"Formatted DOCX written : {saved_path} ({saved_path.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()

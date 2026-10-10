"""Compile the complete SLIIT IT4010 Individual Thesis (Markdown + DOCX) for Component 4.

Strictly implements all formatting, layout, cover/title page, pagination (Roman i..ix
centered on preliminary pages, Arabic 1..N bottom-right on main body & appendices, no
page number on Cover/Title pages), font, margin, table-caption-above, figure-caption-below,
and reference-spacing rules from the SLIIT 'Guidelines on Documentation and Submission
of Dissertations' (Appendices I-VII) and the IT4010 Individual Thesis Template.
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
        # Preserve section numbers like '1.1', acronyms, or hyphenated tokens with capitals
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
    """Configure section page numbering per SLIIT Section 03 (Pagination).

    - fmt=None: No page number (Cover Page and Title Page)
    - fmt='lowerRoman', align=CENTER, start=1: Preliminary pages (i, ii, iii...) centered at bottom
    - fmt='decimal', align=RIGHT, start=1: Main body & appendices (1, 2, 3...) bottom-right
    """
    footer = section.footer
    footer.is_linked_to_previous = False
    for p in list(footer.paragraphs):
        p.clear()

    if fmt is None:
        return

    # Set w:pgNumType on sectPr
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


def add_centered_line(doc: Document, text: str, pt_size: float, bold: bool = False) -> None:
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(0)
    p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE
    if text:
        r = p.add_run(text)
        r.font.name = "Times New Roman"
        r.font.size = Pt(pt_size)
        r.bold = bold
        r.font.color.rgb = RGBColor(0, 0, 0)
    else:
        r = p.add_run("")
        r.font.name = "Times New Roman"
        r.font.size = Pt(pt_size)


def add_sliit_cover_and_title_pages(doc: Document) -> None:
    """Render Appendix I (Cover Page) and Appendix III (Title Page) with exact SLIIT line spacing."""
    # --- APPENDIX I: COVER PAGE ---
    add_centered_line(
        doc,
        "TRUSTRAG: SPAN-LEVEL CALIBRATED HALLUCINATION DETECTION FOR RETRIEVAL-AUGMENTED GENERATION",
        16,
        bold=True,
    )
    add_centered_line(
        doc,
        "RETRIEVAL-ATTRIBUTION ALIGNMENT AND EVIDENCE GROUNDING",
        14,
        bold=True,
    )
    # 5 blank lines (16pt, 1.5 space)
    for _ in range(5):
        add_centered_line(doc, "", 16)

    add_centered_line(doc, "Epitakumbure Mudunkothgedara Sakindu Nivein Rathnasiri", 14, bold=False)
    add_centered_line(doc, "", 16)
    add_centered_line(doc, "(IT22047724)", 14, bold=False)

    for _ in range(2):
        add_centered_line(doc, "", 16)

    add_centered_line(doc, "BSc (Hons) Degree in Information Technology", 14, bold=False)
    add_centered_line(doc, "Specialising in Data Science", 14, bold=False)

    for _ in range(3):
        add_centered_line(doc, "", 16)

    add_centered_line(doc, "Department of Information Technology", 14, bold=False)
    add_centered_line(doc, "", 16)
    add_centered_line(doc, "Sri Lanka Institute of Information Technology", 14, bold=False)
    add_centered_line(doc, "Sri Lanka", 14, bold=False)

    for _ in range(2):
        add_centered_line(doc, "", 16)

    add_centered_line(doc, "October 2026", 14, bold=False)

    # Page break before Appendix III Title Page (still inside Section 1 with no page numbers)
    doc.add_page_break()

    # --- APPENDIX III: TITLE PAGE ---
    add_centered_line(
        doc,
        "TRUSTRAG: SPAN-LEVEL CALIBRATED HALLUCINATION DETECTION FOR RETRIEVAL-AUGMENTED GENERATION",
        16,
        bold=True,
    )
    add_centered_line(
        doc,
        "RETRIEVAL-ATTRIBUTION ALIGNMENT AND EVIDENCE GROUNDING",
        14,
        bold=True,
    )
    for _ in range(5):
        add_centered_line(doc, "", 16)

    add_centered_line(doc, "Epitakumbure Mudunkothgedara Sakindu Nivein Rathnasiri", 14, bold=False)
    add_centered_line(doc, "", 16)
    add_centered_line(doc, "(IT22047724)", 14, bold=False)

    for _ in range(2):
        add_centered_line(doc, "", 16)

    add_centered_line(
        doc,
        "Dissertation submitted in partial fulfillment of the requirements for the Degree of",
        12,
        bold=False,
    )
    add_centered_line(
        doc,
        "Bachelor of Science Honours in Information Technology Specialising in Data Science",
        12,
        bold=False,
    )

    for _ in range(3):
        add_centered_line(doc, "", 16)

    add_centered_line(doc, "Department of Information Technology", 14, bold=False)
    add_centered_line(doc, "", 16)
    add_centered_line(doc, "Sri Lanka Institute of Information Technology", 14, bold=False)
    add_centered_line(doc, "Sri Lanka", 14, bold=False)
    add_centered_line(doc, "", 16)
    add_centered_line(doc, "October 2026", 14, bold=False)


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
    """Add a centered, single-spaced Table caption ABOVE the table per SLIIT Section 04."""
    clean_cap = caption_text.strip()
    if clean_cap.startswith("*") and clean_cap.endswith("*") and not clean_cap.startswith("**"):
        clean_cap = clean_cap[1:-1].strip()
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(8)
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
    p.paragraph_format.keep_with_next = True
    format_inline_runs(p, clean_cap, font_size_pt=11.0, default_bold=False)


def add_markdown_table(doc: Document, table_lines: list[str]) -> None:
    """Render a Markdown table into a centered, bordered, single-spaced Word table."""
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
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER

    font_pt = 9.0 if num_cols >= 7 else (9.5 if num_cols >= 5 else 10.5)

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

    sp = doc.add_paragraph()
    sp.paragraph_format.space_before = Pt(0)
    sp.paragraph_format.space_after = Pt(4)


def build_word_document(full_md: str, out_docx: Path) -> None:
    """Convert the full Markdown dissertation into a strictly formatted SLIIT IT4010 .docx file."""
    doc = Document()

    # Section 1: Cover Page & Title Page (NO page numbers)
    sec1 = doc.sections[0]
    configure_section_margins(sec1)
    set_section_pagination(sec1, fmt=None, align=WD_ALIGN_PARAGRAPH.CENTER)

    normal_style = doc.styles["Normal"]
    normal_style.font.name = "Times New Roman"
    normal_style.font.size = Pt(12)
    normal_style.font.color.rgb = RGBColor(0, 0, 0)

    # Render Appendix I (Cover Page) and Appendix III (Title Page)
    add_sliit_cover_and_title_pages(doc)

    # Section 2: Preliminary Pages (Lowercase Roman numerals i, ii, iii... centered at bottom)
    sec2 = doc.add_section(WD_SECTION.NEW_PAGE)
    configure_section_margins(sec2)
    set_section_pagination(sec2, fmt="lowerRoman", align=WD_ALIGN_PARAGRAPH.CENTER, start=1)

    lines = full_md.splitlines()
    i = 0
    n = len(lines)
    in_single_space_section = False
    in_abstract_section = False
    in_references_section = False
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

        # Handle Markdown tables (and check if a Table caption immediately follows the table)
        if stripped.startswith("|") and "|" in stripped[1:]:
            table_lines = []
            while i < n and lines[i].strip().startswith("|"):
                table_lines.append(lines[i])
                i += 1
            # Look ahead to see if the next non-empty line is a trailing "*Table X.Y: ...*" caption
            lookahead = i
            while lookahead < n and not lines[lookahead].strip():
                lookahead += 1
            if lookahead < n and re.match(r"^\*+Table\s+\d+", lines[lookahead].strip()):
                add_table_caption_paragraph(doc, lines[lookahead].strip())
                i = lookahead + 1
            add_markdown_table(doc, table_lines)
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
                    p_cap = doc.add_paragraph()
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
                # Check if this is Chapter 1 (# 1 INTRODUCTION) -> Start Section 3 (Arabic 1, 2, 3... bottom-right)
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

                p = doc.add_paragraph()
                p.paragraph_format.space_before = Pt(12)
                p.paragraph_format.space_after = Pt(12)
                p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
                p.paragraph_format.keep_with_next = True

                # Per SLIIT Section 1.4 (Page 6): Abstract Title = Title case / Times New Roman 12-point Bold
                if in_abstract_section:
                    r = p.add_run("Abstract")
                    r.font.name = "Times New Roman"
                    r.font.size = Pt(12)
                    r.bold = True
                    r.font.color.rgb = RGBColor(0, 0, 0)
                else:
                    # Chapter Heading: ALL CAPITAL — 14 Font size, Bold
                    r = p.add_run(upper_h)
                    r.font.name = "Times New Roman"
                    r.font.size = Pt(14)
                    r.bold = True
                    r.font.color.rgb = RGBColor(0, 0, 0)
            elif level == 2:
                # Section Heading: Title case — 12 Font size, Bold (SLIIT Section 2.2)
                p = doc.add_paragraph()
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
                # Sub-section Heading: Sentence case — 12 Font size, Bold (SLIIT Section 2.2)
                p = doc.add_paragraph()
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

        # Handle Blockquotes (> ...) — Indented quotations use single line spacing per SLIIT Section 2.1
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

        # Handle IEEE Reference entries [1] ... (Single-spaced entry with double spacing between entries, SLIIT Section 1.12)
        if re.match(r"^\[\d+\]\s+", stripped):
            p = doc.add_paragraph()
            p.paragraph_format.left_indent = Mm(8)
            p.paragraph_format.first_line_indent = Mm(-8)
            p.paragraph_format.space_before = Pt(0)
            p.paragraph_format.space_after = Pt(12)  # Double spacing between single-spaced entries
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
            # Per SLIIT Section 1.4 (Page 6): Abstract Text = Times New Roman 11-point, Single line spacing
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
    doc.save(str(out_docx))


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

    # Figure 6.1: Attribution Backend Top-1 Accuracy by Task vs Baselines
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

    # Figure 6.2: Score Calibration ECE Comparison Across Calibrators
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

    # Figure 6.5: Calibrated No-Evidence Abstention Precision, Recall, and F1
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

    # Figure 6.6: Leave-One-Task-Out Cross-Task Transfer Matrix
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

"""Layer 15 -- Chapter 4 Product and Solution Design architectural diagrams.

Generates three publication-grade vector (.pdf) and high-resolution (.png, 300 DPI)
diagrams in `paper/figures/` for Chapter 4 of the individual dissertation:

  1. `c4_arch_system.png/.pdf`   (Figure 4.1: Overall TrustRAG 4-Stage Architecture)
  2. `c4_arch_pipeline.png/.pdf` (Figure 4.2: Component 4 Internal Pipeline & Tie-Break)
  3. `c4_arch_schema.png/.pdf`   (Figure 4.3: Frozen Data Contract & Artefact Schema)

Kept in a standalone module so `figures.FIGURE_NAMES` stays in 1-to-1 lockstep
with the backend's `FIGURE_ORDER` whitelist tested in `test_c4_tables_figures.py`.

    python -m src.c4_attribution.arch_diagrams
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "paper/figures"

DIAGRAM_NAMES: Tuple[str, ...] = (
    "c4_arch_system",
    "c4_arch_pipeline",
    "c4_arch_schema",
)


def _box(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    lines: Sequence[str],
    facecolor: str = "#f8fafc",
    edgecolor: str = "#334155",
    title_color: str = "#0f172a",
    lw: float = 1.6,
    title_size: float = 9.5,
    body_size: float = 8.0,
) -> None:
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.008,rounding_size=0.015",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=lw,
        zorder=2,
    )
    ax.add_patch(patch)
    top_y = y + h - 0.020
    ax.text(
        x + w / 2.0,
        top_y,
        title,
        ha="center",
        va="top",
        fontsize=title_size,
        fontweight="bold",
        color=title_color,
        linespacing=1.22,
        zorder=3,
    )
    if lines:
        n_title_lines = title.count("\n") + 1
        body_top = top_y - (0.036 * n_title_lines + 0.014)
        body_text = "\n".join(lines)
        ax.text(
            x + w / 2.0,
            body_top,
            body_text,
            ha="center",
            va="top",
            fontsize=body_size,
            color="#1e293b",
            linespacing=1.34,
            zorder=3,
        )


def _arrow(
    ax: plt.Axes,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    label: str = "",
    color: str = "#334155",
    lw: float = 1.6,
    rad: float = 0.0,
    label_dx: float = 0.0,
    label_dy: float = 0.018,
) -> None:
    arr = FancyArrowPatch(
        (x0, y0),
        (x1, y1),
        arrowstyle="-|>,head_length=6.5,head_width=3.8",
        connectionstyle=f"arc3,rad={rad}",
        color=color,
        linewidth=lw,
        zorder=4,
    )
    ax.add_patch(arr)
    if label:
        mx = (x0 + x1) / 2.0 + label_dx
        my = (y0 + y1) / 2.0 + label_dy
        ax.text(
            mx,
            my,
            label,
            ha="center",
            va="center",
            fontsize=7.8,
            fontweight="bold",
            color=color,
            bbox={
                "boxstyle": "round,pad=0.18",
                "fc": "white",
                "ec": color,
                "lw": 0.6,
                "alpha": 0.96,
            },
            zorder=5,
        )


def _save(fig: plt.Figure, name: str, out_dir: Path = OUT_DIR) -> List[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / f"{name}.png"
    pdf_path = out_dir / f"{name}.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return [png_path, pdf_path]


def draw_system_architecture(out_dir: Path = OUT_DIR) -> List[Path]:
    """Figure 4.1: Overall TrustRAG 4-Stage Architecture highlighting C4."""
    fig, ax = plt.subplots(figsize=(14.8, 7.2))
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")

    ax.text(
        0.5,
        0.968,
        "Figure 4.1: TrustRAG Four-Stage Calibrated Architecture (Component 4 Highlighted)",
        ha="center",
        va="top",
        fontsize=12.5,
        fontweight="bold",
        color="#0f172a",
    )

    # Input Request
    _box(
        ax,
        0.015,
        0.54,
        0.155,
        0.34,
        "AnalyzeRequest",
        [
            "• question: str",
            "• context: str",
            "• answer: str",
            "• alpha: float (0.10)",
            "",
            "Naive Splitter:",
            "(?<=[.!?])\\s+",
        ],
        facecolor="#f1f5f9",
        edgecolor="#475569",
    )

    # C1 Box
    _box(
        ax,
        0.215,
        0.54,
        0.165,
        0.34,
        "Stage 1: C1 Detector",
        [
            "ModernBERT-base",
            "Token-Level Tagger",
            "",
            "• Masked CE Loss",
            "• Char-offset spans",
            "• Raw span_score",
            "• Example F1: 0.7623",
        ],
        facecolor="#eff6ff",
        edgecolor="#2563eb",
        title_color="#1d4ed8",
    )

    # C2 Box
    _box(
        ax,
        0.425,
        0.54,
        0.175,
        0.34,
        "Stage 2: C2 Calibration",
        [
            "Platt + Split Conformal",
            "",
            "• calibrated_score",
            "• FLAG / ABSTAIN / PASS",
            "• 37.4% ABSTAIN rate",
            "• Exchangeability Gate:",
            "  guarantee_applies",
        ],
        facecolor="#f0fdf4",
        edgecolor="#16a34a",
        title_color="#15803d",
    )

    # C4 Box (Highlighted)
    _box(
        ax,
        0.645,
        0.49,
        0.34,
        0.41,
        "Stage 4: C4 Evidence Grounding (Owned)",
        [
            "Retrieval-Attribution Alignment & Conformal Sets",
            "──────────────────────────────────────────",
            "1. IDF Content-Token Scorer (Top-1: 0.8952, MRR: 0.9307)",
            "2. No-Evidence Abstention Gate (tau = 0.1973, F1: 0.4814)",
            "3. LAC Conformal Set C_alpha (90.03% cov, 4.61 / 19.05 sents)",
            "   -> 70.63% Reviewer Reading Effort Reduction",
            "4. C1×C2×C4 Tie-Break on C2 Abstentions (+5.96 F1 pts)",
            "5. Served via c4_artifact.json (1.31 ms p50, 0 MB RAM)",
        ],
        facecolor="#fff7ed",
        edgecolor="#ea580c",
        title_color="#c2410c",
        lw=2.5,
    )

    # C3 Box (Bottom middle)
    _box(
        ax,
        0.215,
        0.08,
        0.385,
        0.33,
        "Stage 3: C3 Calibration-Consistent Explanation",
        [
            "• Consumes C2 calibrated_score + C4 evidence_sentence",
            "• Enforces monotonic hedge words aligned with risk tier",
            "• Respects OOD guarantee withdrawal (guarantee_applies = False)",
            "• Shares identical sentence splitter with C4 & Backend",
        ],
        facecolor="#faf5ff",
        edgecolor="#9333ea",
        title_color="#7e22ce",
    )

    # Output Contract Box (Bottom right)
    _box(
        ax,
        0.645,
        0.08,
        0.34,
        0.31,
        "Frozen AnalysisResult Contract & React UI",
        [
            "• Span.evidence_sentence: str | None",
            "• Span.evidence_index: int | None",
            "• Span.entailment_score: float | None",
            "• Context Sentence Highlighter & Conformal Set View",
            "• Serving Rules 1-3 Enforced by 147 C4 Pytest Tests",
        ],
        facecolor="#f8fafc",
        edgecolor="#0f172a",
        title_color="#0f172a",
        lw=1.8,
    )

    # Arrows with clean label offsets above the gap
    _arrow(ax, 0.170, 0.71, 0.215, 0.71, "input", label_dy=0.032)
    _arrow(ax, 0.380, 0.71, 0.425, 0.71, "raw spans", label_dy=0.032)
    _arrow(ax, 0.600, 0.71, 0.645, 0.71, "decisions", label_dy=0.032)
    _arrow(ax, 0.490, 0.54, 0.410, 0.41, "calibrated risk", label_dx=-0.04, label_dy=0.01)
    _arrow(
        ax,
        0.700,
        0.49,
        0.540,
        0.41,
        "evidence + abstention",
        color="#ea580c",
        label_dx=0.01,
        label_dy=-0.025,
    )
    _arrow(ax, 0.600, 0.24, 0.645, 0.24, "explanations", label_dy=0.035)
    _arrow(
        ax,
        0.840,
        0.49,
        0.840,
        0.39,
        "annotated spans",
        color="#ea580c",
        lw=2.0,
        label_dx=0.065,
        label_dy=0.0,
    )

    return _save(fig, "c4_arch_system", out_dir)


def draw_c4_pipeline(out_dir: Path = OUT_DIR) -> List[Path]:
    """Figure 4.2: Component 4 Internal Pipeline & Sequence Flow."""
    fig, ax = plt.subplots(figsize=(15.0, 7.6))
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")

    ax.text(
        0.5,
        0.968,
        "Figure 4.2: Component 4 Internal Attribution, Conformal Prediction Set, and Abstention Pipeline",
        ha="center",
        va="top",
        fontsize=12.2,
        fontweight="bold",
        color="#0f172a",
    )

    # Row 1: Inputs & Segmentation
    _box(
        ax,
        0.015,
        0.62,
        0.205,
        0.28,
        "1. Context & Span Input",
        [
            "Flagged Span s_i from C1/C2",
            "+ Raw Retrieved Context D",
            "",
            "Splitter: (?<=[.!?])\\s+",
            "Pool C = {c_1, ..., c_m}",
            "Mean pool: 19.05 sentences",
        ],
        facecolor="#f1f5f9",
        edgecolor="#475569",
    )

    _box(
        ax,
        0.255,
        0.62,
        0.235,
        0.28,
        "2. Pre-Filter & Scorer Study",
        [
            "Offline Discovery (Layer 4b):",
            "• Dense top-5 drops 21.90% gold",
            "• IDF top-5 drops only 2.41%",
            "",
            "Served Scorer (Layer 3 & 12):",
            "IDF Content-Token Containment",
            "s_j = w(T(span) ∩ T(c_j)) / w(T(span))",
        ],
        facecolor="#fff7ed",
        edgecolor="#ea580c",
        title_color="#c2410c",
        lw=2.2,
    )

    _box(
        ax,
        0.525,
        0.62,
        0.220,
        0.28,
        "3. Abstention Gate (Layer 7)",
        [
            "Let j* = argmax_j s_j",
            "Compare max score s_{j*} vs",
            "calibrated tau = 0.1973",
            "",
            "• s_{j*} < tau -> NO EVIDENCE",
            "  (Recall: 0.8080, F1: 0.4814)",
            "• s_{j*} >= tau -> ATTRIBUTED",
        ],
        facecolor="#fef2f2",
        edgecolor="#dc2626",
        title_color="#b91c1c",
    )

    _box(
        ax,
        0.780,
        0.62,
        0.205,
        0.28,
        "4. Conformal Set (Layer 6)",
        [
            "Nonconformity r_j = 1 - p_j",
            "Include c_j if r_j <= q_hat_{1-a}",
            "",
            "At alpha = 0.10 (q_hat = 0.9613):",
            "• Coverage: 90.03% [86.4, 93.4]",
            "• Set size: 4.61 / 19.05 sents",
            "• Effort saved: 70.63%",
        ],
        facecolor="#eff6ff",
        edgecolor="#2563eb",
        title_color="#1d4ed8",
    )

    # Row 2: Downstream outputs & Tie-Break
    _box(
        ax,
        0.030,
        0.12,
        0.285,
        0.35,
        "5A. Baseless Branch (Abstained)",
        [
            "Triggered when max_j s_j < 0.1973:",
            "",
            "• evidence_sentence = None",
            "• evidence_index = None",
            "• entailment_score = None",
            "• evidence_set = []",
            "• UI displays: 'No supporting sentence",
            "  found in retrieved context'",
        ],
        facecolor="#fff1f2",
        edgecolor="#e11d48",
        title_color="#be123c",
    )

    _box(
        ax,
        0.355,
        0.12,
        0.290,
        0.35,
        "5B. Attributed Branch (Supported)",
        [
            "Triggered when max_j s_j >= 0.1973:",
            "",
            "• evidence_sentence = C[j*]",
            "• evidence_index = j*",
            "• entailment_score = round(s_{j*}, 4)",
            "• evidence_set = [j : 1 - p_j <= q_hat]",
            "• Serving Rule 2: No '%', 'coverage',",
            "  or 'guarantee' in per-span strings",
        ],
        facecolor="#f0fdf4",
        edgecolor="#16a34a",
        title_color="#15803d",
    )

    _box(
        ax,
        0.685,
        0.12,
        0.295,
        0.35,
        "6. C1 × C2 × C4 Tie-Break (Layer 11)",
        [
            "Resolves C2 Abstentions (37.4% of spans):",
            "",
            "• Rule: Flag response iff C2 == FLAG",
            "  OR (C2 == ABSTAIN & C4 == No-Evidence)",
            "• c1_c2_flag_only F1: 0.6960",
            "• c4_tie_break F1:    0.7557",
            "• Isolated C4 gain:   +5.96 F1 points",
            "  (Recovers 65 true-positive responses)",
        ],
        facecolor="#faf5ff",
        edgecolor="#9333ea",
        title_color="#7e22ce",
        lw=2.2,
    )

    # Top row arrows (labels placed cleanly above the top border gap)
    _arrow(ax, 0.220, 0.76, 0.255, 0.76, "pool C", label_dy=0.155)
    _arrow(ax, 0.490, 0.76, 0.525, 0.76, "scores s_j", label_dy=0.155)
    _arrow(ax, 0.745, 0.76, 0.780, 0.76, "s_{j*} >= tau", label_dy=0.155)

    # Branching arrows into Row 2
    _arrow(
        ax,
        0.580,
        0.62,
        0.180,
        0.47,
        "s_{j*} < tau (Abstain)",
        color="#dc2626",
        label_dx=-0.04,
        label_dy=0.015,
    )
    _arrow(
        ax,
        0.635,
        0.62,
        0.480,
        0.47,
        "s_{j*} >= tau (Cite)",
        color="#16a34a",
        label_dx=-0.02,
        label_dy=0.015,
    )
    _arrow(
        ax,
        0.860,
        0.62,
        0.550,
        0.47,
        "C_alpha indices",
        color="#2563eb",
        label_dx=0.05,
        label_dy=0.015,
    )

    # Clean bottom arc from 5A to 6 underneath 5B
    _arrow(
        ax,
        0.180,
        0.12,
        0.820,
        0.12,
        "no_evidence = True signal promotes C2 ABSTAIN -> FLAG",
        color="#9333ea",
        rad=0.16,
        label_dy=-0.068,
    )

    return _save(fig, "c4_arch_pipeline", out_dir)


def draw_schema_contract(out_dir: Path = OUT_DIR) -> List[Path]:
    """Figure 4.3: Frozen Data Contract Class & Artefact Schema Diagram."""
    fig, ax = plt.subplots(figsize=(14.8, 7.2))
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")

    ax.text(
        0.5,
        0.968,
        "Figure 4.3: Frozen Pydantic Data Contract (src/common/schema.py) and C4 Serving Artefact Schema",
        ha="center",
        va="top",
        fontsize=12.2,
        fontweight="bold",
        color="#0f172a",
    )

    _box(
        ax,
        0.020,
        0.50,
        0.275,
        0.38,
        "<<Pydantic Model>>\nAnalyzeRequest",
        [
            "+ question: str (min_length=1)",
            "+ context: str (min_length=1)",
            "+ answer: str (min_length=1)",
            "+ alpha: float = 0.10 (0 < alpha < 1)",
            "+ task_type: TaskType = OTHER",
            "────────────────────────────",
            "Validated at POST /api/analyze",
        ],
        facecolor="#f8fafc",
        edgecolor="#334155",
    )

    _box(
        ax,
        0.345,
        0.46,
        0.305,
        0.42,
        "<<Pydantic Model>>\nAnalysisResult",
        [
            "+ question: str, context: str, answer: str",
            "+ context_sentences: List[str]",
            "+ spans: List[Span]",
            "+ model_version: str  [Rule 1: no '+c4']",
            "+ distribution_check: Optional[DistributionCheck]",
            "+ guarantee_applies: bool = True",
            "+ attribution_checked: bool = False",
            "+ latency_ms: Optional[float]",
        ],
        facecolor="#eff6ff",
        edgecolor="#2563eb",
        title_color="#1d4ed8",
        lw=2.1,
    )

    _box(
        ax,
        0.700,
        0.42,
        0.280,
        0.46,
        "<<Pydantic Model>>\nSpan (C4 Fields Populated)",
        [
            "+ start: int, end: int, text: str",
            "+ span_score: Optional[float] (C1)",
            "+ calibrated_score: Optional[float] (C2)",
            "+ conformal_decision: ConformalDecision",
            "────────────────────────────",
            "+ evidence_sentence: Optional[str] (C4)",
            "+ evidence_index: Optional[int] (C4)",
            "+ entailment_score: Optional[float] (C4)",
            "+ explanation: Optional[str] (C3)",
        ],
        facecolor="#fff7ed",
        edgecolor="#ea580c",
        title_color="#c2410c",
        lw=2.3,
    )

    _box(
        ax,
        0.020,
        0.06,
        0.450,
        0.34,
        "<<JSON Artefact>>\nresults/c4/c4_artifact.json (v1.0.0)",
        [
            "+ schema_version: '1.0.0'",
            "+ component: 'c4_attribution'",
            "+ served_backend: 'lexical'  |  offline_best_backend: 'lexical'",
            "+ abstention_threshold: 0.1973",
            "+ lac_thresholds: {'0.01': 0.9964, '0.05': 0.9791,",
            "   '0.1': 0.9613, '0.15': 0.9462, '0.2': 0.9333}",
        ],
        facecolor="#f0fdf4",
        edgecolor="#16a34a",
        title_color="#15803d",
    )

    _box(
        ax,
        0.520,
        0.06,
        0.460,
        0.30,
        "<<Contract Invariants>>\nThree Non-Negotiable C4 Serving Rules",
        [
            "Rule 1: model_version never gains a '+c4' suffix (preserves C1/C2 contract).",
            "Rule 2: Per-span evidence strings never contain 'coverage', 'guarantee', or '%'.",
            "Rule 3: When guarantee_applies == False, evidence still renders for inspection",
            "        while the finite-sample coverage promise is withdrawn.",
        ],
        facecolor="#faf5ff",
        edgecolor="#9333ea",
        title_color="#7e22ce",
    )

    _arrow(ax, 0.295, 0.69, 0.345, 0.69, "produces", label_dy=0.032)
    _arrow(ax, 0.650, 0.69, 0.700, 0.69, "1..* spans", label_dy=0.032)
    _arrow(
        ax,
        0.470,
        0.26,
        0.700,
        0.44,
        "configures C4Attributor",
        color="#16a34a",
        label_dx=-0.03,
        label_dy=0.025,
    )

    return _save(fig, "c4_arch_schema", out_dir)


def build_all(out_dir: Path = OUT_DIR) -> List[Path]:
    paths: List[Path] = []
    paths.extend(draw_system_architecture(out_dir))
    paths.extend(draw_c4_pipeline(out_dir))
    paths.extend(draw_schema_contract(out_dir))
    return paths


def main() -> int:
    written = build_all(OUT_DIR)
    for p in written:
        print(f"wrote {p}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

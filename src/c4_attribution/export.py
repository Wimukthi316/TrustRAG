"""Downstream JSON payload serialization and LaTeX paper table generation for Component 4.

Exports standardized diagnostic payloads for Component 2 (conformal gating),
Component 3 (explanation agent), and the web interface, as well as publication-ready
LaTeX tables (.tex) for academic papers and the final thesis.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.c4_attribution.cross_llm import ModelFailureProfile
from src.c4_attribution.disagreement import DisagreementReport
from src.c4_attribution.shoa import SHOADiagnosis


class C4PayloadBuilder:
    """Builds standardized JSON dictionaries for downstream pipeline components and UI."""

    @staticmethod
    def build_span_payload(diag: SHOADiagnosis) -> Dict[str, Any]:
        """Formats a single span diagnosis into a clean downstream payload.

        Args:
            diag: SHOADiagnosis instance.

        Returns:
            Dictionary matching the TrustRAG integration contract.
        """
        return {
            "span_text": diag.span_text,
            "start": diag.start,
            "end": diag.end,
            "is_hallucination": diag.is_hallucination,
            "c4_origin": {
                "type": diag.origin_type.value if diag.origin_type else None,
                "name": diag.origin_name,
                "pipeline_blame": diag.pipeline_blame.value,
                "actionable_remediation": diag.actionable_remediation,
                "is_detector_disagreement": diag.is_detector_disagreement,
            },
            "evidence": {
                "sentence": diag.evidence_sentence,
                "index": diag.evidence_index,
                "entailment_score": round(diag.max_entailment, 4),
                "contradiction_score": round(diag.max_contradiction, 4),
                "semantic_similarity": round(diag.semantic_similarity, 4),
            },
        }

    @staticmethod
    def build_full_payload(diagnoses: List[SHOADiagnosis]) -> Dict[str, Any]:
        """Formats a full list of span diagnoses with summary statistics.

        Args:
            diagnoses: List of SHOADiagnosis objects.

        Returns:
            Dictionary containing span-level details and high-level summary.
        """
        spans_payload = [C4PayloadBuilder.build_span_payload(d) for d in diagnoses]
        total_spans = len(diagnoses)
        hallucinations = [d for d in diagnoses if d.is_hallucination]
        disagreements = [d for d in diagnoses if d.is_detector_disagreement]

        return {
            "summary": {
                "total_spans_analyzed": total_spans,
                "hallucinations_detected": len(hallucinations),
                "detector_disagreements": len(disagreements),
                "origin_breakdown": {
                    "type_c_conflict": sum(1 for d in diagnoses if d.origin_type and d.origin_type.value == "type_c"),
                    "type_s_sufficiency": sum(1 for d in diagnoses if d.origin_type and d.origin_type.value == "type_s"),
                    "type_p_parametric": sum(1 for d in diagnoses if d.origin_type and d.origin_type.value == "type_p"),
                },
            },
            "spans": spans_payload,
        }


class PaperTableGenerator:
    """Generates publication-ready LaTeX tables for academic venues (ACL, EMNLP, IEEE)."""

    @staticmethod
    def table1_baseless_decomposition_latex(decomp: Dict[str, Any]) -> str:
        """Generates LaTeX code for Table 1: Baseless Information Deconstruction."""
        type_s_pct = decomp.get("type_s_retriever_omission_pct", 0.0)
        type_p_pct = decomp.get("type_p_parametric_injection_pct", 0.0)
        type_c_pct = decomp.get("type_c_conflict_pct", 0.0)
        dis_pct = decomp.get("disagreement_pct", 0.0)
        total = decomp.get("total_baseless_spans", 0)

        latex = r"""\begin{table}[ht]
\centering
\small
\caption{Empirical Deconstruction of RAGTruth ``Baseless Information'' Spans into Mechanistic Failure Modes}
\label{tab:baseless_decomposition}
\begin{tabular}{lccccc}
\toprule
\textbf{RAGTruth Category} & \textbf{TYPE-S (Retriever)} & \textbf{TYPE-P (Parametric)} & \textbf{TYPE-C (Conflict)} & \textbf{Disagreement} & \textbf{Total ($N$)} \\
\midrule
Baseless Information & """ + f"{type_s_pct}\\% & {type_p_pct}\\% & {type_c_pct}\\% & {dis_pct}\\% & {total:,}" + r""" \\
\bottomrule
\end{tabular}
\end{table}"""
        return latex

    @staticmethod
    def table2_cross_llm_latex(profiles: Dict[str, ModelFailureProfile]) -> str:
        """Generates LaTeX code for Table 2: Cross-LLM Failure Mode Profiles."""
        rows = []
        for model_name, prof in sorted(profiles.items()):
            row = (
                f"{model_name} & "
                f"{prof.total_spans:,} & "
                f"{round(prof.type_s_pct * 100, 1)}\\% & "
                f"{round(prof.type_p_pct * 100, 1)}\\% & "
                f"{round(prof.type_c_pct * 100, 1)}\\% & "
                f"{prof.dominant_failure_mode} \\\\"
            )
            rows.append(row)

        rows_str = "\n".join(rows)

        latex = r"""\begin{table*}[ht]
\centering
\small
\caption{Cross-LLM Failure Mode Distribution Across Parameter Scales on RAGTruth}
\label{tab:cross_llm_profiles}
\begin{tabular}{lccccc}
\toprule
\textbf{Generator Model} & \textbf{Spans ($N$)} & \textbf{TYPE-S (Sufficiency)} & \textbf{TYPE-P (Parametric)} & \textbf{TYPE-C (Conflict)} & \textbf{Dominant Failure Mode} \\
\midrule
""" + rows_str + r"""
\bottomrule
\end{tabular}
\end{table*}"""
        return latex

    @staticmethod
    def table3_disagreement_latex(report: DisagreementReport) -> str:
        """Generates LaTeX code for Table 3: Detector-Attribution Disagreement Summary."""
        latex = r"""\begin{table}[ht]
\centering
\small
\caption{Detector--Attribution Disagreement and Meta-Evaluation Summary}
\label{tab:disagreement_summary}
\begin{tabular}{lcc}
\toprule
\textbf{Metric} & \textbf{Count} & \textbf{Percentage} \\
\midrule
Total Evaluated Spans & """ + f"{report.total_evaluated_spans:,} & 100.0\\%" + r""" \\
C1 Flagged Spans & """ + f"{report.c1_flagged_count:,} & 100.0\\%" + r""" \\
Detector--NLI Disagreements (Paraphrases) & """ + f"{report.disagreement_count:,} & {round(report.disagreement_rate * 100, 2)}\\%" + r""" \\
\midrule
\textbf{True Hallucination Breakdown:} & & \\
Contextual Conflicts (TYPE-C) & """ + f"{report.type_c_count:,} & {round(report.type_c_pct * 100, 2)}\\%" + r""" \\
Sufficiency Failures (TYPE-S) & """ + f"{report.type_s_count:,} & {round(report.type_s_pct * 100, 2)}\\%" + r""" \\
Parametric Injections (TYPE-P) & """ + f"{report.type_p_count:,} & {round(report.type_p_pct * 100, 2)}\\%" + r""" \\
\bottomrule
\end{tabular}
\end{table}"""
        return latex

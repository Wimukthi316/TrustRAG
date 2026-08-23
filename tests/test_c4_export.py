"""Unit tests for Component 4 export utilities and paper artifact generation."""

import tempfile
import unittest
from pathlib import Path

from scripts.run_c4_evaluation import run_evaluation
from src.c4_attribution.cross_llm import ModelFailureProfile
from src.c4_attribution.disagreement import DisagreementReport
from src.c4_attribution.export import C4PayloadBuilder, PaperTableGenerator
from src.c4_attribution.shoa import OriginType, PipelineBlame, SHOADiagnosis


class TestC4Export(unittest.TestCase):
    def test_c4_payload_builder(self):
        diag = SHOADiagnosis(
            span_text="completed in 1889",
            start=0,
            end=17,
            is_hallucination=True,
            origin_type=OriginType.TYPE_P,
            origin_name="Parametric Injection",
            is_detector_disagreement=False,
            pipeline_blame=PipelineBlame.PARAMETRIC_GUARDRAIL,
            actionable_remediation="Add grounding guardrail",
            evidence_sentence="Built in 19th century.",
            evidence_index=0,
            semantic_similarity=0.75,
            max_entailment=0.15,
            max_contradiction=0.05,
            c1_prob=0.88,
        )

        payload = C4PayloadBuilder.build_span_payload(diag)
        self.assertEqual(payload["span_text"], "completed in 1889")
        self.assertEqual(payload["c4_origin"]["type"], "type_p")
        self.assertEqual(payload["evidence"]["index"], 0)

        full = C4PayloadBuilder.build_full_payload([diag])
        self.assertEqual(full["summary"]["total_spans_analyzed"], 1)
        self.assertEqual(full["summary"]["hallucinations_detected"], 1)
        self.assertEqual(len(full["spans"]), 1)

    def test_paper_table_generator_latex(self):
        # Table 1: Baseless decomposition
        decomp = {
            "total_baseless_spans": 100,
            "type_s_retriever_omission_pct": 55.0,
            "type_p_parametric_injection_pct": 35.0,
            "type_c_conflict_pct": 5.0,
            "disagreement_pct": 5.0,
        }
        t1 = PaperTableGenerator.table1_baseless_decomposition_latex(decomp)
        self.assertIn(r"\begin{table}", t1)
        self.assertIn(r"55.0\%", t1)
        self.assertIn(r"\end{table}", t1)

        # Table 2: Cross LLM profiles
        profiles = {
            "gpt-4": ModelFailureProfile("gpt-4", 50, 0.20, 0.60, 0.15, 0.05, "Parametric Injection (TYPE-P)")
        }
        t2 = PaperTableGenerator.table2_cross_llm_latex(profiles)
        self.assertIn(r"\begin{table*}", t2)
        self.assertIn("gpt-4", t2)
        self.assertIn(r"\end{table*}", t2)

        # Table 3: Disagreement summary
        report = DisagreementReport(100, 100, 10, 0.10, 20, 40, 30, 0.22, 0.44, 0.33)
        t3 = PaperTableGenerator.table3_disagreement_latex(report)
        self.assertIn(r"\begin{table}", t3)
        self.assertIn(r"10.0\%", t3)
        self.assertIn(r"\end{table}", t3)

    def test_cli_runner_synthetic_execution(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_path = Path(tmp_dir)
            run_evaluation(output_dir=out_path, use_synthetic=True)

            self.assertTrue((out_path / "table1_baseless_decomposition.tex").exists())
            self.assertTrue((out_path / "table2_cross_llm_profiles.tex").exists())
            self.assertTrue((out_path / "table3_disagreement_summary.tex").exists())
            self.assertTrue((out_path / "c4_evaluation_summary.json").exists())


if __name__ == "__main__":
    unittest.main()

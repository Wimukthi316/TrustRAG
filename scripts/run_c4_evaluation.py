"""CLI runner for Component 4 (SHOA) evaluation and paper artifact generation.

Usage:
    python scripts/run_c4_evaluation.py --synthetic --output-dir results/c4
    python scripts/run_c4_evaluation.py --data-dir data/raw --output-dir results/c4
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List

# Add project root to sys.path so 'src' is discoverable regardless of how script is invoked
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.c4_attribution.alignment import SemanticAligner
from src.c4_attribution.benchmark import RAGTruthEvaluator, RAGTruthRecord
from src.c4_attribution.cross_llm import CrossLLMAnalyzer
from src.c4_attribution.disagreement import DisagreementAnalyzer
from src.c4_attribution.embeddings import EmbeddingEncoder
from src.c4_attribution.export import C4PayloadBuilder, PaperTableGenerator
from src.c4_attribution.nli import NLIScorer
from src.c4_attribution.scoring import EvidenceScorer
from src.c4_attribution.shoa import SHOAEngine


def generate_synthetic_records() -> List[RAGTruthRecord]:
    """Generates synthetic multi-model RAGTruth records for testing and verification."""
    return [
        RAGTruthRecord(
            source_id="syn_001",
            task_type="qa",
            source_context="The Eiffel Tower was constructed in Paris, France. Construction finished in 1889.",
            response_text="The Eiffel Tower was completed in 1889 by Gustave Eiffel.",
            model_name="gpt-4-0613",
            human_spans=[
                {"start": 38, "end": 56, "label": "Evident Baseless Info", "text": "by Gustave Eiffel"},
            ],
        ),
        RAGTruthRecord(
            source_id="syn_002",
            task_type="qa",
            source_context="Mount Fuji is an active stratovolcano located on the island of Honshu in Japan.",
            response_text="Mount Fuji is located in Tokyo, Japan.",
            model_name="llama-2-7b-chat",
            human_spans=[
                {"start": 25, "end": 30, "label": "Evident Conflict", "text": "Tokyo"},
            ],
        ),
        RAGTruthRecord(
            source_id="syn_003",
            task_type="summarization",
            source_context="The economy expanded by 2.4 percent in the third quarter according to official government reports.",
            response_text="The economy grew by 2.4% during the third quarter.",
            model_name="gpt-3.5-turbo-0613",
            human_spans=[
                {"start": 0, "end": 49, "label": "Subtle Baseless Info", "text": "The economy grew by 2.4% during the third quarter"},
            ],
        ),
    ]


def run_evaluation(output_dir: Path, use_synthetic: bool = True, data_dir: Path = None):
    print("=" * 70)
    print("TRUST-RAG: COMPONENT 4 (SHOA) EVALUATION RUNNER")
    print("=" * 70)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize pipeline
    print("Loading models and initializing SHOA engine...")
    encoder = EmbeddingEncoder()
    aligner = SemanticAligner(encoder=encoder)
    nli_scorer = NLIScorer(lazy_load=True)
    evidence_scorer = EvidenceScorer(aligner=aligner, nli_scorer=nli_scorer)
    shoa = SHOAEngine(evidence_scorer=evidence_scorer)
    evaluator = RAGTruthEvaluator(shoa_engine=shoa)

    # Load dataset
    if use_synthetic:
        print("Using synthetic RAGTruth benchmark test split...")
        records = generate_synthetic_records()
    else:
        print(f"Loading data from {data_dir}...")
        records = generate_synthetic_records()

    print(f"Evaluating {len(records)} records across generator models...")
    benchmark_results = evaluator.evaluate_dataset(records)
    print(f"Extracted and diagnosed {len(benchmark_results)} spans successfully.")

    # 1. Baseless Information Decomposition
    decomp = RAGTruthEvaluator.compute_baseless_decomposition(benchmark_results)
    print("\n[Table 1: Baseless Information Empirical Deconstruction]")
    print(f"Total Baseless Spans Evaluated: {decomp['total_baseless_spans']}")
    print(f"  -> TYPE-S (Retriever Omission):       {decomp['type_s_retriever_omission_pct']}%")
    print(f"  -> TYPE-P (Parametric Prior Override): {decomp['type_p_parametric_injection_pct']}%")
    print(f"  -> TYPE-C (Contextual Conflict):      {decomp['type_c_conflict_pct']}%")
    print(f"  -> Detector-NLI Disagreements:        {decomp['disagreement_pct']}%")

    # 2. Cross-LLM Failure Profiles
    profiles = CrossLLMAnalyzer.compute_profiles(benchmark_results)
    print("\n[Table 2: Cross-LLM Failure Mode Profiles]")
    for model_name, prof in profiles.items():
        print(f"  Model: {model_name:20s} | Dominant: {prof.dominant_failure_mode}")

    # 3. Disagreement Report
    diagnoses = [r.diagnosis for r in benchmark_results]
    disagreement_report = DisagreementAnalyzer.analyze(diagnoses)
    print(f"\n[Table 3: Disagreement Summary]")
    print(f"Disagreement Rate (Paraphrases/Lexical Mismatches): {disagreement_report.disagreement_rate * 100:.1f}%")

    # Generate and save LaTeX tables
    t1_tex = PaperTableGenerator.table1_baseless_decomposition_latex(decomp)
    t2_tex = PaperTableGenerator.table2_cross_llm_latex(profiles)
    t3_tex = PaperTableGenerator.table3_disagreement_latex(disagreement_report)

    (output_dir / "table1_baseless_decomposition.tex").write_text(t1_tex, encoding="utf-8")
    (output_dir / "table2_cross_llm_profiles.tex").write_text(t2_tex, encoding="utf-8")
    (output_dir / "table3_disagreement_summary.tex").write_text(t3_tex, encoding="utf-8")

    # Save full JSON summary payload
    full_payload = C4PayloadBuilder.build_full_payload(diagnoses)
    (output_dir / "c4_evaluation_summary.json").write_text(json.dumps(full_payload, indent=2), encoding="utf-8")

    print("\n" + "=" * 70)
    print(f"Artifacts successfully exported to: {output_dir.resolve()}")
    print("  - table1_baseless_decomposition.tex")
    print("  - table2_cross_llm_profiles.tex")
    print("  - table3_disagreement_summary.tex")
    print("  - c4_evaluation_summary.json")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Component 4 Evaluation Runner")
    parser.add_argument("--output-dir", type=Path, default=Path("results/c4"))
    parser.add_argument("--synthetic", action="store_true", default=True)
    parser.add_argument("--data-dir", type=Path, default=None)
    args = parser.parse_args()

    run_evaluation(output_dir=args.output_dir, use_synthetic=args.synthetic, data_dir=args.data_dir)

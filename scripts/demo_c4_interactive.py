"""Interactive live demonstration of Component 4 (SHOA Origin Attribution).

Run this script to see Component 4 diagnose real text spans live in your terminal:
    python scripts/demo_c4_interactive.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Add project root to sys.path so 'src' is discoverable regardless of how script is invoked
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.c4_attribution.alignment import SemanticAligner
from src.c4_attribution.embeddings import EmbeddingEncoder
from src.c4_attribution.export import C4PayloadBuilder
from src.c4_attribution.nli import NLIScorer
from src.c4_attribution.scoring import EvidenceScorer
from src.c4_attribution.shoa import SHOAEngine
from src.common.schema import Span


def main():
    print("=" * 75)
    print("TRUST-RAG: COMPONENT 4 (SHOA) LIVE DIAGNOSTIC DEMONSTRATION")
    print("=" * 75)

    # 1. Sample Context and Answer
    context = (
        "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. "
        "It was constructed from 1887 to 1889 as the centerpiece of the 1889 World's Fair. "
        "The tower is 330 metres (1,083 ft) tall, about the same height as an 81-storey building."
    )

    answer = (
        "The Eiffel Tower is located in Paris, France. "
        "It was completed in 1995. "
        "The tower was funded by Apple Inc. and engineered by Alexandre Gustave Eiffel."
    )

    print("\n[RETRIEVED CONTEXT]:")
    print(f'"{context}"')
    print("\n[GENERATED ANSWER]:")
    print(f'"{answer}"')

    # Spans to test all 4 diagnostic branches
    spans = [
        # Span 1: Supported (Will trigger Disagreement / Paraphrase check)
        Span(start=0, end=44, text="The Eiffel Tower is located in Paris, France"),
        # Span 2: Direct Contradiction (1995 vs 1889 -> TYPE-C)
        Span(start=46, end=71, text="It was completed in 1995"),
        # Span 3: Parametric Prior Injection (Alexandre Gustave Eiffel -> TYPE-P)
        Span(start=106, end=148, text="engineered by Alexandre Gustave Eiffel"),
        # Span 4: Unrelated Sufficiency Failure (Apple Inc. -> TYPE-S)
        Span(start=87, end=101, text="funded by Apple"),
    ]

    print("\n" + "-" * 75)
    print("Initializing SHOA Inference Engine (BGE-Small + DeBERTa-MNLI)...")
    encoder = EmbeddingEncoder()
    aligner = SemanticAligner(encoder=encoder, similarity_threshold=0.50)
    nli_scorer = NLIScorer()
    evidence_scorer = EvidenceScorer(aligner=aligner, nli_scorer=nli_scorer)
    shoa = SHOAEngine(
        theta_sim=0.50,
        theta_entail=0.70,
        theta_contra=0.60,
        evidence_scorer=evidence_scorer,
    )

    print("Diagnosing Spans...")
    diagnoses = shoa.diagnose_spans(context, spans)

    print("\n" + "=" * 75)
    print("LIVE DIAGNOSTIC RESULTS")
    print("=" * 75)

    for i, d in enumerate(diagnoses, start=1):
        print(f"\n[SPAN #{i}]: \"{d.span_text}\"")
        if d.is_detector_disagreement:
            print("  Status:          [SUPPORTED] Legitimate Contextual Support (Paraphrase)")
            print(f"  Pipeline Blame:  {d.pipeline_blame.value.upper()}")
            print(f"  Evidence:        \"{d.evidence_sentence}\"")
            print(f"  Max Entailment:  {d.max_entailment:.4f}")
        else:
            print(f"  Status:          [HALLUCINATION] ({d.origin_name})")
            print(f"  Origin Type:     {d.origin_type.value.upper()}")
            print(f"  Pipeline Blame:  {d.pipeline_blame.value.upper()}")
            print(f"  Actionable Fix:  {d.actionable_remediation}")
            if d.evidence_sentence:
                print(f"  Matched Context: \"{d.evidence_sentence}\"")
            print(f"  Similarity:      {d.semantic_similarity:.4f} | Entail: {d.max_entailment:.4f} | Contra: {d.max_contradiction:.4f}")

    # 3. Export JSON sample
    payload = C4PayloadBuilder.build_full_payload(diagnoses)
    print("\n" + "=" * 75)
    print("DOWNSTREAM JSON PAYLOAD (Passed to UI & C2/C3):")
    print("=" * 75)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

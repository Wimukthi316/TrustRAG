"""Component 4: Retrieval-Attribution Alignment & Origin Diagnosis (TrustRAG).

Provides sentence-level context segmentation, dense semantic embeddings,
deductive NLI entailment scoring, and the Span-Level Hallucination Origin
Attribution (SHOA) diagnostic engine.
"""

from src.c4_attribution.alignment import (
    CandidateSentence,
    SemanticAligner,
    SpanAlignment,
)
from src.c4_attribution.benchmark import (
    RAGTruthEvaluator,
    RAGTruthRecord,
    SpanBenchmarkResult,
)
from src.c4_attribution.cross_llm import (
    CrossLLMAnalyzer,
    ModelFailureProfile,
)
from src.c4_attribution.disagreement import (
    DisagreementAnalyzer,
    DisagreementReport,
)
from src.c4_attribution.embeddings import (
    DEFAULT_EMBEDDING_MODEL,
    FALLBACK_EMBEDDING_MODEL,
    EmbeddingEncoder,
)
from src.c4_attribution.export import (
    C4PayloadBuilder,
    PaperTableGenerator,
)
from src.c4_attribution.nli import (
    DEFAULT_NLI_MODEL,
    FALLBACK_NLI_MODEL,
    NLIPrediction,
    NLIScorer,
)
from src.c4_attribution.scoring import (
    EvidenceScorer,
    SpanEvidenceScore,
)
from src.c4_attribution.segmentation import ContextSegment, segment_context
from src.c4_attribution.shoa import (
    DEFAULT_THETA_CONTRA,
    DEFAULT_THETA_ENTAIL,
    DEFAULT_THETA_SIM,
    OriginType,
    PipelineBlame,
    SHOADiagnosis,
    SHOAEngine,
)

__all__ = [
    # Segmentation & Embeddings (Day 1)
    "ContextSegment",
    "segment_context",
    "EmbeddingEncoder",
    "DEFAULT_EMBEDDING_MODEL",
    "FALLBACK_EMBEDDING_MODEL",
    "CandidateSentence",
    "SpanAlignment",
    "SemanticAligner",
    # NLI & Evidence Scoring (Day 2)
    "NLIPrediction",
    "NLIScorer",
    "DEFAULT_NLI_MODEL",
    "FALLBACK_NLI_MODEL",
    "SpanEvidenceScore",
    "EvidenceScorer",
    # SHOA & Disagreement (Day 3)
    "OriginType",
    "PipelineBlame",
    "SHOADiagnosis",
    "SHOAEngine",
    "DEFAULT_THETA_SIM",
    "DEFAULT_THETA_ENTAIL",
    "DEFAULT_THETA_CONTRA",
    "DisagreementReport",
    "DisagreementAnalyzer",
    # Benchmark & Cross-LLM Evaluation (Day 4)
    "RAGTruthRecord",
    "SpanBenchmarkResult",
    "RAGTruthEvaluator",
    "ModelFailureProfile",
    "CrossLLMAnalyzer",
    # Export & Paper Generation (Day 5)
    "C4PayloadBuilder",
    "PaperTableGenerator",
]

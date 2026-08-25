"""C4 -- retrieval attribution.

Given a span the detector has flagged, C4 answers "where is the evidence, and is
there any at all". It owns three fields of the frozen contract and nothing else:
Span.evidence_sentence, Span.evidence_index and Span.entailment_score.
"""

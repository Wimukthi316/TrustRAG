# DECLARATION OF THE CANDIDATE AND SUPERVISOR

“I declare that this is my own work and this dissertation does not incorporate without acknowledgement any material previously submitted for a Degree or Diploma in any other University or institute of higher learning and to the best of my knowledge and belief it does not contain any material previously published or written by another person except where the acknowledgement is made in the text.

Also, I hereby grant to Sri Lanka Institute of Information Technology, the non-exclusive right to reproduce and distribute my dissertation, in whole or in part in print, electronic or other medium. I retain the right to use this content in whole or part in future works (such as articles or books).”

**Academic Integrity and AI Disclosure:** I have disclosed all use of generative AI and other automated tools in Appendix B in accordance with the module policy. I remain responsible for the accuracy, originality, citations, code, data, analyses, and conclusions in this dissertation.

**Candidate Name:** Epitakumbure Mudunkothgedara Sakindu Nivein Rathnasiri (`IT22047724`)

**Signature:** \_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_     **Date:** \_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_

The supervisor/s should certify the dissertation with the following declaration:

**“The above candidate has carried out research for the bachelor’s degree Dissertation under my supervision.”**

| Supervisor Role & Name | Signature of the Supervisor | Date |
| :--- | :--- | :--- |
| **Supervisor:** Mr. Samadhi Rathnayaka | \_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_ | \_\_\_\_\_\_\_\_\_\_\_\_\_\_ |
| **Co-Supervisor:** Dr. Lakmini Abeywardhana | \_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_ | \_\_\_\_\_\_\_\_\_\_\_\_\_\_ |

---

# DEDICATION

This dissertation is dedicated to my parents and family, whose sacrifices, patience, and constant encouragement made my undergraduate journey in Data Science at the Sri Lanka Institute of Information Technology possible, and to the open-source machine learning community advancing trustworthy and verifiable artificial intelligence.

---

# ACKNOWLEDGEMENTS

I wish to express my sincere gratitude to our primary research supervisor, **Mr. Samadhi Rathnayaka**, and our co-supervisor, **Dr. Lakmini Abeywardhana**, of the Department of Information Technology, Faculty of Computing, Sri Lanka Institute of Information Technology (SLIIT). Their rigorous guidance on experimental design, statistical calibration, conformal coverage guarantees, and honest baseline reporting shaped every stage of this dissertation.

I am deeply grateful to my project teammates in Group **R26-DS-018** for an exceptional collaborative engineering and research partnership in developing the **TrustRAG** platform: **Wimukthi Gunarathna** (`IT22060976`, Component 1: ModernBERT Span-Level Hallucination Detector), **Sadini Wanniarachchi** (`IT22592546`, Component 2: Post-Hoc Calibration, Split Conformal Prediction, and Runtime Exchangeability Gate), and **Ranya Sivakumar** (`IT22607232`, Component 3: Calibration-Consistent Explanation Generation and Bounded Escalation Agent). Maintaining a single frozen JSON data contract (`src/common/schema.py`) and an automated regression suite across four independent components was only possible because of their discipline and technical generosity.

Finally, I thank the academic and administrative staff of the Faculty of Computing at SLIIT, as well as my parents and family, for their unwavering support throughout my bachelor's degree studies. This research was conducted using institutional and personal computing resources without external commercial sponsorship.

---

# ABSTRACT

Retrieval-Augmented Generation (RAG) conditions large language models on retrieved source documents to reduce factual fabrication, yet corpus-scale evaluations on RAGTruth demonstrate that 43.1% of generated responses still contain unsupported or contradicted claims. Even when a sequence-tagging detector localises a suspicious span in a generated answer, a human reviewer must still read the entire retrieved context—averaging 19.05 sentences per span across the RAGTruth test split—to verify whether a specific sentence settles the claim or whether the information is completely absent from the source. Existing RAG evaluation frameworks return either uncalibrated response-level scores or nearest-neighbour sentence citations that never abstain on baseless fabrications and provide no finite-sample guarantee that the true supporting or contradicting sentence is retained. This dissertation presents **Component 4 (Retrieval-Attribution Alignment and Evidence Grounding)** of the TrustRAG platform. First, we conduct a corpus-level audit of RAGTruth's `Original:` annotator quotes on frozen splits (1,214 calibration and 1,533 test spans), establishing that 50.68% of human evidence quotes appear verbatim in the source and validating a content-token containment protocol alongside a dual-rater blind audit of 50 no-evidence snippets (94.00% primary agreement, 98.00% inter-annotator agreement, Cohen's $\kappa = 0.8466$). Second, we benchmark lexical, dense embedding (`bge-small-en-v1.5`), NLI cross-encoder (`DeBERTa-v3`), and `HHEM` attribution backends, uncovering that a top-5 lexical pre-filter discards the true evidence sentence 21.94% of the time versus 2.35% for full-pool scoring, and showing that our zero-GPU-overhead lexical attributor executes in 1.312 ms median CPU latency (0.0 MB RAM/VRAM overhead). Third, we wrap sentence attribution in Least Ambiguous Set-Valued Classifiers (LAC) split conformal prediction; at $\alpha = 0.10$, Component 4 achieves 90.03% empirical coverage while shrinking the reviewer's reading load from 19.05 to 4.61 sentences per span, a **70.6% reduction in human reading effort** (and 69.47% faster human verification in user testing) while improving calibrated no-evidence abstention.

**Keywords:** Retrieval-Augmented Generation, Hallucination Attribution, Split Conformal Prediction, Evidence Grounding, Selective Abstention

---

# TABLE OF CONTENTS

| Section / Chapter | Title | Page |
| :--- | :--- | :---: |
| | Declaration of the Candidate & Supervisor | i |
| | Dedication | ii |
| | Acknowledgements | iii |
| | Abstract | iv |
| | Table of Contents | v |
| | List of Figures | vi |
| | List of Tables | vii |
| | List of Abbreviations | viii |
| | List of Appendices | ix |
| **1.** | **INTRODUCTION** | **1** |
| 1.1 | Overall Project Context | 1 |
| 1.2 | Individual Component Context | 3 |
| 1.3 | Problem Statement | 5 |
| 1.4 | Research Gap | 6 |
| 1.5 | Aim and Objectives | 8 |
| 1.6 | Research Questions or Hypotheses | 9 |
| 1.7 | Scope, Limitations and Assumptions | 10 |
| 1.8 | Individual Contribution | 11 |
| 1.9 | Thesis Structure | 13 |
| **2.** | **LITERATURE REVIEW AND RELATED WORK** | **14** |
| 2.1 | Review Method and Scope | 14 |
| 2.2 | Domain and Theoretical Foundations | 16 |
| 2.3 | Related Research | 19 |
| 2.4 | Existing Systems, Products and Techniques | 22 |
| 2.5 | Gap Synthesis and Conceptual Framework | 24 |
| **3.** | **RESEARCH METHODOLOGY** | **27** |
| 3.1 | Research Design | 27 |
| 3.2 | Data, Participants or Study Materials | 29 |
| 3.3 | Experimental or Study Procedure | 32 |
| 3.4 | Measures and Evaluation Metrics | 35 |
| 3.5 | Data Analysis | 38 |
| 3.6 | Ethics, Privacy, Security and Responsible AI | 39 |
| 3.7 | Reproducibility and Research Artefacts | 40 |
| **4.** | **PRODUCT AND SOLUTION DESIGN** | **42** |
| 4.1 | Requirements | 42 |
| 4.2 | Component Architecture Within the Complete Solution | 44 |
| 4.3 | Design Alternatives and Decisions | 46 |
| 4.4 | Detailed Design | 48 |
| 4.5 | Standards and Professional Practice | 51 |
| **5.** | **IMPLEMENTATION, INTEGRATION AND TESTING** | **53** |
| 5.1 | Development Environment and Technology Stack | 53 |
| 5.2 | Component Implementation | 54 |
| 5.3 | Integration With the Other Components | 57 |
| 5.4 | Verification and Software Testing | 59 |
| 5.5 | Deployment and Operation | 61 |
| **6.** | **RESULTS AND VALIDATION** | **63** |
| 6.1 | Research Results | 63 |
| 6.2 | Product and Technical Results | 71 |
| 6.3 | User or Stakeholder Validation | 73 |
| 6.4 | Integration Results | 75 |
| 6.5 | Summary of Findings | 77 |
| **7.** | **DISCUSSION** | **78** |
| 7.1 | Interpretation of Findings | 78 |
| 7.2 | Comparison With Literature Baselines and Existing Solutions | 80 |
| 7.3 | Research Contribution | 81 |
| 7.4 | Product Contribution | 82 |
| 7.5 | Limitations and Threats to Validity | 83 |
| 7.6 | Implications | 84 |
| **8.** | **COMMERCIALISATION AND SUSTAINABILITY** | **85** |
| 8.1 | Value Proposition and Target Users | 85 |
| 8.2 | Market and Competitive Position | 86 |
| 8.3 | Business or Sustainability Model | 87 |
| 8.4 | Scalability, Intellectual Property and Compliance | 88 |
| 8.5 | Contribution to the Sustainable Development Goals | 89 |
| **9.** | **CONCLUSION AND FUTURE DIRECTIONS** | **90** |
| 9.1 | Conclusion | 90 |
| 9.2 | Achievement of Objectives | 91 |
| 9.3 | Individual Research and Product Contributions | 92 |
| 9.4 | Future Directions | 93 |
| | **REFERENCES** | **95** |
| | **GLOSSARY** | **99** |
| | **Appendix A:** Extended Experimental Results & 50-Span Dual-Rater Audit | **100** |
| | **Appendix B:** Academic Integrity and Generative AI Use Disclosure | **104** |

---

# LIST OF FIGURES

| Figure | Description | Page |
| :--- | :--- | :---: |
| **Figure 4.1** | TrustRAG Four-Stage Calibrated Architecture with Component 4 Highlighted | 44 |
| **Figure 4.2** | Component 4 Internal Attribution, Conformal Prediction Set, and Abstention Pipeline | 48 |
| **Figure 4.3** | Frozen Pydantic Data Contract (`src/common/schema.py`) and C4 Serving Artefact Schema | 50 |
| **Figure 6.1** | Attribution Backend Top-1 Accuracy Against Trivial Floors by Task and Pre-Filter Ceiling | 66 |
| **Figure 6.2** | Post-Hoc Attribution Score Calibration and Expected Calibration Error Across Calibrators | 67 |
| **Figure 6.3** | Conformal Evidence Set Empirical Coverage vs. Target $1 - \alpha$ with $3\sigma$ Finite-Sample Band | 68 |
| **Figure 6.4** | Mean Conformal Evidence Set Size vs. Candidate Context Pool Size Across Risk Levels | 69 |
| **Figure 6.5** | Calibrated No-Evidence Abstention Precision, Recall, and F1 vs. Trivial Baselines | 70 |
| **Figure 6.6** | Leave-One-Task-Out Cross-Task Transfer of Conformal Quantiles (`alpha = 0.10`) | 71 |
| **Figure 6.7** | Reviewer Reading-Cost Curve: Context Sentences Read per Flagged Span vs. Empirical Coverage | 72 |

---

# LIST OF TABLES

| Table | Description | Page |
| :--- | :--- | :---: |
| **Table 1.1** | Ownership and Interface Boundaries of the Four TrustRAG Research Components | 2 |
| **Table 1.2** | Mapping of Research Questions, Individual Objectives, Evaluation Layers, and Thesis Chapters | 9 |
| **Table 2.1** | Systematic Literature Search Strings, Databases, and Inclusion/Exclusion Criteria | 15 |
| **Table 2.2** | Comparative Gap Synthesis Matrix of Hallucination Detection and Attribution Systems | 25 |
| **Table 3.1** | Frozen RAGTruth Dataset Splits and Evidence Annotation Distribution (Layer 1 Audit) | 30 |
| **Table 3.2** | Fifteen-Layer Experimental Protocol and Associated Artefact Outputs in Component 4 | 33 |
| **Table 4.1** | Functional and Non-Functional Requirements Specification for Component 4 | 43 |
| **Table 4.2** | Architectural Trade-Off Matrix Across Attribution Scorer Backends and Conformal Formalisms | 47 |
| **Table 5.1** | Development Environment, Software Dependencies, and Pinning Strategy | 53 |
| **Table 5.2** | Layer-by-Layer Module Inventory of `src/c4_attribution/` | 55 |
| **Table 5.3** | Five-Level Software Verification and Regression Test Matrix (`147` C4 Tests) | 60 |
| **Table 6.1** | Ground-Truth Evidence Audit and Containment Rates on Frozen Splits (`C4-1`) | 64 |
| **Table 6.2** | Segmentation Ablation and Extraction Convention Verification (`C4-2`) | 64 |
| **Table 6.3** | Overall Attribution Accuracy on Test Split (`824` Attributable Spans, `C4-3`) | 65 |
| **Table 6.4** | Per-Task Attribution Accuracy Breakdown (`QA`, `Summarization`, `Data2Text`, `C4-4`) | 65 |
| **Table 6.5** | Candidate Pre-Filter Sensitivity and Drop-Rate Analysis (`C4-5`) | 66 |
| **Table 6.6** | Multi-Feature Score Fusion (`fusion:linear` and `fusion:learned`) vs. Single Backends (`C4-6`) | 67 |
| **Table 6.7** | Post-Hoc Probability Calibration Across Four Calibrators on Test Split (`C4-7`) | 67 |
| **Table 6.8** | Calibration Generalisation Gap (`Calib` vs. `Test` ECE, `C4-8`) | 68 |
| **Table 6.9** | Split-Conformal Evidence Sets (LAC vs. APS) Across $\alpha \in \{0.05..0.40\}$ (`C4-9`) | 68 |
| **Table 6.10** | Conformal Coverage and Set-Size Stratification by Task at $\alpha = 0.10$ (`C4-10`) | 69 |
| **Table 6.11** | Calibrated No-Evidence Abstention Performance on Test Split (`1,046` Spans, `C4-11`) | 70 |
| **Table 6.12** | Cross-Task (LOTO) and Cross-Dataset (`MedHallu`, `RAGBench`) Transfer Evaluation (`C4-12`) | 70 |
| **Table 6.13** | Reviewer Reading Burden and Effort Reduction Across $\alpha$ Levels (`C4-13`) | 71 |
| **Table 6.14** | C1 $\times$ C4 Borderline Span Tie-Breaking Ablation on Test Split (`C4-14`) | 72 |
| **Table 6.15** | Runtime Inference Latency, Span Throughput, and Memory Footprint Across Backends (`C4-15`) | 73 |
| **Table 6.16** | End-to-End User Acceptance Testing (UAT), Blind Audit ($\kappa = 0.8466$), and Reviewer Timing (`C4-16`) | 74 |
| **Table 8.1** | Competitive Positioning of TrustRAG (with Component 4) Against Existing RAG Verification Products | 86 |
| **Table 9.1** | Summary of Achievement of Research Objectives and Research Questions | 91 |
| **Table A.1** | Complete 50-Span Dual-Rater Blind Human Audit of the `no_evidence` Ground-Truth Label | 101 |
| **Table B.1** | Generative AI Use Disclosure Log | 104 |

---

# LIST OF ABBREVIATIONS

| Abbreviation | Description |
| :--- | :--- |
| **API** | Application Programming Interface |
| **APS** | Adaptive Prediction Sets |
| **AUROC** | Area Under the Receiver Operating Characteristic Curve |
| **BGE** | BAAI General Embedding (`BAAI/bge-small-en-v1.5`) |
| **BM25** | Best Matching 25 Probabilistic Retrieval Function |
| **CE** | Cross-Entropy |
| **CI** | Confidence Interval |
| **CPU** | Central Processing Unit |
| **CRC** | Conformal Risk Control |
| **D2T** | Data-to-Text Generation |
| **ECE** | Expected Calibration Error |
| **F1** | Harmonic Mean of Precision and Recall |
| **FN / FP** | False Negative / False Positive |
| **FR / NFR** | Functional Requirement / Non-Functional Requirement |
| **GPU** | Graphics Processing Unit |
| **HHEM** | Hughes Hallucination Evaluation Model |
| **IDF** | Inverse Document Frequency |
| **IoU** | Intersection over Union |
| **JSON** | JavaScript Object Notation |
| **LAC** | Least Ambiguous Set-Valued Classifiers |
| **LLM** | Large Language Model |
| **LOTO** | Leave-One-Task-Out |
| **MRR** | Mean Reciprocal Rank |
| **NDCG** | Normalised Discounted Cumulative Gain |
| **NLI** | Natural Language Inference |
| **OOD** | Out-of-Distribution |
| **QA** | Question Answering |
| **RAG** | Retrieval-Augmented Generation |
| **RAM / VRAM** | Random-Access Memory / Video Random-Access Memory |
| **REST** | Representational State Transfer |
| **SDG** | United Nations Sustainable Development Goal |
| **SLIIT** | Sri Lanka Institute of Information Technology |
| **SUS** | System Usability Scale |
| **TAF** | Topic Assessment Form |
| **TP / TN** | True Positive / True Negative |
| **UAT** | User Acceptance Testing |
| **UI** | User Interface |

---

# LIST OF APPENDICES

| Appendix | Description | Page |
| :--- | :--- | :---: |
| **Appendix - A** | Extended Experimental Results, Production Calibration Artifact (`c4_artifact.json`), and Complete 50-Span Dual-Rater Blind Audit Table | 100 |
| **Appendix - B** | Academic Integrity and Generative AI Use Disclosure Log | 104 |

---


# 1 INTRODUCTION

## 1.1 Overall project context

Large Language Models (LLMs) have become the foundational interface for enterprise knowledge management, clinical decision support, legal document analysis, and automated customer assistance. Despite dramatic improvements in pre-training scale and instruction tuning, standalone parametric language models remain fundamentally prone to *hallucination*—the generation of fluent, authoritative, and syntactically coherent statements that are factually incorrect or unsupported by authoritative sources [1], [2]. To mitigate parametric fabrication, modern production deployments overwhelmingly adopt **Retrieval-Augmented Generation (RAG)** [3]. In a standard RAG architecture, a user query first retrieves a set of relevant source passages from a verified document index; the language model is then prompted to synthesise its response strictly from the retrieved context.

While RAG substantially reduces open-ended fabrication compared to closed-book generation, empirical evidence proves that retrieval augmentation does not eliminate hallucinations. In their landmark corpus study introducing **RAGTruth**, Niu et al. [4] annotated $17,790$ LLM responses generated across Question Answering (QA), Data-to-Text (D2T), and News Summarization tasks, revealing that **$43.1\%$ of RAG responses still contain at least one hallucinated span**. Crucially, RAG hallucinations manifest in two structurally distinct failure modes that require opposite human interventions:
1. **Context Conflicts (`Evident Conflict` and `Subtle Conflict`):** The retrieved context contains the relevant fact, but the generator misreads, inverts, or numerically distorts it (for example, stating that an event occurred in *March 1925* when Sentence 1 of the retrieved passage explicitly states *March 1889*).
2. **Baseless Fabrications (`Evident Baseless Info` and `Subtle Baseless Info`):** The generator introduces extrinsic claims—drawn from parametric memory or speculative extrapolation—for which **no supporting or contradicting sentence exists anywhere in the retrieved context**.

When RAG systems are deployed in high-stakes domains such as healthcare, financial auditing, or legal compliance, an undetected hallucination carries severe operational and regulatory consequences. However, existing industrial evaluation frameworks—such as Ragas [5], TruLens [6], Vectara's Hughes Hallucination Evaluation Model (HHEM) [7], and prompt-based LLM-as-a-Judge pipelines [8]—suffer from four systemic deficiencies:
* **Response-Level Granularity:** Most evaluators return a single scalar faithfulness score (e.g., $0.64$) for an entire multi-paragraph answer without identifying *which specific words or clauses* are hallucinated.
* **Miscalibrated Probabilities:** Neural sequence classifiers and cross-encoders emit raw softmax logits that do not correspond to true empirical error frequencies, preventing risk-aware thresholding [9].
* **Absence of Finite-Sample Guarantees or Shift Detection:** When split conformal prediction [10], [11] is applied naively, its finite-sample coverage guarantee ($1 - \alpha$) holds only if test inputs remain exchangeable with the calibration distribution. Under domain shift (for instance, transferring from RAGTruth to RAGBench [12]), empirical coverage collapses silently without warning the operator.
* **Lack of Calibrated Evidence Grounding and No-Evidence Abstention:** Highlighting a suspicious span in the generated answer solves only half of the verification problem. To audit a flagged claim, a human reviewer must still read through the entire retrieved context—which averages **$19.05$ sentences per span** (and up to $89$ sentences) on RAGTruth—to locate the relevant source sentence or confirm that the claim is completely baseless.

To address these four interlocking limitations within a single unified software platform, Research Group **R26-DS-018** developed **TrustRAG: Span-Level Calibrated Hallucination Detection for Retrieval-Augmented Generation**. TrustRAG is architected as a four-stage, contract-driven verification pipeline governed by a single frozen Pydantic data schema (`src/common/schema.py`) and verified by $861$ automated pytest regression tests:

| Component | Owner & Student ID | Core Research & Engineering Responsibility | Primary Frozen Schema Fields (`src/common/schema.py`) |
| :--- | :--- | :--- | :--- |
| **Component 1 (C1)** | Wimukthi Gunarathna (`IT22060976`) | Fine-tuned `ModernBERT-base` token-level sequence detector with masked cross-entropy loss and character-offset span decoding ($0.7623$ example F1). | `Span.start`, `Span.end`, `Span.text`, `Span.token_probs`, `Span.span_score` |
| **Component 2 (C2)** | Sadini Wanniarachchi (`IT22592546`) | Post-hoc probability calibration (Platt scaling), split conformal tri-state classification (`FLAG`, `ABSTAIN`, `PASS`), and a one-sided run-time exchangeability gate (`DistributionCheck`). | `Span.calibrated_score`, `Span.conformal_decision`, `Span.prediction_set`, `AnalysisResult.guarantee_applies` |
| **Component 3 (C3)** | Ranya Sivakumar (`IT22607232`) | Calibration-consistent natural-language explanation generation with risk-monotonic linguistic hedging and bounded escalation on C2 abstentions. | `Span.error_type`, `Span.explanation`, `Span.escalated` |
| **Component 4 (C4)** *(This Thesis)* | E. M. S. N. Rathnasiri (`IT22047724`) | Retrieval-attribution alignment, split conformal evidence prediction sets ($70.63\%$ reading reduction at $90.03\%$ coverage), calibrated no-evidence abstention, and C1$\times$C2$\times$C4 tie-break fusion ($+5.96$ F1 pts). | `Span.evidence_sentence`, `Span.evidence_index`, `Span.entailment_score`, `AnalysisResult.attribution_checked` |

*Table 1.1: Ownership and Interface Boundaries of the Four TrustRAG Research Components.*

---

## 1.2 Individual component context

Within the TrustRAG architecture, **Component 4 (Retrieval-Attribution Alignment and Evidence Grounding)**—the subject of this individual thesis—operates at the crucial bridge between *answer-side span detection* (Components 1 and 2) and *source-side human verification*. Once Component 1 localises candidate spans $s_1, \dots, s_k$ within the generated answer and Component 2 assigns each span a calibrated hallucination probability and a conformal decision (`FLAG`, `ABSTAIN`, or `PASS`), a human auditor or downstream automated workflow faces an immediate evidentiary question:

> *Which specific sentence(s) in the retrieved source context $D = (c_1, c_2, \dots, c_m)$ settle this span—or is the span a baseless fabrication with no corresponding evidence anywhere in $D$?*

At first glance, matching a flagged span $s_i$ to a context sentence $c_j \in D$ might appear to be a routine application of semantic similarity or Natural Language Inference (NLI). However, a rigorous investigation reveals three fundamental scientific and engineering challenges that make retrieval-attribution alignment for hallucinated spans uniquely difficult:

1. **The Supervision & Reference Problem (Human Evidence Quotes Are Not Verbatim):**
   Benchmark datasets such as RAGTruth include human annotator notes containing an `Original:` field that quotes the source text against which a hallucinated span was judged. Prior studies have largely ignored this supervision or assumed that `Original:` strings can be matched to context sentences via simple substring search. In Layer 1 of this thesis (`src/c4_attribution/evidence_audit.py`), we audit all $1,214$ calibration spans and $1,533$ test spans on Component 1's frozen splits and discover that **only $50.68\%$ of attributable human evidence quotes appear verbatim in the source context**. Human annotators routinely trim clauses, normalise whitespace, paraphrase, or combine multi-line structured JSON entries. Evaluating an attribution model without first auditing and formalising the reference matching protocol means scoring a model against a moving target.

2. **The Forced-Citation Fallacy vs. Calibrated No-Evidence Abstention:**
   Standard attribution utilities—including the lexical nearest-neighbour baseline in Component 3 (`best_matching_sentence`) and dense bi-encoder retrieval—compute $\arg\max_{j} \text{score}(s_i, c_j)$ and unconditionally return the highest-scoring sentence $c_{j^*}$. When a span is an *Evident Conflict*, pointing to $c_{j^*}$ is helpful because $c_{j^*}$ directly contradicts the span. However, when a span is *Evident Baseless Info* or *Subtle Baseless Info*—which account for **$72.73\%$ of all annotated spans with `Original:` markers in QA**—no sentence in $D$ addresses the claim. Forcing a nearest-neighbour citation on a baseless span misleads the reviewer by highlighting an irrelevant sentence with high apparent confidence, yielding **$0.0000$ recall** on the no-evidence class by construction. Component 4 must therefore formulate a calibrated *no-evidence abstention gate* that returns `None` (`abstained = True`) when $\max_j \text{score}(s_i, c_j) < \tau$.

3. **Single-Sentence Point Citations vs. Conformal Evidence Prediction Sets:**
   When a context document contains $m = 19.05$ sentences on average (and up to $89$ sentences in long news summarization passages), returning only a single top-1 sentence $c_{j^*}$ provides no statistical assurance that the true gold sentence $j_{\text{gold}}$ was captured, especially when multiple sentences discuss related facets of an entity. Conversely, asking a human reviewer to read all $19.05$ sentences defeats the purpose of automated triage. By formulating evidence retrieval as a **split conformal prediction set** $\mathcal{C}_\alpha(s_i) \subseteq \{1, \dots, m\}$ via Least Ambiguous Set-Valued Classifiers (LAC) [13] and Conformal Risk Control (CRC) [14], Component 4 guarantees that the gold evidence sentence is contained in $\mathcal{C}_\alpha(s_i)$ with probability at least $1 - \alpha$ ($90.03\%$ empirical coverage at $\alpha = 0.10$) while shrinking the reviewer's reading load from $19.05$ sentences down to just **$4.61$ sentences per span**—a **$70.63\%$ reduction in human reading effort**.

4. **Resolving Component 2's Conformal Abstention Band via C1$\times$C2$\times$C4 Fusion:**
   To maintain $90\%$ span coverage on Component 1's predictions, Component 2's split conformal classifier must abstain (`ABSTAIN`) on $37.4\%$ of predicted spans—leaving a large band of uncertain spans that are neither flagged nor passed. Component 4 provides an orthogonal source-side signal: if Component 2 is uncertain (`ABSTAIN`) about a span, *and* Component 4 certifies that **no supporting evidence exists anywhere in the retrieved context**, promoting that span to a positive detection (`c4_tie_break`) recovers $65$ true-positive hallucinated responses and improves response-level F1 from $0.6960$ (`c1_c2_flag_only`) to **$0.7557$** (**$+5.96$ F1 points**).

---

## 1.3 Problem statement

The specific research and engineering problem addressed by this individual dissertation is formulated as follows:

> **Problem Statement:** Given a retrieved source document $D$ segmented into $m$ candidate sentences $(c_1, \dots, c_m)$ and a candidate answer span $s_i$ localised by a token-level hallucination detector, existing RAG evaluation pipelines lack a calibrated, low-latency attribution mechanism capable of: (i) aligning attributable conflict spans to their exact source evidence sentence under non-verbatim human supervision; (ii) constructing finite-sample conformal evidence sets $\mathcal{C}_\alpha(s_i)$ that bound reviewer reading effort while guaranteeing $\ge 1 - \alpha$ gold-sentence coverage; (iii) abstaining when a baseless span has no supporting evidence in $D$; and (iv) operating within a $<50\text{ ms}$ interactive CPU latency and $0\text{ MB}$ additional GPU memory envelope alongside an in-memory transformer detector.

Empirically, the severity of this problem is grounded in four concrete measurements established in our baseline audit of the RAGTruth corpus (`results/c4/analysis/evidence_audit.json` and `results/c4/tables.md`):
* **High Unassisted Reading Burden:** Across the $1,476$ attributable spans in the frozen calibration and test splits, the retrieved context pool contains a mean of **$18.92$ sentences** ($19.05$ on the test split, median $16.0$, 90th percentile $31.0$, maximum $89.0$). Without evidence set compression, verifying every flagged span requires reading nearly twenty sentences per span.
* **Non-Verbatim Supervision Noise:** Across all $1,476$ attributable spans with `Original:` annotations, only **$50.68\%$** ($0.5068$) appear verbatim in the segmented context (`qa`: $63.06\%$, `summarization`: $70.11\%$, `data2text`: $42.54\%$), rendering exact substring evaluation invalid for nearly half the benchmark.
* **High Prevalence of Baseless Claims in QA:** On the QA task, **$72.73\%$** ($0.7273$) of spans carrying an `Original:` note explicitly state that the information is absent from the retrieved passages. Any nearest-neighbour attributor without an abstention threshold errs on $100\%$ of these spans.
* **Pre-Filter Bottleneck in Two-Stage Neural Pipelines:** When a dense bi-encoder (`BAAI/bge-small-en-v1.5`) is used to pre-filter the context to the top $k=5$ sentences before running a cross-encoder (`DeBERTa-v3` or `HHEM`), the pre-filter alone discards the true gold sentence **$21.90\%$** of the time (`Recall@5` $= 0.7810$), imposing an irrecoverable $21.90\%$ error ceiling on every downstream neural scorer.

---

## 1.4 Research gap

A critical synthesis of recent literature on hallucination detection, citation attribution, and conformal prediction exposes four distinct research gaps that Component 4 directly resolves:

1. **Gap 1: Unaudited Supervision and Unmeasured Trivial Floors in Evidence Attribution.**
   While RAGTruth [4] and RAGBench [12] have catalysed span-level and response-level hallucination detection (e.g., LettuceDetect [15]), no prior study has systematically audited RAGTruth's `Original:` annotator quotes to separate *attributable quotes* from *no-evidence denials*, nor quantified the discrepancy between extraction conventions (first-line-only vs. until-next-marker, which disagree on $6.42\%$ of comments) and sentence splitters. Furthermore, prior attribution studies report pooled accuracy without comparing against task-stratified trivial floors (`first sentence`, `longest sentence`, `uniform random`, and `always abstain`). As we prove in Layer 3 (`Table C4-4`), on structured Data-to-Text tables, simply picking the **longest sentence** achieves **$0.5430$ Top-1 accuracy** because `json.dumps` serialisation concentrates most table fields into one long string—a critical confound that pooled metrics conceal.

2. **Gap 2: Unmeasured Pre-Filter Error Ceilings vs. Exact Lexical Containment.**
   Neural attribution pipelines routinely pair a bi-encoder pre-filter with an NLI cross-encoder [16], [17] under the assumption that dense embeddings capture semantic paraphrases better than lexical overlap. However, the pre-filter's own recall ceiling is rarely isolated from the cross-encoder's accuracy. In Layer 4b (`Table C4-6`), we demonstrate that dense top-5 pre-filtering suffers a $21.90\%$ error floor on RAGTruth, whereas parameter-free IDF-weighted content-token containment (`lexical`) achieves a $2.41\%$ error floor at $k=5$ (`Recall@5` $= 0.9759$) and an unconditional Top-1 accuracy of **$0.8952$**—outperforming the pre-filtered `HHEM` ($0.6959$) and `DeBERTa-v3` NLI ($0.6392$) pipelines while requiring $0\text{ MB}$ of memory.

3. **Gap 3: Absence of Split Conformal Prediction Sets for Context Evidence Grounding.**
   Split conformal prediction [10], [11], [13] and Conformal Risk Control [14] have been applied to classification, question answering, and response-level factuality [18], [19], and in Component 2 to binary span decisions. However, prior RAG attribution systems output either a single top-1 sentence or an uncalibrated top-$k$ list of fixed size $k$. Fixed top-$k$ lists cannot adapt to context difficulty: they under-cover ambiguous spans in long documents and over-include irrelevant sentences in short documents. Constructing adaptive conformal evidence sets $\mathcal{C}_\alpha(s_i)$ over variable-length sentence pools ($4$ to $89$ sentences) and translating set size directly into a **human reviewer reading-cost curve** (`Table C4-13`) is an unaddressed gap in the RAG verification literature.

4. **Gap 4: Forced Attribution on Baseless Hallucinations and Disconnected Pipeline Stages.**
   Existing attribution tools treat span detection and evidence retrieval as independent unidirectional steps: the detector flags a span, and the attributor cites the closest sentence. None of the existing frameworks (Ragas [5], TruLens [6], HHEM [7], LettuceDetect [15]) use a calibrated **no-evidence abstention gate** both to protect reviewers from false citations on baseless spans *and* to feed back into the detection layer as a tie-breaker when the primary conformal detector abstains (`ABSTAIN`).

---

## 1.5 Aim and objectives

### 1.5.1 Overall project aim and individual component aim
The overall aim of the **TrustRAG** project is to design, implement, and validate an end-to-end, statistically calibrated span-level hallucination detection, explanation, and evidence-attribution system for Retrieval-Augmented Generation with runtime distribution-shift safeguards.

Within this group aim, the **Individual Aim of Component 4** is:
> *To design, calibrate, and evaluate a contract-compliant retrieval-attribution alignment and evidence-grounding subsystem that maps flagged answer spans to minimal split-conformal context sentence sets with finite-sample coverage guarantees, abstains on baseless claims lacking source evidence, resolves uncertain detector abstentions via multi-component fusion, and executes in real time with zero additional model memory overhead.*

### 1.5.2 Measurable individual objectives
To achieve this aim, four measurable individual objectives are established:

* **Objective 1 (Data Supervision Audit, Matching Protocol, and Blind Human Validation):**
  To audit the evidence supervision (`Original:` annotator quotes) across Component 1's frozen RAGTruth calibration ($1,214$ spans) and test ($1,533$ spans) splits, quantify verbatim vs. content-token containment rates across tasks (`qa`, `data2text`, `summarization`) and error types, formalise a deterministic gold-sentence matching protocol ($\theta = 0.50$), and validate the no-evidence phrase heuristic via a 50-span dual-rater blind human audit achieving $\ge 0.90$ accuracy and Cohen's $\kappa > 0.80$.
* **Objective 2 (Multi-Backend Attribution Benchmarking and Pre-Filter Ceiling Isolation):**
  To implement and benchmark four attribution backends—IDF lexical containment (`lexical`), dense bi-encoder (`BAAI/bge-small-en-v1.5`), NLI cross-encoder (`MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`), and support cross-encoder (`vectara/hallucination_evaluation_model`)—under raw-span and decontextualised-sentence hypotheses against four task-stratified trivial floors (`first`, `longest`, `random`, `abstain`), while isolating the Recall@k error floor of candidate pre-filtering.
* **Objective 3 (Post-Hoc Score Calibration, Conformal Evidence Sets, and Reviewer Effort Reduction):**
  To evaluate post-hoc probability calibration (Temperature scaling, Platt scaling, Isotonic regression) on attribution scores and construct split conformal evidence prediction sets using Least Ambiguous Set-Valued Classifiers (LAC) and Conformal Risk Control (CRC) across miscoverage levels $\alpha \in \{0.01, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40\}$, achieving $\ge 90\%$ empirical coverage at $\alpha = 0.10$ while reducing the reviewer's per-span sentence reading burden by at least $65\%$.
* **Objective 4 (Calibrated No-Evidence Abstention, Cross-Task Transfer, C1$\times$C2$\times$C4 Fusion, and Live Serving):**
  To fit a calibration-split no-evidence abstention threshold ($\tau = 0.1973$), evaluate leave-one-task-out cross-task transferability, quantify the isolated example-F1 gain of promoting Component 2 `ABSTAIN` spans when Component 4 certifies no evidence (`c4_tie_break`), and ship a dependency-free JSON serving artefact (`c4_artifact.json`) achieving $<5\text{ ms}$ per-span CPU latency and $100\%$ numerical parity with offline evaluation.

---

## 1.6 Research questions or hypotheses

This thesis investigates four primary research questions (**RQ1–RQ4**) and tests their corresponding falsifiable hypotheses:

* **RQ1 (Supervision Structure & Pre-Filter Bottleneck):** *How faithfully do RAGTruth's human `Original:` annotations map to segmented context sentences across tasks, and how does candidate pre-filtering affect the comparative ceiling of lexical vs. neural cross-encoder attribution backends?*
  * **Hypothesis H1:** Fewer than $60\%$ of human `Original:` quotes appear verbatim in the segmented context (requiring content-token containment matching), and an IDF-weighted lexical containment scorer over the full sentence pool will outperform a top-5 dense-embedding pre-filtered cross-encoder pipeline because dense pre-filtering discards $>15\%$ of true gold evidence sentences before the cross-encoder runs.
* **RQ2 (Conformal Evidence Sets & Reviewer Reading Burden):** *Can split conformal prediction (LAC and CRC) over sentence attribution scores guarantee finite-sample empirical coverage $\ge 1 - \alpha$ across variable-length RAG contexts while substantially compressing the number of sentences a human reviewer must read?*
  * **Hypothesis H2:** At target miscoverage $\alpha = 0.10$ ($90\%$ target coverage), split conformal prediction will achieve empirical test coverage within the finite-sample $3\sigma$ calibration band ($[0.8640, 0.9338]$) and reduce the mean number of sentences read per span by more than $65\%$ relative to the full context pool ($19.05$ sentences), while revealing significant task-conditional set-size heterogeneity between QA, Data-to-Text, and Summarization.
* **RQ3 (No-Evidence Abstention & Task Base-Rate Confounds):** *To what extent can a calibration-fitted maximum-score threshold ($\tau$) distinguish baseless hallucinations (where no evidence exists in the context) from attributable spans, and when does selective abstention beat task-specific trivial floors?*
  * **Hypothesis H3:** A calibration-fitted threshold on maximum sentence attribution score will achieve $>0.80$ recall on no-evidence spans and outperform the `always_abstain` floor on pooled data, Data-to-Text, and Summarization, whereas on QA—where the no-evidence base rate exceeds $70\%$—the trivial `always_abstain` policy will remain a stringent baseline that must be reported transparently per task.
* **RQ4 (Multi-Component C1$\times$C2$\times$C4 Tie-Break Fusion):** *Does using Component 4's calibrated no-evidence signal to resolve Component 2's uncertain conformal abstentions (`ABSTAIN`) yield a statistically meaningful improvement in response-level hallucination detection F1?*
  * **Hypothesis H4:** Promoting a Component 2 `ABSTAIN` span to a positive detection if and only if Component 4 certifies `no_evidence = True` (`c4_tie_break`) will increase response-level F1 over `c1_c2_flag_only` by more than the $4.18$-point bootstrap noise width of Component 1.

| Research Question | Linked Objective | Experimental Layers in `src/c4_attribution/` | Primary Empirical Tables & Figures | Thesis Chapters |
| :--- | :--- | :--- | :--- | :--- |
| **RQ1** | Objective 1 & Objective 2 | Layers 1, 1b, 1c, 2, 3, 4, 4b | Tables `C4-1` to `C4-6`, `C4-15`; Figure 6.1 | Chapters 3, 4, 6.1, 6.2, 7.1 |
| **RQ2** | Objective 3 | Layers 5, 6, 10 | Tables `C4-7` to `C4-10`, `C4-13`; Figures 6.2, 6.3, 6.4, 6.7 | Chapters 3, 4, 6.1, 6.3, 7.1 |
| **RQ3** | Objective 1 & Objective 4 | Layers 1b, 7, 8, 9 | Tables `C4-1`, `C4-11`, `C4-12`, `C4-16`; Figures 6.5, 6.6 | Chapters 3, 5, 6.1, 6.3, 7.2 |
| **RQ4** | Objective 4 | Layers 11, 12, 13, 14 | Tables `C4-14`, `C4-15`, `C4-16`; Figures 4.1, 4.2, 4.3 | Chapters 4, 5, 6.2, 6.4, 7.3 |

*Table 1.2: Mapping of Research Questions, Individual Objectives, Evaluation Layers, and Thesis Chapters.*

---

## 1.7 Scope limitations and assumptions

### 1.7.1 Scope
The scope of this individual thesis encompasses:
1. **Dataset & Split Alignment:** All offline experiments are conducted on the official **RAGTruth** benchmark [4] across its three core tasks (`qa`, `data2text`, `summarization`) using the exact frozen calibration ($n = 297$ responses, $1,214$ gold spans) and test ($n = 2,670$ responses, $1,533$ gold spans; $1,509$ responses with C1 probability dumps for C1$\times$C2$\times$C4 fusion) splits established by Component 1 (`results/c1/modernbert-base/split_ids.json`).
2. **Granularity of Attribution:** Component 4 attributes each span to sentence units $c_j \in \text{context\_sentences}$ produced by the shared serving sentence splitter `(?<=[.!?])\s+`, populating `Span.evidence_sentence`, `Span.evidence_index`, and `Span.entailment_score` in `src/common/schema.py`.
3. **Offline vs. Online Architecture:** Offline, we evaluate four backends (`lexical`, `embed`, `hhem_support`, `nli_entail`), three post-hoc calibrators (`temperature`, `platt`, `isotonic`), and two conformal formalisms (`LAC`, `CRC`). Online, we ship the calibrated `lexical` attributor configured via `results/c4/c4_artifact.json`.

### 1.7.2 Limitations
1. **Inherited Naive Sentence Segmentation:** Because `Span.evidence_index` is an integer index into `AnalysisResult.context_sentences` shared across the FastAPI backend, Component 3, and Component 4, replacing the regex splitter `(?<=[.!?])\s+` would break contract parity across components. Instead of silently ignoring this constraint, Layer 1c (`segmentation_ablation`) explicitly measures what the naive splitter costs against a passage- and blank-line-aware variant splitter.
2. **Absence of Sentence-Level Gold Annotations on RAGBench:** While Component 2 evaluates response-level domain shift on RAGBench [12], RAGBench provides only response-level binary labels without `Original:` sentence quotes. Consequently, cross-domain transfer for Component 4 (Layer 9) is evaluated via Leave-One-Task-Out (LOTO) transfer across RAGTruth's three structurally distinct tasks (`qa`, `data2text`, `summarization`), and RAGBench's structural absence of span-level evidence supervision is documented explicitly in `results/c4/transfer/transfer_metrics.json`.
3. **Single-Sentence Gold Reference in Conformal Calibration:** When an annotator quote spans two adjacent context sentences, the matching protocol ($\theta = 0.50$) selects the single sentence with maximum content-token containment as the canonical reference index $j_{\text{gold}}$. Multi-hop joint sentence synthesis is identified as a future direction in Chapter 9.

### 1.7.3 Assumptions
1. **Within-Split Exchangeability:** Split conformal coverage guarantees ($1 - \alpha$) assume that test examples within an in-domain evaluation split are exchangeable with the calibration examples. When out-of-distribution inputs violate exchangeability at run time, Component 2's `DistributionCheck` sets `guarantee_applies = False`, and Component 4's Serving Rule 3 withdraws the coverage promise while continuing to render candidate evidence sentences for human inspection.
2. **Shared Hardware Envelope:** The live TrustRAG server is assumed to run on a standard developer workstation or laptop with a 6 GB GPU (already occupied by Component 1's `ModernBERT-base` detector) or CPU-only environment, motivating a zero-VRAM online attribution path.

---

## 1.8 Individual contribution

Every module, experiment, statistical analysis, JSON artefact, figure, and test in **Component 4** was independently conceived, implemented, and verified by the author (**E. M. S. N. Rathnasiri, `IT22047724`**). Specifically, my individual research and software engineering contributions are:

1. **Corpus-Level Supervision Audit, Extraction Convention Analysis, and Dual-Rater Blind Validation (Layers 1, 1b, 1c — `evidence_labels.py`, `evidence_audit.py`):**
   * Built the closed-vocabulary marker parser (`original`, `generative`, `generated`, `aigc`, `aogc`) that prevents false truncation on embedded colon tokens (`Sunday:`, `Review:`, `Weekends:`).
   * Quantified the $6.42\%$ disagreement between first-line-only extraction and until-next-marker extraction across $1,835$ comments with `Original:` markers.
   * Executed the segmentation ablation (`Layer 1c`) across $1,476$ attributable spans and completed the 50-span dual-rater blind human audit (`no_evidence_sample.json`), establishing **$94.00\%$ primary accuracy** ($47/50$), **$98.00\%$ inter-annotator agreement** ($49/50$), and **Cohen's $\kappa = 0.8466$**.
2. **Deterministic Content-Token Containment Matching Protocol and Task-Stratified Floor Suite (Layers 2 & 3 — `matching.py`, `lexical.py`):**
   * Formulated the $\theta = 0.50$ content-token containment reference matcher (fixed prior to model evaluation with a sensitivity sweep across $\theta \in \{0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90\}$).
   * Implemented the IDF-weighted lexical containment scorer and benchmarked it against four non-negotiable trivial floors (`first`, `longest`, `random`, `abstain`), discovering the `data2text` JSON-length confound where `floor_longest` reaches $0.5430$ Top-1 accuracy.
3. **Multi-Backend Neural Comparison and Pre-Filter Ceiling Discovery (Layers 4, 4b, 5 — `prefilter.py`, `calibration.py`):**
   * Evaluated four attribution backends (`lexical`, `embed`, `hhem_support`, `nli_entail`) across two hypothesis-formation strategies (`span` vs. `sentence`).
   * Isolated and measured the candidate pre-filter ceiling (`Table C4-6`), proving that dense bi-encoder top-5 pre-filtering drops $21.90\%$ of gold sentences compared to $2.41\%$ for IDF lexical containment, and evaluated Temperature, Platt, and Isotonic calibration on attribution scores (`Table C4-7`).
4. **Conformal Evidence Prediction Sets and Reviewer Reading-Cost Curve (Layers 6 & 10 — `evidence_conformal.py`, `review_cost.py`):**
   * Implemented split conformal prediction using both Least Ambiguous Set-Valued Classifiers (LAC) and Conformal Risk Control (CRC) over variable-size sentence pools, proving exact numerical agreement between LAC and CRC ($0.9003$ empirical coverage, $4.61$ mean set size at $\alpha = 0.10$) and quantifying the **$70.63\%$ reduction in human reading effort** (`Table C4-13`).
5. **Calibrated No-Evidence Abstention, LOTO Cross-Task Transfer, and C1$\times$C2$\times$C4 Tie-Break Fusion (Layers 7, 8, 9, 11 — `abstention.py`, `faithfulness.py`, `transfer.py`, `fusion.py`):**
   * Fitted the calibration-split abstention gate ($\tau = 0.1973$, achieving $0.8080$ recall and $0.4814$ F1 vs. $0.3522$ floor), ran Leave-One-Task-Out transfer across `qa`, `data2text`, and `summarization`, and designed the `c4_tie_break` fusion rule that adds **$+5.96$ example-F1 points** on the $1,509$-response test split.
6. **Production Serving Artefact, Contract Safety Guards, Diagrams, and 147 Automated Tests (Layers 12–15 — `serving.py`, `tables.py`, `figures.py`, `latency_benchmark.py`, `uat_validation.py`, `arch_diagrams.py`, `tests/test_c4_*.py`):**
   * Built `C4Attributor` and `results/c4/c4_artifact.json`, enforcing the three serving rules (`model_version` preservation, forbidden-token filter on per-span strings, and OOD guarantee withdrawal compatibility), achieving $1.312\text{ ms}$ p50 latency ($538.9\text{ spans/s}$), $16$ automated markdown tables (`C4-1` to `C4-16`), $10$ publication figures, and $147$ passing pytest unit and contract tests.

---

## 1.9 Thesis structure

The remainder of this dissertation is organised into eight chapters, followed by references, a glossary, and two appendices:
* **Chapter 2 (Literature Review and Related Work)** presents the systematic review methodology, theoretical foundations in transformer sequence tagging, post-hoc calibration, and split conformal prediction, a critical evaluation of related research and commercial tools, and a comparative gap synthesis matrix.
* **Chapter 3 (Research Methodology)** details the 12-layer experimental research design, the frozen RAGTruth dataset splits, mathematical formulations for IDF containment, calibration, LAC/CRC conformal sets, no-evidence abstention, and C1$\times$C2$\times$C4 fusion, alongside bootstrap significance testing, ethical compliance, and reproducibility artefacts.
* **Chapter 4 (Product and Solution Design)** specifies the functional and non-functional requirements, the four-stage TrustRAG system architecture (**Figure 4.1**), the internal Component 4 pipeline (**Figure 4.2**), design trade-offs, the frozen Pydantic data contract (**Figure 4.3**), core algorithms, and IEEE/ISO engineering standards.
* **Chapter 5 (Implementation, Integration and Testing)** documents the software stack, module-by-module implementation of `src/c4_attribution/`, REST API and React UI integration, the five-level software testing strategy (`147` C4 tests within `861` total tests), and local/Docker deployment.
* **Chapter 6 (Results and Validation)** reports the complete empirical findings across **Tables `C4-1` to `C4-16`** and **Figures 6.1 to 6.7**, covering supervision audits, backend ablations, pre-filter ceilings, score calibration, conformal evidence sets, no-evidence abstention, cross-task transfer, latency/memory benchmarks (`C4-15`), UAT and blind human validation (`C4-16`), and multi-component fusion (`C4-14`).
* **Chapter 7 (Discussion)** interprets the four major scientific findings of Component 4, contrasts our results with published literature, and examines threats to internal, external, and construct validity.
* **Chapter 8 (Commercialisation and Sustainability)** analyses the target market, competitive positioning, open-core business model, intellectual property, EU AI Act compliance, and alignment with UN Sustainable Development Goals (SDGs 3, 4, 9, and 16).
* **Chapter 9 (Conclusion and Future Directions)** synthesises the conclusions, maps each objective to its verified quantitative outcome, and outlines four concrete directions for future research.

---

# 2 LITERATURE REVIEW AND RELATED WORK

## 2.1 Review method and scope

To establish a rigorous theoretical and empirical foundation for Component 4, a structured literature review was conducted following Kitchenham and Charters' guidelines for systematic reviews in software engineering and machine learning [20]. The review focused on five intersecting research themes: (1) taxonomy and benchmarks of hallucinations in Retrieval-Augmented Generation; (2) span-level hallucination detection; (3) post-hoc citation and evidence attribution in grounded generation; (4) probability calibration and split conformal prediction for NLP; and (5) selective abstention and unanswerability in reading comprehension.

### 2.1.1 Databases, search strings, and selection criteria
Literature searches were executed across **ACL Anthology**, **IEEE Xplore**, **ACM Digital Library**, **arXiv (cs.CL, cs.AI, cs.LG, stat.ML)**, and **Google Scholar** covering publications from January 2017 to October 2026 (with foundational statistical references such as Platt [21] and Vovk et al. [10] included for mathematical completeness). Table 2.1 summarises the structured boolean search strings, target repositories, and inclusion/exclusion criteria.

| Research Theme | Boolean Search String | Primary Databases | Inclusion Criteria | Exclusion Criteria |
| :--- | :--- | :--- | :--- | :--- |
| **1. RAG Hallucination Benchmarks & Taxonomy** | `("Retrieval-Augmented Generation" OR "RAG") AND ("hallucination" OR "faithfulness") AND ("span-level" OR "benchmark" OR "RAGTruth" OR "RAGBench")` | ACL Anthology, arXiv, ACM DL | Peer-reviewed or high-impact benchmark papers defining intrinsic/extrinsic or conflict/baseless RAG hallucinations. | Purely parametric (closed-book) trivia QA studies without retrieved context. |
| **2. Evidence Attribution & Citation Grounding** | `("evidence attribution" OR "citation generation" OR "source grounding" OR "ALCE") AND ("large language model" OR "NLI" OR "entailment")` | ACL Anthology, EMNLP, TACL, IEEE Xplore | Studies aligning generated claims or spans to supporting/contradicting source sentences. | Gradient-based token saliency maps that do not index into retrieved documents. |
| **3. Conformal Prediction & Risk Control in NLP** | `("conformal prediction" OR "conformal risk control" OR "prediction sets" OR "LAC") AND ("natural language processing" OR "question answering" OR "hallucination")` | JMLR, NeurIPS, ICML, TACL, arXiv | Papers deriving finite-sample coverage bounds ($1 - \alpha$) or risk bounds for set-valued NLP predictors. | Heuristic top-$k$ retrieval studies lacking distribution-free coverage guarantees. |
| **4. Selective Abstention & Unanswerability** | `("selective prediction" OR "abstention" OR "no evidence" OR "unanswerable") AND ("reading comprehension" OR "fact verification" OR "FEVER" OR "SQuAD 2.0")` | ACL Anthology, AAAI, TACL | Works formulating `NOT ENOUGH INFO` or no-evidence abstention thresholds on retrieved passages. | Refusal training for safety/toxicity unrelated to factual context sufficiency. |

*Table 2.1: Systematic Literature Search Strings, Databases, and Inclusion/Exclusion Criteria.*

From an initial pool of $218$ candidate papers retrieved across the four queries, title and abstract screening removed $124$ duplicates and out-of-scope closed-book studies. Full-text assessment of the remaining $94$ articles yielded **$45$ core references** synthesised in this chapter and listed in the References section.

---

## 2.2 Domain and theoretical foundations

### 2.2.1 Retrieval-augmented generation and the anatomy of context failures
In the formal RAG framework introduced by Lewis et al. [3], given a user query $q$ and a corpus $\mathcal{D}$, a retriever selects a subset of passages $D \subset \mathcal{D}$, and a generator $p_\theta(y \mid q, D)$ produces a response $y$. While conditioning on $D$ anchors the generator in external text, Ji et al. [1] and Huang et al. [2] classify the resulting generation errors along two orthogonal axes:
* **Intrinsic Hallucinations (Context Conflicts):** The generated span $s_i \subset y$ directly contradicts facts stated in $D$. In RAGTruth [4], these are subdivided into *Evident Conflict* (direct negation, numerical distortion, or entity swap) and *Subtle Conflict* (nuanced semantic shift, modality distortion, or pragmatic misrepresentation).
* **Extrinsic Hallucinations (Baseless Information):** The generated span $s_i \subset y$ introduces claims that are neither supported nor contradicted by $D$. In RAGTruth [4], these are subdivided into *Evident Baseless Info* (fabricated clauses or unsupported specifics) and *Subtle Baseless Info* (plausible common-sense elaborations or world knowledge absent from $D$, including spans marked `implicit_true = True`).

This distinction has a profound theoretical consequence for **Component 4**: for an *Intrinsic Conflict*, a true evidence sentence $c_{j_{\text{gold}}} \in D$ exists (the sentence that the model contradicted), so attribution is a **sentence localisation problem**. For an *Extrinsic Baseless* span, $j_{\text{gold}} = \varnothing$ (no sentence in $D$ settles the claim), so attribution becomes a **selective abstention problem**. Any attribution architecture that conflates these two regimes—by forcing every flagged span to point to a context sentence—violates the underlying data-generating process.

### 2.2.2 Lexical vs. neural cross-encoder scoring for sentence alignment
Given a candidate span $s_i$ (or its enclosing response sentence $h(s_i)$) and a pool of context sentences $D = (c_1, \dots, c_m)$, sentence attribution requires a scoring function $f(s_i, c_j) \in \mathbb{R}$. Three primary families of scoring functions exist in the information retrieval and NLP literature:

1. **IDF-Weighted Lexical Content-Token Containment:**
   Classical lexical matching (Robertson and Zaragoza [22]) weights shared terms by their inverse document frequency. When adapted to sentence-level evidence grounding within a single context document $D$, let $T(x)$ denote the set of lowercased non-stopword alphanumeric content tokens of length $>1$ in string $x$. The within-document inverse document frequency of token $t \in T(c_1) \cup \dots \cup T(c_m)$ is:
   $$\text{idf}(t; D) = \ln\!\left(\frac{m + 1}{\text{df}(t; D) + 1}\right) + 1.0, \quad \text{where } \text{df}(t; D) = \sum_{j=1}^{m} \mathbb{I}\!\left[t \in T(c_j)\right]$$
   The IDF-weighted content-token overlap (containment) score of context sentence $c_j$ with respect to span $s_i$ is the fraction of the span's IDF mass carried by $c_j$:
   $$s_j = f_{\text{lex}}(s_i, c_j) = \frac{\sum_{t \in T(s_i) \cap T(c_j)} \text{idf}(t; D)}{\sum_{t \in T(s_i)} \text{idf}(t; D)} \in [0, 1]$$
   Crucially, because $\text{idf}(t; D)$ is computed strictly over the $m$ sentences of the current document $D$, zero corpus statistics leak across examples, and the score is bounded in $[0, 1]$ with zero learned parameters.

2. **Dense Bi-Encoder Cosine Similarity:**
   Dense passage retrievers such as DPR [23] and BGE (`BAAI/bge-small-en-v1.5`) [24] map the span and each context sentence independently into a $d$-dimensional unit hypersphere via a pretrained transformer encoder $E_\phi(\cdot)$ and compute cosine similarity:
   $$f_{\text{embed}}(s_i, c_j) = \frac{\langle E_\phi(s_i), E_\phi(c_j) \rangle}{\|E_\phi(s_i)\|_2 \, \|E_\phi(c_j)\|_2}$$
   While bi-encoders allow $O(m)$ independent encoding, compressing an entire sentence (or a long JSON record line in Data-to-Text) into a single $384$-dimensional vector dilutes fine-grained entity, numerical, and negation tokens.

3. **Natural Language Inference (NLI) and Factuality Cross-Encoders:**
   Cross-encoders such as `DeBERTa-v3` fine-tuned on MNLI/FEVER/ANLI [16], [25] and Vectara's `HHEM` [7] concatenate the premise $c_j$ and hypothesis $s_i$ as `[CLS] c_j [SEP] s_i [SEP]` and apply full self-attention across all token pairs. While cross-attention captures fine-grained syntactic interactions, its computational complexity is $O(m \cdot L^2)$ per span. In practice, systems pre-filter $D$ to the top $k \ll m$ sentences using a fast scorer before invoking the cross-encoder—introducing a hard recall ceiling whenever the pre-filter drops $c_{j_{\text{gold}}}$.

### 2.2.3 Post-hoc probability calibration mathematics
A scoring function $s_j \in [0, 1]$ is **calibrated** if, among all candidate pairs assigned score $p$, the empirical fraction that are true gold evidence sentences equals $p$: $\mathbb{P}(Y = 1 \mid \hat{P} = p) = p$ [9]. Neural logits and raw overlap ratios typically violate this property. Given a held-out calibration set $\mathcal{D}_{\text{cal}} = \{(z_r, y_r)\}_{r=1}^{N}$, three standard post-hoc calibration maps are defined:
* **Temperature Scaling [9]:** Scales logit $z_r = \text{logit}(s_r)$ by a single scalar $T > 0$: $\hat{p}_r = \sigma(z_r / T)$. Because $T > 0$ is a strictly monotonic rescaling around zero, it preserves ranking (AUROC) unchanged.
* **Platt Scaling [21]:** Fits a two-parameter logistic regression $\hat{p}_r = \sigma(a z_r + b)$ with slope $a \in \mathbb{R}$ and intercept $b \in \mathbb{R}$ by minimising Negative Log-Likelihood (NLL) on $\mathcal{D}_{\text{cal}}$.
* **Isotonic Regression [26]:** Fits a non-parametric, piecewise-constant monotonically non-decreasing step function $\hat{p}_r = g(s_r)$ via the Pool Adjacent Violators Algorithm (PAVA) minimising $\sum_r (y_r - g(s_r))^2$.

Calibration quality is quantified via **Expected Calibration Error (ECE)** across $B$ equal-width probability bins $I_1, \dots, I_B$:
$$\text{ECE} = \sum_{b=1}^{B} \frac{|I_b|}{N} \left| \text{acc}(I_b) - \text{conf}(I_b) \right|, \quad \text{MCE} = \max_{b \in \{1,\dots,B\}} \left| \text{acc}(I_b) - \text{conf}(I_b) \right|$$
alongside the **Brier score** $\frac{1}{N}\sum_{r=1}^N (\hat{p}_r - y_r)^2$. Crucially, as proved in Layer 5 of this thesis (`src/c4_attribution/calibration.py`), **every ECE and Brier score must be read against the constant base-rate predictor** ($\hat{p}_r \equiv \bar{y}_{\text{cal}}$), which ignores its input entirely yet achieves near-zero ECE by construction.

### 2.2.4 Split conformal prediction (LAC) and Conformal Risk Control (CRC)
To convert point attribution scores into a set of candidate context sentences $\mathcal{C}_\alpha(s_i) \subseteq \{1, \dots, m\}$ with a finite-sample coverage guarantee, we build on **Split Conformal Prediction** (Papadopoulos et al. [27]; Vovk et al. [10]; Angelopoulos and Bates [11]) and **Least Ambiguous Set-Valued Classifiers (LAC)** (Sadinle et al. [13]).

Let $\mathcal{D}_{\text{cal}} = \{(s_r, D_r, j_{\text{gold}}^{(r)})\}_{r=1}^{n}$ be $n$ exchangeable calibration spans where a gold evidence sentence $j_{\text{gold}}^{(r)} \in \{1, \dots, m_r\}$ exists. For each span $r$, raw sentence scores $(s_{r,1}, \dots, s_{r,m_r})$ are normalised into `[0, 1]` probabilities $p_{r,j}$ via max-normalisation $p_{r,j} = s_{r,j} / \max(1, \max_{k} s_{r,k})$ so that the relative support of each sentence is preserved. The **LAC nonconformity score** of the true gold sentence is:
$$R_r = 1 - p_{r, j_{\text{gold}}^{(r)}} \in [0, 1]$$
Given a user-specified miscoverage rate $\alpha \in (0, 1)$, the finite-sample corrected empirical quantile $\hat{q}_{1-\alpha}$ over $R_1, \dots, R_n$ is the $\lceil (n + 1)(1 - \alpha) \rceil$-th smallest value of $\{R_1, \dots, R_n\}$:
$$\hat{q}_{1-\alpha} = \text{Quantile}\!\left(\{R_1, \dots, R_n\};\, \frac{\lceil (n + 1)(1 - \alpha) \rceil}{n}\right)$$
At test time, for a new span $s_{\text{test}}$ with normalised sentence probabilities $(p_1, \dots, p_m)$, the conformal evidence set includes every sentence whose nonconformity score does not exceed $\hat{q}_{1-\alpha}$:
$$\mathcal{C}_\alpha(s_{\text{test}}) = \left\{\, j \in \{1, \dots, m\} \;\middle|\; 1 - p_j \le \hat{q}_{1-\alpha} \,\right\}$$
By the exchangeability of $(R_1, \dots, R_n, R_{\text{test}})$, Sadinle et al. [13] and Vovk et al. [10] prove that the marginal coverage satisfies the finite-sample lower bound:
$$\mathbb{P}\!\left(j_{\text{gold}}^{(\text{test})} \in \mathcal{C}_\alpha(s_{\text{test}})\right) \ge 1 - \alpha$$
Furthermore, **Conformal Risk Control (CRC)** (Angelopoulos et al. [14]) generalises split conformal prediction from binary indicator losses to any bounded, monotonically non-increasing loss function $\mathcal{L}(\lambda) \in [0, 1]$ by choosing:
$$\hat{\lambda}_{\text{CRC}} = \inf \left\{\, \lambda \in \Lambda \;\middle|\; \frac{n}{n + 1}\hat{R}_n(\lambda) + \frac{1}{n + 1} \le \alpha \,\right\}$$
When the loss function is the binary miss indicator $\mathcal{L}_r(\lambda) = \mathbb{I}[j_{\text{gold}}^{(r)} \notin \mathcal{C}_\lambda(s_r)]$, CRC and LAC are theoretically dual; Layer 6 (`Table C4-10`) tests whether they agree numerically on RAGTruth.

---

## 2.3 Related research

### 2.3.1 Corpus-scale RAG hallucination benchmarks: RAGTruth and RAGBench
The empirical study of RAG hallucinations was transformed by **RAGTruth** (Niu et al., ACL 2024 [4]), which remains the only large-scale benchmark providing human-annotated **character-offset hallucination spans** across three distinct generation tasks: open-domain Question Answering (MS MARCO passages), Data-to-Text generation (Yelp restaurant JSON records), and News Summarization (CNN/DailyMail articles). Crucially, RAGTruth annotators recorded free-text justification metadata (`meta`) for every span, containing `Original:` quotes and error-type tags (`Evident Conflict`, `Subtle Conflict`, `Evident Baseless Info`, `Subtle Baseless Info`, `implicit_true`, `due_to_null`). However, Niu et al. [4] used these annotations solely to train and evaluate response- and span-level detectors (e.g., fine-tuned Llama-2-13B); they never audited the `Original:` field as a supervision source for **evidence sentence attribution**.

Complementing RAGTruth, Friel et al. [12] introduced **RAGBench** (`100k` examples across five industry domains) and the **TRACe** evaluation framework, which measures *Utilization*, *Relevance*, *Adherence*, and *Completeness*. While TRACe separates retrieval sufficiency from generation adherence at the passage level, its public test splits provide response-level binary labels rather than character-offset spans paired with `Original:` sentence quotes. In TrustRAG, RAGBench serves as Component 2's out-of-distribution (OOD) shift benchmark (where conformal coverage drops from $0.8808$ in-domain to $0.7410$ OOD, validating the runtime exchangeability gate), while Component 4 evaluates cross-domain attribution transfer via Leave-One-Task-Out folds across RAGTruth's three structurally divergent task types (`Table C4-12`).

### 2.3.2 Span-level hallucination detection: LettuceDetect and ModernBERT
Prior to encoder-based token tagging, fine-grained hallucination detection relied on autoregressive LLM prompting or sentence-by-sentence NLI [17], [28], both of which incur multi-second latency per response. **LettuceDetect** (Kovács and Recski, 2025 [15]) reformulated RAG hallucination detection as token-level sequence classification over `[CLS] Context [SEP] Question [SEP] Answer [SEP]` using **ModernBERT** (Warner et al., 2024 [29]), which supports an $8,192$-token context window via Rotary Position Embeddings (RoPE) and alternating local/global attention. On RAGTruth, `LettuceDetect-large` achieves $0.7918$ example-level F1, and Component 1 of TrustRAG builds a calibrated `ModernBERT-base` detector achieving $0.7623$ example-level F1.

However, both LettuceDetect [15] and Component 1 are **answer-only taggers**: they output character offsets $[a_{\text{start}}, a_{\text{end}})$ inside the generated answer $y$, but they return **zero information about the retrieved context $D$**. As Kovács and Recski [15] explicitly note in their limitations, a token classifier flags *what* looks suspicious in the answer, leaving the human operator to manually search the source document to determine *why* it was flagged and whether contradicting evidence exists.

### 2.3.3 Post-hoc citation and evidence attribution (ALCE, AttrScore, RARR)
A parallel stream of literature investigates attributing LLM claims to source documents:
* **ALCE (Gao et al., EMNLP 2023 [30])** and **AttrScore (Yue et al., 2023 [31])** evaluate whether inline citations `[1][2]` generated by an LLM truly entail the statement they are attached to, using NLI cross-encoders (`TRUE` / `T5-XXL`). They find that up to $50\%$ of LLM-generated inline citations are post-hoc rationalisations that do not support the claim.
* **RARR (Gao et al., ACL 2023 [32])** retrieves external web evidence post-hoc to revise unsupported claims in closed-book LLM outputs.
* **AlignScore (Zha et al., ACL 2023 [33])** and **SummaC (Laban et al., TACL 2022 [17])** compute sentence-by-sentence alignment matrices between source documents and summaries using RoBERTa/DeBERTa NLI models.

Critically, these works address **attributing supported claims**, whereas Component 4 addresses the inverse and more nuanced problem of **attributing flagged hallucinated spans**. When a span is a *Conflict*, the goal is to retrieve the source sentence that the model distorted (so lexical/entity overlap remains high while NLI entailment is *low* or *contradictory*). This explains a central empirical finding of our thesis (`Table C4-5`): sorting context sentences by `nli_entail` ($P(\text{entailment})$) underperforms IDF lexical containment (`0.8176` vs. `0.8952` even when the gold sentence survives pre-filtering), because a contradicted source sentence does not *entail* the hallucinated span—it shares the span's content tokens while contradicting its predicate!

### 2.3.4 Conformal prediction and selective abstention in NLP
Conformal prediction has gained rapid traction in NLP for bounding uncertainty in classification (Maltoudoglou et al. [34]), extractive QA (Fisch et al. [35]), and LLM factuality filtering (Mohri and Hashimoto [18]; Cherian et al. [19]). Fisch et al. [35] construct conformal answer sets over candidate spans in reading comprehension and introduce an explicit abstention label when a question is unanswerable (as in SQuAD 2.0 [36] and FEVER's `NOT ENOUGH INFO` class [37]). However, no prior work has applied LAC [13] and CRC [14] to **context sentence evidence sets for span-level RAG hallucination audits**, nor measured how conformal set size translates into a concrete human reading-burden reduction curve across heterogeneous RAG tasks.

---

## 2.4 Existing systems products and techniques

To situate Component 4 against deployed industry tools, we evaluate four widely used RAG verification products and open-source frameworks:

1. **Vectara Hughes Hallucination Evaluation Model (`HHEM` / `HHEM-2.1`) [7]:**
   Vectara's open-weights cross-encoder (`vectara/hallucination_evaluation_model`, $184\text{M}$ parameters) scores `(context, summary)` pairs in `[0, 1]`. While lightweight compared to an LLM judge, HHEM is trained for **response-level** consistency. When adapted to score candidate context sentences in Layer 4 (`hhem_support`), it requires $440\text{ MB}$ of memory, takes $24.6\text{ ms/span}$ on CPU even when restricted to a $k=5$ pre-filter, and achieves $0.6959$ unconditional Top-1 accuracy (capped by the pre-filter's $0.7810$ recall). Moreover, HHEM provides no conformal prediction sets and no calibrated no-evidence abstention.
2. **Ragas (`faithfulness` and `context_utilization`) [5]:**
   Ragas decomposes an answer into atomic statements via an external LLM call (`GPT-4o`) and prompts the LLM a second time to judge whether each statement is inferred from the context. While popular for offline developer dashboards, Ragas requires $1,500\text{--}4,000\text{ ms}$ per response, incurs recurring commercial API costs, leaks private enterprise contexts to third-party APIs, and returns neither character-offset spans nor conformal sentence sets.
3. **TruLens (`Groundedness` Triad) [6]:**
   TruLens evaluates RAG pipelines along Context Relevance, Groundedness, and Answer Relevance. Its groundedness feedback function prompts an LLM or runs an NLI model to score sentences in `[0, 1]` and extracts a single supporting quote. However, TruLens does not calibrate its scores, never abstains with a calibrated no-evidence threshold, and provides no finite-sample coverage guarantee on retrieved evidence sentences.
4. **LettuceDetect Web/API Service [15]:**
   LettuceDetect provides fast token-level hallucination span detection via ModernBERT, matching the role of Component 1 in TrustRAG. However, LettuceDetect stops at highlighting the answer text: it does not segment the context, does not align spans to source sentences, cannot distinguish conflicts from baseless claims via context abstention, and provides no conformal coverage or distribution-shift guarantees.

---

## 2.5 Gap synthesis and conceptual framework

### 2.5.1 Comparative gap matrix
Table 2.2 synthesises the capabilities and limitations of representative research methods and commercial tools across seven evaluation dimensions directly relevant to Component 4.

| System / Study | Granularity | Evidence Sentence Attribution | Gold Supervision Audited? | Conformal Evidence Sets ($1-\alpha$) | Calibrated No-Evidence Abstention | Detector Tie-Break Fusion | Online Overhead (Latency / RAM) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Ragas** (Es et al. [5]) | Atomic claim (LLM) | Prompt-based (unindexed) | No | No | No | No | $>2,000\text{ ms}$ / API cloud |
| **TruLens** [6] | Answer sentence | Top-1 NLI / LLM | No | No | No | No | $>500\text{ ms}$ / $>700\text{ MB}$ |
| **Vectara HHEM** [7] | Response-level | Adapted Top-1 ($0.6959$) | No | No | No | No | $24.6\text{ ms}$ / $440\text{ MB}$ |
| **SummaC / AlignScore** [17], [33] | Sentence matrix | Top-1 NLI cross-encoder | No | No | No | No | $>30\text{ ms}$ / $>700\text{ MB}$ |
| **LettuceDetect** (Kovács & Recski [15]) | Answer span (`ModernBERT`) | **None** (answer-only) | N/A | No | No | No | Detector only |
| **C3 Lexical Baseline** (`grounding.py`) | Answer span | Nearest sentence (forced) | First-line only | No | **No** ($0.0000$ recall) | No | $1.1\text{ ms}$ / $0\text{ MB}$ |
| **TrustRAG Component 4 (Ours)** | **Answer span $\rightarrow$ Context sentence(s)** | **IDF Lexical + 4-Backend Ablation ($0.8952$ Top-1)** | **Yes (Layer 1 Audit + Blind $\kappa = 0.8466$)** | **Yes (LAC & CRC, $70.63\%$ reading saved)** | **Yes ($\tau = 0.1973$, $0.8080$ recall)** | **Yes (`c4_tie_break`, $+5.96$ F1 pts)** | **$1.31\text{ ms}$ p50 / $0\text{ MB}$ RAM** |

*Table 2.2: Comparative Gap Synthesis Matrix of Hallucination Detection and Attribution Systems.*

### 2.5.2 Conceptual framework of Component 4
Guided by the gaps identified in Table 2.2, the conceptual framework of Component 4 is structured around four methodological pillars that directly govern Chapters 3–6:
1. **Audit Before Modelling (Layers 1–2):** Never evaluate an attribution model against raw annotator strings without first quantifying verbatim rates ($50.68\%$), extraction convention disagreements ($6.42\%$), splitter sensitivity ($0.068\%$ index movement), and dual-rater human label validity ($\kappa = 0.8466$).
2. **Isolate Pipeline Bottlenecks Against Task-Specific Floors (Layers 3–4b):** Compare parameter-free IDF lexical containment against dense and cross-encoder neural backends on both unconditional accuracy and *conditional accuracy given pre-filter survival*, while printing four trivial floors (`first`, `longest`, `random`, `abstain`) per task.
3. **Set-Valued Conformal Grounding for Reviewer Ergonomics (Layers 5, 6, 10):** Replace uncalibrated point citations with LAC and CRC conformal prediction sets $\mathcal{C}_\alpha(s_i)$ calibrated on frozen splits, measuring the exact trade-off between empirical coverage ($90.03\%$) and reviewer reading load ($4.61$ vs. $19.05$ sentences per span).
4. **Closed-Loop Abstention and Multi-Component Tie-Breaking (Layers 7–9, 11–12):** Use a calibration-fitted abstention threshold ($\tau = 0.1973$) to certify when no evidence exists in the context, both to prevent misleading citations in the UI and to resolve Component 2's uncertain `ABSTAIN` spans (`+5.96` F1 points), shipped via a zero-drift JSON artefact (`c4_artifact.json`).

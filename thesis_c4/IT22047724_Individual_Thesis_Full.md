# DECLARATION OF THE CANDIDATE AND SUPERVISOR

“I declare that this is my own work and this dissertation does not incorporate without acknowledgement any material previously submitted for a Degree or Diploma in any other University or institute of higher learning and to the best of my knowledge and belief it does not contain any material previously published or written by another person except where the acknowledgement is made in the text.

Also, I hereby grant to Sri Lanka Institute of Information Technology, the non-exclusive right to reproduce and distribute my dissertation, in whole or in part in print, electronic or other medium. I retain the right to use this content in whole or part in future works (such as articles or books).”

**Academic Integrity and AI Disclosure:** I have disclosed all use of generative AI and other automated tools in Appendix B in accordance with the module policy. I remain responsible for the accuracy, originality, citations, code, data, analyses, and conclusions in this dissertation.

**Candidate Name:** Epitakumbure Mudunkothgedara Sakindu Nivein Rathnasiri (`IT22047724`)

**Signature:** ________________________________________     **Date:** ___________________

The supervisor/s should certify the dissertation with the following declaration:

**“The above candidate has carried out research for the bachelor’s degree Dissertation under my supervision.”**

| Supervisor Role & Name | Signature of the Supervisor | Date |
| :--- | :--- | :--- |
| **Supervisor:** Mr. Samadhi Rathnayaka | ________________________________________ | ______________ |
| **Co-Supervisor:** Dr. Lakmini Abeywardhana | ________________________________________ | ______________ |

---

# DEDICATION

This dissertation is dedicated to my parents and family for their endless sacrifices, patience, and encouragement throughout my undergraduate studies at the Sri Lanka Institute of Information Technology, and to the researchers and open-source developers working toward trustworthy, verifiable artificial intelligence.

---

# ACKNOWLEDGEMENTS

I would like to express my sincere gratitude to our primary supervisor, **Mr. Samadhi Rathnayaka**, and our co-supervisor, **Dr. Lakmini Abeywardhana**, of the Department of Information Technology, Faculty of Computing, Sri Lanka Institute of Information Technology (SLIIT). Their feedback on our experimental setup, statistical calibration methods, and baseline evaluations guided this project from the earliest stages.

I am also deeply grateful to my project teammates in Group **R26-DS-018** for a fantastic collaborative partnership on the **TrustRAG** platform: **Wimukthi Gunarathna** (`IT22060976`, Component 1 Lead), **Sadini Wanniarachchi** (`IT22592546`, Component 2 Lead), and **Ranya Sivakumar** (`IT22607232`, Component 3 Lead). Maintaining a unified Pydantic schema (`src/common/schema.py`) and a shared suite of regression tests across all four components was made possible through their dedication and team spirit.

Finally, I thank the academic and administrative staff of the Faculty of Computing at SLIIT, as well as my family and friends, for their constant support throughout my degree. This research was conducted using institutional and personal hardware resources without commercial sponsorship.

---

# ABSTRACT

Retrieval-Augmented Generation (RAG) grounds Large Language Models in external text to reduce factual errors, yet evaluations on the RAGTruth benchmark show that 43.1% of generated responses still contain unsupported or contradicted claims. While token-level detectors can highlight a suspicious span in a model's answer, human auditors still face a difficult task: searching through the retrieved context—averaging 19.05 sentences per span on RAGTruth—to check whether a specific sentence disproves the claim or if the information is missing entirely. Current evaluation tools output either uncalibrated document scores or nearest-neighbour citations that never abstain on baseless fabrications and provide no guarantees that the true supporting sentence is included. This dissertation presents **Component 4 (Retrieval-Attribution Alignment and Evidence Grounding)** of the TrustRAG platform. First, we audit RAGTruth's human evidence annotations across frozen splits (1,214 calibration and 1,533 test spans), finding that only 50.68% of human quotes match source sentences verbatim. We establish a content-token containment matching protocol alongside a dual-rater blind audit of 50 no-evidence cases (94.0% agreement, Cohen's $\kappa = 0.8466$). Second, we benchmark lexical, dense embedding (`bge-small-en-v1.5`), and cross-encoder models (`DeBERTa-v3`, `HHEM`), showing that an aggressive top-5 pre-filter drops the true evidence sentence 21.94% of the time, whereas full-pool scoring drops only 2.35%. Our lexical attributor runs in 1.312 ms median CPU latency with 0.0 MB extra memory. Third, we frame evidence retrieval as split-conformal prediction using Least Ambiguous Set-Valued Classifiers (LAC); at $\alpha = 0.10$, Component 4 achieves 90.03% empirical coverage while reducing reviewer reading load from 19.05 to 4.61 sentences per span—a **70.6% reduction in reading effort** that translated to a 69.47% review speedup in user testing. Finally, using our no-evidence signal to resolve uncertain detector abstentions yields an isolated **+5.96 point gain** in response-level F1.

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
| **Table 2.2** | Comparative Gap Synthesis Matrix of Hallucination Detection and Attribution Systems | 24 |
| **Table 3.1** | Frozen RAGTruth Dataset Splits and Evidence Annotation Distribution (Layer 1 Audit) | 30 |
| **Table 3.2** | Twelve-Layer Experimental Protocol and Associated Artefact Outputs in Component 4 | 33 |
| **Table 5.1** | Development Environment, Software Dependencies, and Pinning Strategy | 53 |
| **Table 5.2** | Layer-by-Layer Module Inventory of `src/c4_attribution/` | 55 |
| **Table 5.3** | Five-Level Software Verification and Regression Test Matrix (`147` C4 Tests / `861` Suite Tests) | 59 |
| **Table 6.1** | Evidence Supervision on Frozen Calibration and Test Splits (`C4-1`) | 63 |
| **Table 6.2** | Verbatim Rate, Content-Token Containment, and No-Evidence Share by Task and Error Type (`C4-2`) | 64 |
| **Table 6.3** | Attribution Matching Protocol and Containment Threshold $\theta$ Sweep on Test Split (`C4-3`) | 65 |
| **Table 6.4** | Lexical Attribution Baseline Against Four Trivial Floors on Test Split (`C4-4`) | 65 |
| **Table 6.5** | Attribution Backend Comparison and Hypothesis-Formation Ablations on Test Split (`C4-5`) | 66 |
| **Table 6.6** | Candidate Pre-Filter Recall@k and Irrecoverable Error Floor Across $k \in \{1, 3, 5, 10\}$ (`C4-6`) | 66 |
| **Table 6.7** | Post-Hoc Attribution Score Calibration (Temperature, Platt, Isotonic) vs. Constant Floor (`C4-7`) | 67 |
| **Table 6.8** | Split Conformal Evidence Prediction Sets and Reviewer Reading Reduction on Test Split (`C4-8`) | 68 |
| **Table 6.9** | Marginal vs. Group-Conditional Conformal Coverage and Set Size by Task at $\alpha = 0.10$ (`C4-9`) | 68 |
| **Table 6.10** | Comparison of Least Ambiguous Set-Valued Classifiers (LAC) and Conformal Risk Control (CRC, `C4-10`) | 69 |
| **Table 6.11** | Calibrated No-Evidence Abstention Performance on Test Split (`1,046` Spans, `C4-11`) | 70 |
| **Table 6.12** | Cross-Task (LOTO) and Cross-Dataset (`MedHallu`, `RAGBench`) Transfer Evaluation (`C4-12`) | 70 |
| **Table 6.13** | Reviewer Reading Burden and Effort Reduction Across $\alpha$ Levels (`C4-13`) | 71 |
| **Table 6.14** | Runtime Inference Latency, Span Throughput, and Memory Footprint Across Backends (`C4-15`) | 73 |
| **Table 6.15** | End-to-End User Acceptance Testing (UAT), Blind Audit ($\kappa = 0.8466$), and Reviewer Timing (`C4-16`) | 74 |
| **Table 6.16** | Multi-Component C1 $\times$ C2 $\times$ C4 Fusion and Disagreement Matrix on Test Split (`C4-14`) | 75 |
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

Large Language Models (LLMs) are widely used today for enterprise knowledge systems, clinical summarisation, legal analysis, and automated customer support. Despite significant improvements in training scale and instruction tuning, standalone models remain prone to *hallucinations*—generating confident, syntactically fluent statements that are factually incorrect or unsupported by reliable sources [1], [2]. To address this, real-world systems heavily rely on **Retrieval-Augmented Generation (RAG)** [3]. In a standard RAG pipeline, a user query retrieves relevant source passages from an indexed document store, and the language model synthesises its answer based on those passages.

While retrieval grounding helps reduce pure fabrication, it does not eliminate hallucinations. On the **RAGTruth** benchmark, Niu et al. [6] evaluated $17,790$ responses across Question Answering (QA), Data-to-Text (D2T), and Summarization, finding that **$43.1\%$ of RAG responses still contain at least one hallucinated span**. These errors fall into two distinct categories:
1. **Context Conflicts (`Evident Conflict` and `Subtle Conflict`):** The retrieved context contains the relevant fact, but the model misinterprets or distorts it (for example, stating an event happened in *1925* when the source passage states *1889*).
2. **Baseless Fabrications (`Evident Baseless Info` and `Subtle Baseless Info`):** The model introduces claims—either from parametric training memory or pure fabrication—for which **no evidence exists anywhere in the retrieved passage**.

In high-stakes fields like healthcare, finance, or law, undetected hallucinations create serious operational risks. Yet existing evaluation tools—such as Ragas [12], TruLens [30], Vectara HHEM [28], and general LLM judges—suffer from four main limitations:
* **Response-Level Granularity:** Most tools return a single overall faithfulness score (e.g., $0.64$) for a multi-paragraph response without showing *which specific words* are hallucinated.
* **Miscalibrated Scores:** Neural classifiers produce raw scores that do not reflect true empirical error rates, making threshold setting unreliable [25].
* **Lack of Shift Awareness:** Split conformal prediction [16] provides coverage guarantees only if test data remains exchangeable with calibration data. Under domain shift, coverage can drop without warning.
* **No Calibrated Grounding or Abstention:** Flagging a suspicious span in an answer only solves half the problem. A human auditor still has to read through the entire retrieved context—averaging **$19.05$ sentences per span** on RAGTruth—to find the source sentence or verify that the claim has no evidence.

To resolve these challenges in an integrated software platform, Research Group **R26-DS-018** developed **TrustRAG: Span-Level Calibrated Hallucination Detection for Retrieval-Augmented Generation**. TrustRAG is organised into four contract-driven components governed by a unified Pydantic schema (`src/common/schema.py`) and verified by $861$ automated tests:

| Component | Owner & Student ID | Core Research & Engineering Responsibility | Primary Frozen Schema Fields (`src/common/schema.py`) |
| :--- | :--- | :--- | :--- |
| **Component 1 (C1)** | Wimukthi Gunarathna (`IT22060976`) | Fine-tuned `ModernBERT-base` token-level sequence detector with character-offset span decoding ($0.7623$ example F1). | `Span.start`, `Span.end`, `Span.text`, `Span.token_probs`, `Span.span_score` |
| **Component 2 (C2)** | Sadini Wanniarachchi (`IT22592546`) | Post-hoc probability calibration (Platt scaling), split conformal tri-state classification (`FLAG`, `ABSTAIN`, `PASS`), and runtime exchangeability check. | `Span.calibrated_score`, `Span.conformal_decision`, `Span.prediction_set`, `AnalysisResult.guarantee_applies` |
| **Component 3 (C3)** | Ranya Sivakumar (`IT22607232`) | Calibration-consistent explanation generation with linguistic hedging and bounded escalation on C2 abstentions. | `Span.error_type`, `Span.explanation`, `Span.escalated` |
| **Component 4 (C4)** *(This Thesis)* | E. M. S. N. Rathnasiri (`IT22047724`) | Retrieval-attribution alignment, split conformal evidence prediction sets ($70.63\%$ reading reduction at $90.03\%$ coverage), calibrated no-evidence abstention, and C1$\times$C2$\times$C4 tie-break fusion ($+5.96$ F1 pts). | `Span.evidence_sentence`, `Span.evidence_index`, `Span.entailment_score`, `AnalysisResult.attribution_checked` |

*Table 1.1: Ownership and Interface Boundaries of the Four TrustRAG Research Components.*

---

## 1.2 Individual component context

Within TrustRAG, **Component 4 (Retrieval-Attribution Alignment and Evidence Grounding)** serves as the bridge between answer-side span detection (Components 1 and 2) and source-side human verification. Once a span is flagged in a response, a human reviewer faces a practical question:

> *Which sentence in the retrieved passage settles this claim—or is the claim completely unsupported by the source?*

Retrieving evidence for hallucinated spans presents several specific technical challenges:
1. **Human Evidence Quotes Are Rarely Verbatim:** In RAGTruth, human annotators provide an `Original:` field citing the source text. In Layer 1 of this project (`src/c4_attribution/evidence_audit.py`), we find that **only $50.68\%$ of attributable quotes appear verbatim in the context**. Annotators frequently shorten sentences, reformat whitespace, or combine multi-line JSON values. Evaluating an attribution model requires first establishing a proper containment matching protocol.
2. **Forced Citations vs. Abstention on Baseless Claims:** Standard retrieval tools return the highest-scoring sentence unconditionally. When an error is an *Evident Conflict*, this sentence shows the contradiction. But when a claim is *Baseless*—which makes up **$72.73\%$ of annotated spans in QA**—no supporting sentence exists. Forcing a citation misleads the reviewer and results in $0.0000$ recall on baseless errors. Component 4 introduces a calibrated abstention threshold ($\tau = 0.1973$) that explicitly flags when evidence is missing.
3. **Point Citations vs. Conformal Prediction Sets:** Across RAGTruth, context passages average $19.05$ sentences (reaching up to $89$). Returning a single top-1 sentence provides no guarantee that the true evidence sentence was captured. Conversely, reading all $19$ sentences takes too long. Using split-conformal prediction via Least Ambiguous Set-Valued Classifiers (LAC), Component 4 constructs a small evidence set $\mathcal{C}_\alpha(s)$ that guarantees $90.03\%$ coverage at $\alpha = 0.10$ while shrinking average reading burden to **$4.61$ sentences per span**—a **$70.63\%$ reduction in reading effort**.
4. **Resolving Detector Abstentions via Multi-Component Fusion:** To preserve coverage, Component 2 abstains on $37.4\%$ of spans. When Component 2 is uncertain (`ABSTAIN`), and Component 4 confirms that **no evidence exists anywhere in the source passage**, promoting that span (`c4_tie_break`) recovers $65$ true-positive responses and improves response F1 from $0.6960$ to **$0.7557$** ($+5.96$ points).

---

## 1.3 Problem statement

The core problem investigated in this dissertation is stated as follows:

> **Problem Statement:** Given a retrieved passage $D$ segmented into sentences $(c_1, \dots, c_m)$ and an answer span $s_i$ identified by a hallucination detector, current RAG evaluation systems lack a calibrated, low-latency attribution method that can: (i) map conflict spans to their exact source sentence under non-verbatim human supervision; (ii) construct finite-sample conformal evidence sets that bound reviewer reading effort with guaranteed coverage; (iii) abstain when a baseless span has no source evidence; and (iv) run within a $<50\text{ ms}$ CPU latency and $0\text{ MB}$ extra GPU memory footprint alongside an active transformer model.

This problem is grounded in four empirical measurements on RAGTruth:
* **High reading burden:** Contexts average **$18.92$ sentences** ($19.05$ on test), meaning unassisted review requires reading roughly twenty sentences per span.
* **Supervision noise:** Only **$50.68\%$** of human quotes match context text verbatim, making exact substring matching inadequate for evaluation.
* **High prevalence of baseless errors in QA:** On QA, **$72.73\%$** of annotated spans have no evidence in the source passages, causing unthresholded models to fail completely on abstention.
* **Pre-filter bottleneck in two-stage pipelines:** Dense bi-encoders (`bge-small`) used to select candidate sentences for cross-encoders drop the true gold sentence **$21.90\%$** of the time at $k=5$, setting an irrecoverable ceiling on downstream neural models.

---

## 1.4 Research gap

Reviewing recent literature highlights four main research gaps:
1. **Gap 1: Unaudited Supervision and Unmeasured Trivial Floors:** No prior work has audited RAGTruth's `Original:` annotations to separate attributable quotes from missing-evidence denials, nor reported results against task-specific trivial floors (`longest sentence`, `first sentence`, `random`). On Data-to-Text, always choosing the longest sentence achieves $0.5430$ Top-1 accuracy due to JSON formatting—a key factor that pooled numbers obscure.
2. **Gap 2: Unmeasured Pre-Filter Ceilings vs. Lexical Scoring:** Two-stage neural retrieval pipelines rarely measure how many gold sentences are lost during initial filtering. We show that dense top-5 pre-filtering loses $21.90\%$ of gold sentences, whereas full-pool IDF lexical containment achieves $0.8952$ Top-1 accuracy with zero GPU memory.
3. **Gap 3: Lack of Conformal Prediction Sets for Context Grounding:** Existing systems return either a single sentence or a fixed top-$k$ list that cannot adapt to passage length. Developing adaptive conformal evidence sets with distribution-free coverage bounds across variable passage lengths ($4$ to $89$ sentences) addresses an important practical gap.
4. **Gap 4: Forced Attribution and Disconnected Pipeline Stages:** Standard tools treat span detection and retrieval as one-way steps without an abstention mechanism, and do not use evidence availability to resolve detector uncertainty.

---

## 1.5 Aim and objectives

### 1.5.1 Project aim and component aim
The overall aim of the **TrustRAG** project is to develop and validate a statistically calibrated, span-level hallucination detection, explanation, and evidence-grounding platform for RAG with distribution-shift safeguards.

The individual aim of Component 4 is:
> *To design, calibrate, and evaluate an evidence-grounding module that aligns flagged answer spans with minimal conformal sentence sets under finite-sample guarantees, abstains on baseless claims lacking source evidence, resolves detector abstentions via multi-component fusion, and operates in real time with zero extra GPU memory overhead.*

### 1.5.2 Measurable objectives
1. **Objective 1 (Data Audit & Ground-Truth Protocol):** Audit `Original:` annotations across RAGTruth's frozen splits ($1,214$ calibration and $1,533$ test spans), establish a $\theta = 0.50$ containment matching protocol, and validate the no-evidence heuristic through a 50-span blind audit achieving $\ge 0.90$ agreement and Cohen's $\kappa > 0.80$.
2. **Objective 2 (Multi-Backend Benchmarking & Pre-Filter Isolation):** Implement and compare lexical, dense (`bge-small`), and cross-encoder backends (`DeBERTa-v3`, `HHEM`) against four trivial baselines, while isolating the pre-filter recall ceiling.
3. **Objective 3 (Score Calibration & Conformal Sets):** Evaluate post-hoc calibration methods (Temperature, Platt, Isotonic) and construct LAC/CRC conformal evidence sets across $\alpha \in [0.01, 0.40]$, achieving $\ge 90\%$ empirical coverage at $\alpha = 0.10$ while cutting reviewer reading burden by at least $65\%$.
4. **Objective 4 (Calibrated Abstention, Transfer, Fusion & Serving):** Calibrate a no-evidence abstention threshold ($\tau = 0.1973$), evaluate Leave-One-Task-Out transfer, quantify the F1 gain from C1$\times$C2$\times$C4 tie-breaking, and deploy a JSON configuration artifact (`c4_artifact.json`) delivering $<5\text{ ms}$ CPU latency.

---

## 1.6 Research questions or hypotheses

* **RQ1 (Supervision Structure & Pre-Filter Ceiling):** *How reliably do RAGTruth's human quotes map to context sentences, and how does candidate pre-filtering impact lexical vs. cross-encoder attribution?*
  * **Hypothesis H1:** Under $60\%$ of quotes appear verbatim in the context, and full-pool lexical containment will outperform dense pre-filtered cross-encoders because dense filtering drops $>15\%$ of gold sentences.
* **RQ2 (Conformal Sets & Reviewer Effort):** *Can split conformal prediction guarantee $\ge 1 - \alpha$ coverage across variable context lengths while significantly reducing reviewer reading load?*
  * **Hypothesis H2:** At $\alpha = 0.10$, conformal prediction will achieve coverage within the $3\sigma$ band ($[0.8640, 0.9338]$) and reduce mean reading burden by more than $65\%$ compared to full context.
* **RQ3 (No-Evidence Abstention & Task Baselines):** *How effectively does a calibrated threshold distinguish baseless errors from attributable spans across tasks?*
  * **Hypothesis H3:** A calibration-fitted threshold will reach $>0.80$ recall on baseless spans and beat the `always_abstain` floor pooled, on Data-to-Text, and on Summarization.
* **RQ4 (Multi-Component Tie-Break Fusion):** *Does using Component 4's no-evidence signal to resolve Component 2's uncertain abstentions produce a meaningful F1 improvement?*
  * **Hypothesis H4:** Promoting Component 2 `ABSTAIN` spans when Component 4 certifies `no_evidence = True` will increase response F1 by more than the $4.18$-point bootstrap noise margin of Component 1.

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
1. **Dataset & Splits:** Offline evaluations use the **RAGTruth** benchmark [6] across QA, Data-to-Text, and Summarization on the exact frozen calibration ($297$ responses, $1,214$ spans) and test splits ($2,670$ responses, $1,533$ spans; $1,509$ responses for fusion) established by Component 1.
2. **Granularity:** Attribution is performed at the sentence level using the shared regex splitter `(?<=[.!?])\s+`, populating `Span.evidence_sentence` and `Span.evidence_index`.
3. **Serving Mode:** Offline tests compare four backends; the live system deploys the calibrated lexical engine via `c4_artifact.json`.

### 1.7.2 Limitations
1. **Regex Sentence Splitter:** To maintain contract consistency across components, we keep the serving regex splitter. Layer 1c shows this causes zero index shifts compared to a passage-aware splitter.
2. **Lack of Sentence Gold Annotations on RAGBench:** RAGBench provides only response-level labels. Cross-task transfer is therefore evaluated via Leave-One-Task-Out folds across RAGTruth tasks.
3. **Single-Sentence Gold References:** For quotes spanning multiple adjacent sentences, the protocol selects the sentence with highest token containment.

### 1.7.3 Assumptions
1. **Within-Domain Exchangeability:** Conformal coverage bounds assume calibration and test examples are exchangeable. When inputs shift out-of-distribution, Component 2's exchangeability gate withdraws guarantees while still rendering candidate sentences.
2. **Target Hardware:** The serving environment is designed to operate on a standard developer laptop with 6 GB VRAM or CPU-only setups.

---

## 1.8 Individual contribution

All research, mathematical formulation, software implementation, testing, and documentation for **Component 4** were independently conducted by the candidate (**E. M. S. N. Rathnasiri, `IT22047724`**):
1. **Supervision Audit & Dual-Rater Blind Validation (Layers 1, 1b, 1c):** Built marker parsing tools, measured convention disagreements ($6.42\%$), and conducted the 50-span blind audit achieving $94.0\%$ accuracy and Cohen's $\kappa = 0.8466$.
2. **Matching Protocol & Trivial Baselines (Layers 2 & 3):** Designed the $\theta = 0.50$ token containment matcher and benchmarked IDF scoring against four non-trivial baselines, revealing the Data-to-Text length confound.
3. **Backend Comparisons & Pre-Filter Ceiling (Layers 4, 4b, 5):** Evaluated four backends, isolated the $21.90\%$ pre-filter error ceiling in dense bi-encoders, and tested Temperature, Platt, and Isotonic calibration.
4. **Conformal Evidence Sets & Reading Burden Curve (Layers 6 & 10):** Implemented LAC and CRC conformal prediction sets, verified finite-sample coverage ($90.03\%$ at $\alpha = 0.10$), and quantified the $70.63\%$ reduction in reviewer reading load.
5. **Calibrated Abstention, Transfer & Tie-Break Fusion (Layers 7, 8, 9, 11):** Tuned the abstention threshold ($\tau = 0.1973$, $80.80\%$ recall), performed LOTO transfer, and designed the tie-break fusion rule that adds $+5.96$ F1 points.
6. **Serving Artifact, Safety Rules & Test Suite (Layers 12–15):** Created `C4Attributor` and `c4_artifact.json` with $1.312\text{ ms}$ p50 latency, enforced responsible AI rules, and authored $147$ passing unit and contract tests.

---

## 1.9 Thesis structure

The remainder of this dissertation is organised as follows:
* **Chapter 2 (Literature Review and Related Work)** reviews related literature on hallucination detection, citation attribution, calibration, and conformal prediction.
* **Chapter 3 (Research Methodology)** describes the twelve-layer experimental protocol, datasets, mathematical formulations, and evaluation metrics.
* **Chapter 4 (Product and Solution Design)** details system requirements, architectural diagrams, data contracts, and engineering standards.
* **Chapter 5 (Implementation, Integration and Testing)** documents the software modules, REST and React integration, and the five-tier testing strategy.
* **Chapter 6 (Results and Validation)** presents empirical findings across Tables `C4-1` to `C4-16` and Figures 6.1 to 6.7.
* **Chapter 7 (Discussion)** interprets key findings, contrasts results with existing tools, and evaluates limitations.
* **Chapter 8 (Commercialisation and Sustainability)** outlines target users, business models, scalability, and SDG mapping.
* **Chapter 9 (Conclusion and Future Directions)** summarises outcomes against research objectives and presents four future research avenues.

---

# 2 LITERATURE REVIEW AND RELATED WORK

## 2.1 Review method and scope

To provide a sound theoretical foundation, we conducted a structured literature review following Kitchenham and Charters' guidelines for software engineering and machine learning [20]. We focused on five main topics: (1) RAG hallucination taxonomy and benchmarks; (2) token-level hallucination detection; (3) post-hoc citation and evidence attribution; (4) probability calibration and conformal prediction in NLP; and (5) selective abstention and unanswerability in reading comprehension.

### 2.1.1 Databases, search strings, and selection criteria
Searches were conducted across **ACL Anthology**, **IEEE Xplore**, **ACM Digital Library**, **arXiv**, and **Google Scholar** covering papers from 2017 to 2026. Table 2.1 summarises the search queries and selection criteria.

| Research Theme | Boolean Search String | Primary Databases | Inclusion Criteria | Exclusion Criteria |
| :--- | :--- | :--- | :--- | :--- |
| **1. RAG Hallucination Benchmarks & Taxonomy** | `("Retrieval-Augmented Generation" OR "RAG") AND ("hallucination" OR "faithfulness") AND ("span-level" OR "benchmark" OR "RAGTruth" OR "RAGBench")` | ACL Anthology, arXiv, ACM DL | Peer-reviewed or high-impact benchmark papers defining intrinsic/extrinsic or conflict/baseless RAG hallucinations. | Purely parametric (closed-book) trivia QA studies without retrieved context. |
| **2. Evidence Attribution & Citation Grounding** | `("evidence attribution" OR "citation generation" OR "source grounding" OR "ALCE") AND ("large language model" OR "NLI" OR "entailment")` | ACL Anthology, EMNLP, TACL, IEEE Xplore | Studies aligning generated claims or spans to supporting/contradicting source sentences. | Gradient-based token saliency maps that do not index into retrieved documents. |
| **3. Conformal Prediction & Risk Control in NLP** | `("conformal prediction" OR "conformal risk control" OR "prediction sets" OR "LAC") AND ("natural language processing" OR "question answering" OR "hallucination")` | JMLR, NeurIPS, ICML, TACL, arXiv | Papers deriving finite-sample coverage bounds ($1 - \alpha$) or risk bounds for set-valued NLP predictors. | Heuristic top-$k$ retrieval studies lacking distribution-free coverage guarantees. |
| **4. Selective Abstention & Unanswerability** | `("selective prediction" OR "abstention" OR "no evidence" OR "unanswerable") AND ("reading comprehension" OR "fact verification" OR "FEVER" OR "SQuAD 2.0")` | ACL Anthology, AAAI, TACL | Works formulating `NOT ENOUGH INFO` or no-evidence abstention thresholds on retrieved passages. | Refusal training for safety/toxicity unrelated to factual context sufficiency. |

*Table 2.1: Systematic Literature Search Strings, Databases, and Inclusion/Exclusion Criteria.*

From an initial set of $218$ papers, title and abstract screening filtered out $124$ duplicate or out-of-scope studies. Detailed review of the remaining $94$ publications yielded **$45$ core references** synthesised in this chapter.

---

## 2.2 Domain and theoretical foundations

### 2.2.1 Retrieval-augmented generation and context failure types
In standard RAG [1], a retriever selects passages $D \subset \mathcal{D}$ for a query $q$, and a generator produces answer $y$. Generation errors typically fall into two categories [3], [5]:
* **Intrinsic Hallucinations (Context Conflicts):** The generated claim contradicts facts in $D$. In RAGTruth [6], these are split into *Evident Conflict* (direct contradiction or numerical error) and *Subtle Conflict* (slight nuance shifts).
* **Extrinsic Hallucinations (Baseless Information):** The claim introduces facts absent from $D$. In RAGTruth [6], these include *Evident Baseless Info* (completely fabricated details) and *Subtle Baseless Info* (plausible elaborations not found in $D$).

For *Conflicts*, a source sentence exists in $D$, making attribution a **localisation task**. For *Baseless* claims, no supporting sentence exists, making attribution a **selective abstention task**.

### 2.2.2 Lexical vs. neural cross-encoder scoring
Given an answer span $s_i$ and context sentences $D = (c_1, \dots, c_m)$, attribution requires a scoring function $f(s_i, c_j)$:
1. **IDF-Weighted Lexical Overlap:** Content tokens $T(x)$ are weighted by within-document inverse document frequency:
   $$\text{idf}(t; D) = \ln\!\left(\frac{m + 1}{\text{df}(t; D) + 1}\right) + 1.0, \quad s_j = \frac{\sum_{t \in T(s_i) \cap T(c_j)} \text{idf}(t; D)}{\sum_{t \in T(s_i)} \text{idf}(t; D)} \in [0, 1]$$
   This bounded score has zero learned parameters and avoids cross-document leakage.
2. **Dense Bi-Encoders:** Models like BGE [23] encode spans and sentences into vector embeddings and compute cosine similarity. However, sentence-level pooling often dilutes specific dates, numbers, and proper nouns.
3. **Cross-Encoders:** Models like DeBERTa-v3 [24] and HHEM [28] concatenate sentence and span pairs with full cross-attention. While expressive, scoring all candidates takes $O(m \cdot L^2)$ time, requiring pre-filtering that can discard gold sentences.

### 2.2.3 Post-hoc probability calibration
Raw model scores rarely match true empirical frequencies [25]. On a held-out calibration set, three standard calibration maps are commonly used:
* **Temperature Scaling [25]:** Scales logits by a scalar $T > 0$.
* **Platt Scaling [26]:** Fits logistic regression $\hat{p} = \sigma(a z + b)$ with slope and intercept.
* **Isotonic Regression [27]:** Fits a non-parametric monotonic step function via PAVA.

Calibration error is evaluated using **Expected Calibration Error (ECE)** and **Brier score**, interpreted alongside a constant base-rate baseline.

### 2.2.4 Split conformal prediction (LAC) and Conformal Risk Control (CRC)
To produce an evidence set $\mathcal{C}_\alpha(s) \subseteq \{1, \dots, m\}$ with guaranteed coverage, we use split conformal prediction [15], [16] via Least Ambiguous Set-Valued Classifiers (LAC) [17]. On calibration examples, the nonconformity score of the gold sentence is $R_r = 1 - p_{r, j_{\text{gold}}}$. For a chosen error rate $\alpha$, the empirical quantile $\hat{q}_{1-\alpha}$ is computed with a finite-sample correction. At test time, the conformal set includes all sentences meeting the cutoff:
$$\mathcal{C}_\alpha(s_{\text{test}}) = \left\{\, j \in \{1, \dots, m\} \;\middle|\; 1 - p_j \le \hat{q}_{1-\alpha} \,\right\}$$
Under exchangeability, this guarantees that $\mathbb{P}(j_{\text{gold}} \in \mathcal{C}_\alpha) \ge 1 - \alpha$.

---

## 2.3 Related research

### 2.3.1 RAG benchmarks: RAGTruth and RAGBench
**RAGTruth** [6] provides human-annotated character-offset spans across QA, Data-to-Text, and Summarization with `Original:` source notes. Prior work used these notes only for overall detection, leaving them unaudited as evidence references. **RAGBench** [29] offers response-level adherence metrics, used in TrustRAG as an out-of-distribution benchmark.

### 2.3.2 Span-level detection: LettuceDetect and ModernBERT
Recent detectors like **LettuceDetect** [10] frame hallucination detection as token classification using ModernBERT [11]. While fast, sequence taggers only highlight words in the response—they do not inspect or cite the context passage.

### 2.3.3 Post-hoc citation and evidence attribution
Frameworks like ALCE [19], AttrScore, and SummaC [14] evaluate citations for supported claims. Component 4 addresses the reverse task: finding evidence for *flagged hallucinated spans*, where contradicted sentences share content words with the span while disagreeing with its predicate.

### 2.3.4 Conformal prediction in NLP
Conformal methods have been applied to QA answer sets [32] and response factuality [33]. Component 4 applies conformal prediction to context sentence sets, measuring reading-effort reduction across RAG tasks.

---

## 2.4 Existing systems, products and techniques

We compare Component 4 with four prevalent systems:
1. **Vectara HHEM [28]:** A $184\text{M}$-parameter cross-encoder for response-level evaluation. When adapted for sentences, it takes $24.6\text{ ms/span}$ and needs $440\text{ MB}$ of memory without offering conformal sets.
2. **Ragas [12]:** Uses external LLM calls (GPT-4o) to evaluate claims. It incurs high latency ($>1,500\text{ ms}$), API costs, and privacy risks without providing sentence offsets.
3. **TruLens [30]:** Prompts LLMs or runs NLI models for chunk-level groundedness, lacking calibrated abstention and finite-sample guarantees.
4. **LettuceDetect [10]:** Provides fast token-level detection via ModernBERT, but does not segment context or provide evidence attribution.

---

## 2.5 Gap synthesis and conceptual framework

### 2.5.1 Comparative gap matrix
Table 2.2 summarises key capabilities across existing tools and Component 4.

| System / Study | Granularity | Evidence Sentence Attribution | Gold Supervision Audited? | Conformal Evidence Sets ($1-\alpha$) | Calibrated No-Evidence Abstention | Detector Tie-Break Fusion | Online Overhead (Latency / RAM) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Ragas** (Es et al. [12]) | Atomic claim (LLM) | Prompt-based (unindexed) | No | No | No | No | $>2,000\text{ ms}$ / API cloud |
| **TruLens** [30] | Answer sentence | Top-1 NLI / LLM | No | No | No | No | $>500\text{ ms}$ / $>700\text{ MB}$ |
| **Vectara HHEM** [28] | Response-level | Adapted Top-1 ($0.6959$) | No | No | No | No | $24.6\text{ ms}$ / $440\text{ MB}$ |
| **SummaC / AlignScore** [14], [13] | Sentence matrix | Top-1 NLI cross-encoder | No | No | No | No | $>30\text{ ms}$ / $>700\text{ MB}$ |
| **LettuceDetect** [10] | Answer span (`ModernBERT`) | **None** (answer-only) | N/A | No | No | No | Detector only |
| **C3 Lexical Baseline** (`grounding.py`) | Answer span | Nearest sentence (forced) | First-line only | No | **No** ($0.0000$ recall) | No | $1.1\text{ ms}$ / $0\text{ MB}$ |
| **TrustRAG Component 4 (Ours)** | **Answer span $\rightarrow$ Context sentence(s)** | **IDF Lexical + 4-Backend Ablation ($0.8952$ Top-1)** | **Yes (Layer 1 Audit + Blind $\kappa = 0.8466$)** | **Yes (LAC & CRC, $70.63\%$ reading saved)** | **Yes ($\tau = 0.1973$, $0.8080$ recall)** | **Yes (`c4_tie_break`, $+5.96$ F1 pts)** | **$1.31\text{ ms}$ p50 / $0\text{ MB}$ RAM** |

*Table 2.2: Comparative Gap Synthesis Matrix of Hallucination Detection and Attribution Systems.*

### 2.5.2 Conceptual framework
Component 4 is structured around four pillars:
1. **Audit before modelling:** Verify ground-truth annotations and splitter invariance before evaluating models.
2. **Isolate pipeline bottlenecks:** Compare lexical and neural backends against task floors, measuring pre-filter loss directly.
3. **Set-valued conformal grounding:** Replace point citations with conformal sets to give clear statistical coverage bounds.
4. **Closed-loop abstention and fusion:** Use calibrated abstention to handle baseless errors and resolve detector uncertainty.

---

# 3 RESEARCH METHODOLOGY

## 3.1 Research design

We follow an empirical, design-science research methodology structured into a twelve-layer experimental protocol (`Layers 1` through `12`, with runtime benchmarking, UAT, and system checks in `Layers 13–15`). In applied machine learning, evaluating models on unverified labels or tuning thresholds directly on test data introduces significant selection bias. To avoid these issues, our research design enforces five key methodological principles:

1. **Audit supervision before modelling (Layer 1 & 1b):** Before training or scoring any model, we audit the human supervision source (`Original:` annotations in RAGTruth) on Component 1's frozen splits (`src/c4_attribution/evidence_audit.py`). To ensure our regex heuristic for identifying no-evidence claims (`src/c4_attribution/evidence_labels.py`) is reliable, we conducted an independent dual-rater blind audit on $50$ sampled snippets (`seed = 42`). Rater 1 agreed on $47/50$ ($94.0\%$) and Rater 2 on $46/50$ ($92.0\%$), yielding raw agreement of $98.0\%$ and **Cohen's $\kappa = 0.8466$**.
2. **Pre-fixed matching cutoff (Layer 2):** Because fewer than $51\%$ of human quotes match context text verbatim, we define the gold sentence as the candidate maximising content-token containment above a pre-set threshold $\theta = 0.50$ (`src/c4_attribution/matching.py`). We locked this threshold before running any attribution models and report sensitivity sweeps across $\theta \in [0.30, 0.90]$ under both single-sentence (Convention A) and multi-sentence (Convention B) definitions (`Table C4-3`).
3. **Fit on calibration, test once on held-out split (Layers 4–11):** All parameter selection—including the choice of backend (`lexical|sentence`), calibrators, conformal quantiles $\hat{q}_{1-\alpha}$, and the abstention cutoff $\tau = 0.1973$—is performed **exclusively on the calibration split (`calib`)** and evaluated once on the test split (`test`). Unit tests confirm that tuning functions never access test records.
4. **Report against task-specific baseline floors (Layers 3, 5, 7, 11):** Every primary table includes simple baseline floors broken down by task (`qa`, `data2text`, `summarization`):
   * *Attribution baselines:* `first sentence`, `longest sentence`, `random`, and `abstain`.
   * *Calibration floor:* Constant base-rate predictor ($\hat{p} \equiv \bar{y}_{\text{cal}}$).
   * *Abstention baselines:* `always_abstain`, `never_abstain`, and unthresholded nearest-neighbour.
   * *Fusion baselines:* `trivial_always_positive`, `HHEM`, and `C1 alone`, evaluated against bootstrap confidence intervals.
5. **Exact parity between offline and online paths (Layer 12):** The production attributor (`C4Attributor` in `src/c4_attribution/serving.py`) uses the same IDF calculation as offline evaluation and loads calibrated values directly from `c4_artifact.json`, verified by regression tests to match within $10^{-9}$.

---

## 3.2 Data participants or study materials

### 3.2.1 Frozen RAGTruth benchmark splits and provenance
All experiments use the official **RAGTruth** corpus [6] stored locally under `data/raw/response.jsonl` ($17,790$ responses) and `data/raw/source_info.jsonl`. To maintain strict cross-component consistency, Component 4 loads split assignments from `results/c1/modernbert-base/split_ids.json` via `src/c4_attribution/dataset.py` and builds contexts using Component 1's shared text construction function.

Experiments operate over two distinct data sets:
1. **Gold-span attribution & abstention set (`calib` + `test`):** Used in Layers 1 to 10 to evaluate supervision, reference matching, backend accuracy, score calibration, conformal sets, abstention, and transfer. There are **$2,747$ human-annotated spans** in total ($1,214$ in `calib`, $1,533$ in `test`).
2. **Predicted-span pipeline fusion set ($297$ calib / $1,509$ test responses):** Used in Layer 11 (`src/c4_attribution/fusion.py`) to test end-to-end integration on the $2,390$ spans predicted by Component 1.

Table 3.1 summarises the supervision distribution measured in Layer 1 (`results/c4/analysis/evidence_audit.json`).

| Slice / Stratum | Annotated Gold Spans ($n$) | Spans with `Original:` Marker | Share with `Original:` | Attributable Spans (Quote Exists) | No-Evidence Spans (Denial) | No-Evidence Share of `Original:` | Candidate Context Pool Mean / Median / Max (Sentences) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Calibration Split (`calib`)** | $1,214$ | $787$ | $0.6483$ | $652$ | $134$ | $0.1703$ | $18.75$ / $16.0$ / $81.0$ |
| **Test Split (`test`)** | $1,533$ | $1,048$ | $0.6836$ | $824$ | $224$ | $0.2137$ | $19.05$ / $16.0$ / $89.0$ |
| **Combined (`calib` + `test`)** | **$2,747$** | **$1,835$** | **$0.6680$** | **$1,476$** | **$358$** | **$0.1951$** | **$18.92$ / $16.0$ / $89.0$** |
| — *Task: `qa`* | $696$ | $407$ | $0.5848$ | $111$ | $296$ | **$0.7273$** | $11.57$ / $10.0$ / $37.0$ |
| — *Task: `data2text`* | $1,148$ | $948$ | $0.8258$ | $844$ | $103$ | $0.1086$ | $18.67$ / $17.0$ / $36.0$ |
| — *Task: `summarization`* | $903$ | $480$ | $0.5316$ | $521$ | $143$ | $0.2154$ | $20.89$ / $19.0$ / $89.0$ |
| — *Error: `Evident Conflict`* | $981$ | $956$ | $0.9745$ | $937$ | $19$ | $0.0199$ | $18.95$ / $17.0$ / $89.0$ |
| — *Error: `Subtle Conflict`* | $214$ | $183$ | $0.8551$ | $174$ | $9$ | $0.0492$ | $19.42$ / $17.0$ / $62.0$ |
| — *Error: `Evident Baseless Info`* | $1,089$ | $456$ | $0.4187$ | $198$ | $257$ | $0.5636$ | $18.71$ / $16.0$ / $81.0$ |
| — *Error: `Subtle Baseless Info`* | $463$ | $240$ | $0.5184$ | $167$ | $73$ | $0.3042$ | $18.68$ / $16.0$ / $74.0$ |

*Table 3.1: Frozen RAGTruth Dataset Splits and Evidence Annotation Distribution (Layer 1 Audit).*

### 3.2.2 Structure of RAGTruth's `Original:` metadata
Auditing the annotator comments in Layer 1 revealed several important characteristics:
1. **Defined closing markers:** Annotators delimit quotes using five markers: `original:`, `generative:`, `generated:`, `aigc:`, and `aogc:`. Splitting on arbitrary colons incorrectly truncates quotes containing terms like `Review:` or `Hours:`, so we restrict delimiter parsing to this closed set.
2. **Multi-line quotes:** Multi-line quotes (e.g. structured opening hours) wrap across multiple lines before `Generated:`. Extracting only the first line after `Original:` misses details in $6.43\%$ of comments ($118/1,835$). We extract text up to the next marker to retain full multi-line quotes.
3. **Subtle factual labels:** $105$ spans ($3.82\%$) are flagged `implicit_true = True` (claims that are true in general knowledge but absent from the passage). Another $22$ spans are marked `due_to_null = True` (errors resulting from null JSON fields).

---

## 3.3 Experimental or study procedure

Table 3.2 outlines the twelve experimental layers in Component 4.

| Layer | Module (`src/c4_attribution/`) | Experimental Procedure & Scientific Question | Output Artefact on Disk |
| :--- | :--- | :--- | :--- |
| **Layer 1** | `evidence_labels.py`, `evidence_audit.py` | Parse `Original:` blocks on frozen splits; measure verbatim and content-token containment rates across tasks and error types. | `results/c4/analysis/evidence_audit.json` |
| **Layer 1b** | `evidence_audit.py` | Sample $n=50$ (`seed=42`) no-evidence snippets; conduct dual-rater blind human audit and compute raw agreement $P_o$ and Cohen's $\kappa$. | `results/c4/analysis/no_evidence_sample.json`, `no_evidence_sample.md` |
| **Layer 1c** | `sentences.py`, `evidence_audit.py` | Run sentence-segmentation ablation comparing the frozen serving splitter `(?<=[.!?])\s+` against a passage/blank-line-aware variant splitter. | `segmentation_ablation` in `evidence_audit.json` |
| **Layer 2** | `matching.py` | Apply content-token containment matcher at pre-fixed $	heta = 0.50$ under `Convention A` (single best sentence) and `Convention B` (gold set $\ge 	heta$); sweep $	heta \in [0.30, 0.90]$. | `results/c4/analysis/attribution_matching_calib.json`, `attribution_matching_test.json` |
| **Layer 3** | `lexical.py` | Evaluate parameter-free IDF content-token containment scorer against four trivial floors (`first`, `longest`, `random`, `abstain`) per task and convention. | `results/c4/lexical/lexical_metrics.json` |
| **Layer 4** | `prefilter.py` | Benchmark 4 backends (`lexical`, `embed`, `hhem_support`, `nli_entail`) $	imes$ 2 hypothesis modes (`span`, `sentence`); select best operating point on `calib`. | `results/c4/scores/scorer_metrics.json`, `results/c4/scores/{calib,test}_scores.jsonl` |
| **Layer 4b** | `prefilter.py` | Measure pre-filter Recall@k and irrecoverable error floor for $k \in \{1, 3, 5, 10\}$ comparing dense `embedding_used` vs. `idf_lexical_not_used`. | `prefilter` block in `results/c4/scores/scorer_metrics.json` |
| **Layer 5** | `calibration.py` | Fit Temperature, Platt, and Isotonic calibrators on `calib` candidate sentence pairs; evaluate ECE, MCE, Brier, NLL, and AUROC on `test` vs. constant base-rate floor. | `results/c4/calibration/score_calibration.json` |
| **Layer 6** | `evidence_conformal.py` | Construct split conformal evidence sets via LAC and CRC across $\alpha \in \{0.01, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40\}$; test marginal vs. task-conditional coverage. | `results/c4/conformal/c4_conformal.json` |
| **Layer 7** | `abstention.py` | Fit no-evidence abstention threshold $	au$ on `calib` by grid search over $[0.0, 1.0]$; evaluate precision, recall, and F1 on `test` against task-specific `always_abstain` floors. | `results/c4/analysis/no_evidence.json` |
| **Layer 8** | `faithfulness.py` | Audit post-rationalisation / false-citation rate on baseless spans before and after applying the calibrated abstention threshold $	au = 0.1973$. | `results/c4/analysis/faithfulness_audit.json` |
| **Layer 9** | `transfer.py` | Execute Leave-One-Task-Out (LOTO) cross-task transfer (`qa`, `data2text`, `summarization`), fitting $\hat{q}_{1-\alpha}$ and $	au$ on two tasks and testing on the held-out third task. | `results/c4/transfer/transfer_metrics.json` |
| **Layer 10** | `review_cost.py` | Compute reviewer reading-cost curve: mean context sentences read per span and per response vs. empirical coverage across $\alpha$. | `results/c4/analysis/review_cost.json` |
| **Layer 11** | `fusion.py` | Rebuild C2 decisions on $1,509$ test responses; evaluate `c1_alone`, `c1_c2_flag_only`, `c1_c2_gated_by_c4`, and `c4_tie_break` plus 4-cell disagreement table. | `results/c4/fusion/fusion_metrics.json` |
| **Layer 12** | `serving.py` | Export `c4_artifact.json`, verify numerical parity ($\le 10^{-9}$) between offline `lexical` and online `C4Attributor`, and enforce Serving Rules 1–3. | `results/c4/c4_artifact.json` |
| **Layers 13–15** | `latency_benchmark.py`, `uat_validation.py`, `arch_diagrams.py`, `tables.py`, `figures.py` | Benchmark CPU/GPU latency and memory (`C4-15`), execute 10-case UAT and internal team verification timing study (`C4-16`), and render 10 figures + 16 tables. | `results/c4/analysis/latency_benchmark.json`, `uat_validation.json`, `results/c4/tables.md`, `paper/figures/c4_*.png` |

*Table 3.2: Twelve-Layer Experimental Protocol and Associated Artefact Outputs in Component 4.*

### 3.3.1 Matching protocol definitions
In Layer 2 (`src/c4_attribution/matching.py`), we score containment of quote $q$ in sentence $c_j$ as $	ext{cont}(q, c_j) = |T(q) \cap T(c_j)| / |T(q)|$:
* **Convention A (Single Gold Sentence):** We select $j^* = \arg\max_j 	ext{cont}(q, c_j)$. If $	ext{cont}(q, c_{j^*}) \ge 	heta$ ($	heta = 0.50$), $j_{	ext{gold}} = j^*$; otherwise the span is marked unmatched.
* **Convention B (Multiple Gold Sentences):** Any sentence meeting $	ext{cont}(q, c_j) \ge 	heta$ is counted as a valid gold match.

### 3.3.2 Backend comparison setup
Layer 4 compares four backends across raw span queries and enclosing sentence queries:
1. `lexical`: In-context IDF token overlap over all context sentences.
2. `embed`: BGE bi-encoder cosine similarity over all context sentences.
3. `nli_entail`: DeBERTa-v3 cross-encoder entailment score on top-5 pre-filtered sentences.
4. `hhem_support`: Vectara HHEM cross-encoder score on top-5 pre-filtered sentences.

---

## 3.4 Measures and evaluation metrics

We evaluate performance using six core metric groups:
1. **Top-1 Accuracy, MRR, and Recall@k:**
   $$	ext{Top-1} = \frac{1}{N_{	ext{attr}}} \sum_{r=1}^{N_{	ext{attr}}} \mathbb{I}\!\left[\pi_r(1) = j_{	ext{gold}}^{(r)}
ight], \quad 	ext{MRR} = \frac{1}{N_{	ext{attr}}} \sum_{r=1}^{N_{	ext{attr}}} \frac{1}{	ext{rank}_r(j_{	ext{gold}}^{(r)})}$$
   $$	ext{Recall@}k = \frac{1}{N_{	ext{attr}}} \sum_{r=1}^{N_{	ext{attr}}} \mathbb{I}\!\left[	ext{rank}_r(j_{	ext{gold}}^{(r)}) \le k
ight]$$
   We also report conditional Top-1 on cases where the gold sentence survived the candidate pre-filter.
2. **Conformal Empirical Coverage & Finite-Sample Bounds:**
   $$\widehat{	ext{Cov}}(\alpha) = \frac{1}{N_{	ext{test}}} \sum_{r=1}^{N_{	ext{test}}} \mathbb{I}\!\left[j_{	ext{gold}}^{(r)} \in \mathcal{C}_\alpha(s_r)
ight], \quad 	ext{Band}_{3\sigma}(\alpha) = 3 \sqrt{\frac{\alpha(1 - \alpha)}{n_{	ext{test}}} + \frac{\alpha(1 - \alpha)}{n_{	ext{cal}}}}$$
   We compute $95\%$ bootstrap confidence intervals across $1,000$ resamples.
3. **Reviewer Reading Burden Reduction:**
   $$	ext{Reduction}_{	ext{span}}(\alpha) = \frac{1}{N_{	ext{test}}} \sum_{r=1}^{N_{	ext{test}}} \left(1 - \frac{|\mathcal{C}_\alpha(s_r)|}{m_r}
ight)$$
4. **No-Evidence Abstention Precision, Recall, and F1:** Evaluated using standard classification metrics against the task-specific `always_abstain` floor:
   $$	ext{F1}_{	ext{always\_abstain}}(	ext{task}) = \frac{2 \cdot b_{	ext{task}}}{1 + b_{	ext{task}}}$$
5. **Inter-Annotator Agreement (Cohen's $\kappa$):**
   $$\kappa = \frac{P_o - P_e}{1 - P_e}$$
6. **Response-Level Hallucination F1 (Fusion):** Evaluated across the $1,509$ test responses following the RAGTruth benchmark protocol.

---

## 3.5 Data analysis

Analyses are implemented deterministically using Python and NumPy with fixed random seeds (`np.random.default_rng(42)`):
* **Bootstrap intervals:** Coverage bounds are derived from $1,000$ bootstrap iterations.
* **Significance threshold:** Pipeline fusion gains must exceed Component 1's multi-seed variation ($\sigma_{	ext{seed}} = 0.384$) and its $4.18$-point bootstrap interval width.
* **Stratified reporting:** To avoid Simpson's paradox, metrics are reported both overall and disaggregated across QA, Data-to-Text, and Summarization.

---

## 3.6 Ethics privacy security and responsible AI

1. **Compliance with Topic Assessment Form:** In accordance with our signed Topic Assessment Form (Page 9), this project did not collect external personal data or recruit external human participants. All datasets used are public open-access benchmarks (`RAGTruth`, `RAGBench`). Human validation was restricted to internal team review.
2. **Uncertainty Communication Safeguards:**
   * **Serving Rule 2:** Per-span text fields cannot contain terms like `"coverage"` or `"guarantee"`, since conformal coverage is a property of the overall procedure, not an individual span.
   * **Serving Rule 3:** When Component 2 detects out-of-distribution inputs (`guarantee_applies = False`), the interface renders candidate sentences but cleanly withdraws the coverage claim.
3. **Data Privacy:** Because Component 4 runs entirely locally in memory, no user inputs or documents are transmitted to external APIs.

---

## 3.7 Reproducibility and research artefacts

All experiments, tables, and figures can be regenerated from branch `feature/sakindu-c4`:
* Random seeds are fixed to `42`.
* Dataset split IDs are saved in `results/c1/modernbert-base/split_ids.json`.
* Results are recorded in human-readable JSON files under `results/c4/`.
* Reproduction commands:
  * Regenerate tables: `python -m src.c4_attribution.tables --write`
  * Regenerate figures: `python -m src.c4_attribution.figures`
  * Run test suite: `pytest tests/test_c4_*.py`

---

# 4 PRODUCT AND SOLUTION DESIGN

## 4.1 Requirements

Table 4.1 specifies the functional and non-functional requirements for Component 4.

| Req ID | Category | Requirement Description | Target Metric / Acceptance Criterion | Verification Layer & Test |
| :--- | :--- | :--- | :--- | :--- |
| **FR-C4-01** | Functional | Segment raw retrieved context `context` into sentences `context_sentences` and recover exact `(start, end)` character offsets. | $100\%$ parity with backend & C3 splitter `(?<=[.!?])\s+`. | `sentences.py`, `test_c4_sentences.py`, `UAT-01` |
| **FR-C4-02** | Functional | Score every context sentence $c_j$ against each candidate span $s_i$ (or enclosing sentence) and identify the top-1 evidence sentence $j^*$ and score $s_{j^*}$. | Unconditional Top-1 accuracy $\ge 0.85$ on matched test spans (achieved: **$0.8952$**). | `lexical.py`, `serving.py`, `UAT-02` |
| **FR-C4-03** | Functional | Certify `no_evidence` (`abstained = True`) and return `None` for `evidence_sentence`, `evidence_index`, and `entailment_score` when $\max_j s_j < 	au$. | Calibration-fitted $	au = 0.1973$; no-evidence recall $\ge 0.80$ (achieved: **$0.8080$**). | `abstention.py`, `serving.py`, `UAT-03` |
| **FR-C4-04** | Functional | Construct a split conformal evidence prediction set $\mathcal{C}_\alpha(s_i)$ of context sentence indices at user risk level $\alpha$. | Empirical test coverage within $3\sigma$ band of $1 - \alpha$ ($90.03\%$ at $\alpha = 0.10$). | `evidence_conformal.py`, `UAT-04` |
| **FR-C4-05** | Functional | Maintain strict set monotonicity across risk levels: $\mathcal{C}_{\alpha_2}(s_i) \subseteq \mathcal{C}_{\alpha_1}(s_i)$ whenever $\alpha_2 > \alpha_1$. | Verified for all $\alpha \in \{0.01, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40\}$. | `test_c4_conformal.py`, `UAT-05` |
| **FR-C4-06** | Functional | Enforce Serving Rules 1 & 2: never mutate `AnalysisResult.model_version` and block forbidden words (`"coverage"`, `"guarantee"`, `"%"`) in per-span strings. | $0$ contract violations across all $147$ unit tests. | `serving.py`, `test_c4_serving.py`, `UAT-06` |
| **FR-C4-07** | Functional | Enforce Serving Rule 3: when `guarantee_applies == False` (OOD shift detected), continue rendering candidate evidence while withdrawing coverage claims. | `AnalysisResult` validates cleanly with `guarantee_applies=False` and `attribution_checked=True`. | `schema.py`, `serving.py`, `UAT-07` |
| **FR-C4-08** | Functional | Provide a C1$	imes$C2$	imes$C4 tie-break fusion rule (`c4_tie_break`) that promotes C2 `ABSTAIN` spans to positive when C4 certifies `no_evidence = True`. | Statistically significant F1 gain over `c1_c2_flag_only` (achieved: **$+5.96$ F1 pts**). | `fusion.py`, `test_c4_fusion.py`, `UAT-08` |
| **NFR-C4-01** | Non-Functional (Parity) | Guarantee zero numerical drift between the offline evaluation pipeline and the online FastAPI serving class `C4Attributor`. | Max score difference $\le 10^{-9}$ on identical inputs (achieved: **$0.00 	imes 10^0$**). | `test_c4_serving.py`, `UAT-09` |
| **NFR-C4-02** | Non-Functional (Latency) | Execute full context segmentation and multi-span attribution within interactive real-time bounds on a standard laptop CPU. | $<50	ext{ ms}$ per response, $<5	ext{ ms}$ per span (achieved: **$1.312	ext{ ms}$ p50/span, $5.57	ext{ ms}$/resp**). | `latency_benchmark.py`, `UAT-10` |
| **NFR-C4-03** | Non-Functional (Memory) | Co-exist in the same Python process as Component 1's `ModernBERT-base` detector on a 6 GB laptop without GPU OOM errors. | $\le 50	ext{ MB}$ additional RAM/VRAM (achieved: **$0.0	ext{ MB}$ additional model memory**). | `serving.py`, `latency_benchmark.py` |
| **NFR-C4-04** | Non-Functional (Auditability) | Store all serving parameters in a diffable, human-readable JSON artefact rather than opaque binary weights or pickles. | Schema-validated `results/c4/c4_artifact.json` (`v1.0.0`). | `serving.py`, `test_c4_serving.py` |

*Table 4.1: Functional and Non-Functional Requirements Specification for Component 4.*

---

## 4.2 Component architecture within the complete solution

Figure 4.1 shows how Component 4 connects with the other components in TrustRAG.

![Figure 4.1: TrustRAG Four-Stage Calibrated Architecture (Component 4 Highlighted)](file:///S:/Research%20New%20Project/TrustRAG/paper/figures/c4_arch_system.png)
*Figure 4.1: TrustRAG Four-Stage Calibrated Architecture (Component 4 Highlighted).*

In a request to `POST /api/analyze`:
1. **Context Segmentation:** The input passage is split into candidate sentences using the regex `(?<=[.!?])\s+`.
2. **Component 1 (ModernBERT-base):** Highlights candidate spans with character offsets.
3. **Component 2 (Platt Scaling & Conformal Gate):** Assigns calibrated probabilities and decisions (`FLAG`, `ABSTAIN`, `PASS`).
4. **Component 4 (Attribution & Abstention):** Scores sentences via IDF containment, applies the abstention cutoff $	au = 0.1973$, constructs conformal sets $\mathcal{C}_\alpha(s)$, and provides tie-breaking for uncertain spans.
5. **Component 3 (Explanation):** Generates hedged explanations based on risk tiers and identified evidence sentences.

---

## 4.3 Design alternatives and decisions

Table 4.2 reviews key design trade-offs evaluated during development.

| Design Dimension | Alternatives Evaluated | Empirical Findings on Calibration / Test Splits | Final Architectural Decision & Rationale |
| :--- | :--- | :--- | :--- |
| **1. Online vs. Offline Attribution Scorer** | (A) `lexical` (IDF content-token containment)<br>(B) `embed` (`bge-small-en-v1.5`)<br>(C) `hhem_support` (`Vectara HHEM`, $k=5$)<br>(D) `nli_entail` (`DeBERTa-v3-base`, $k=5$) | Unconditional Top-1 (`Table C4-5`):<br>• `lexical`: **$0.8952$** ($1.31	ext{ ms}$, $0	ext{ MB}$)<br>• `hhem_support`: $0.6959$ ($24.6	ext{ ms}$, $440	ext{ MB}$)<br>• `nli_entail`: $0.6392$ ($28.5	ext{ ms}$, $738	ext{ MB}$)<br>• `embed`: $0.3007$ ($8.42	ext{ ms}$, $133	ext{ MB}$) | **Offline 4-backend benchmark + Online `lexical` serving (`c4_artifact.json`).** `lexical` achieves the highest accuracy on both `calib` ($0.9095$) and `test` ($0.8952$) while adding $0	ext{ MB}$ RAM/VRAM on a 6 GB laptop where `ModernBERT` is already loaded. |
| **2. Candidate Pre-Filter Strategy (Layer 4b)** | (A) Dense embedding top-$k$ pre-filter (`bge-small`)<br>(B) IDF lexical top-$k$ pre-filter<br>(C) Exhaustive full-pool lexical scoring | At $k=5$ (`Table C4-6`):<br>• Dense pre-filter drops **$21.90\%$** of gold sentences (`Recall@5` $= 0.7810$).<br>• IDF lexical drops only **$2.41\%$** (`Recall@5` $= 0.9759$). | **Report the $21.90\%$ dense pre-filter bottleneck as a primary scientific finding** and score `lexical` exhaustively over all $m$ sentences in $1.31	ext{ ms}$. |
| **3. Query Hypothesis Formation (Layer 4)** | (A) `span`: raw flagged span substring $s_i$<br>(B) `sentence`: enclosing answer sentence $h(s_i)$ | On `calib` (`Table C4-5`):<br>• `lexical\|sentence`: **$0.9095$** (Conv A), **$0.9629$** (Conv B)<br>• `lexical\|span`: $0.8886$ (Conv A), $0.9629$ (Conv B) | **Select `sentence` hypothesis offline on `calib`** for enclosing-context disambiguation; support direct `span_text` in `C4Attributor` for fast single-call API annotation. |
| **4. Conformal Prediction Formalism (Layer 6)** | (A) Least Ambiguous Set-Valued Classifiers (LAC) [13]<br>(B) Conformal Risk Control (CRC) [14] | At $\alpha = 0.10$ (`Table C4-10`):<br>• LAC: coverage $= 0.9003$, mean set $= 4.61$<br>• CRC: coverage $= 0.9003$, mean set $= 4.61$<br>• $\Delta = 0.0000$ | **Implement and report both LAC and CRC side-by-side (`evidence_conformal.py`)**, shipping LAC quantiles $\hat{q}_{1-\alpha}$ in `c4_artifact.json` due to their closed-form $O(m)$ evaluation. |

*Table 4.2: Architectural Trade-Off Matrix Across Attribution Scorer Backends and Conformal Formalisms.*

---

## 4.4 Detailed design

### 4.4.1 Pipeline flow
Figure 4.2 shows the internal stages of Component 4 from raw sentence scoring to set construction and abstention.

![Figure 4.2: Component 4 Internal Attribution, Conformal Prediction Set, and Abstention Pipeline](file:///S:/Research%20New%20Project/TrustRAG/paper/figures/c4_arch_pipeline.png)
*Figure 4.2: Component 4 Internal Attribution, Conformal Prediction Set, and Abstention Pipeline.*

### 4.4.2 Pydantic schema and JSON artifact
Figure 4.3 illustrates the schema contracts used across the service.

![Figure 4.3: Frozen Pydantic Data Contract and C4 Serving Artefact Schema](file:///S:/Research%20New%20Project/TrustRAG/paper/figures/c4_arch_schema.png)
*Figure 4.3: Frozen Pydantic Data Contract (`src/common/schema.py`) and C4 Serving Artefact Schema.*

The serving configuration (`results/c4/c4_artifact.json`) contains the calibrated parameters:
```json
{
  "schema_version": "1.0.0",
  "component": "c4_attribution",
  "served_backend": "lexical",
  "offline_best_backend": "lexical",
  "abstention_threshold": 0.1973,
  "lac_thresholds": {
    "0.01": 0.9964,
    "0.05": 0.9791,
    "0.1": 0.9613,
    "0.15": 0.9462,
    "0.2": 0.9333,
    "0.3": 0.8744,
    "0.4": 0.8229
  }
}
```

### 4.4.3 Core algorithms
Algorithm 1 outlines online execution in `C4Attributor.annotate`.

```
ALGORITHM 1: Calibrated Evidence Grounding, Conformal Set & No-Evidence Abstention
──────────────────────────────────────────────────────────────────────────────────
Input:  Flagged span text s_i, segmented context sentences C = (c_1, ..., c_m),
        target miscoverage alpha in (0, 1), C4 artefact A = (tau, {q_hat_{1-a}})
Output: Annotation dictionary {evidence_sentence, evidence_index,
        entailment_score, evidence_set, abstained}

1:  if A is empty or m == 0 or s_i is empty then
2:      return {None, None, None, [], abstained = True}
3:  end if
4:  Let T(s_i) <- ContentTokens(s_i)                 // lowercase alphanumeric, len > 1, non-stopword
5:  for j = 1 to m do
6:      T(c_j) <- ContentTokens(c_j)
7:  end for
8:  for each token t in Union_{j=1..m} T(c_j) do
9:      df(t) <- sum_{j=1..m} I[t in T(c_j)]
10:     w(t)  <- ln((m + 1) / (df(t) + 1)) + 1.0     // within-context IDF weight
11: end for
12: w_max <- max_{t} w(t) (or 1.0 if empty)
13: M_span <- sum_{t in T(s_i)} w(t) (using w_max for unseen tokens)
14: for j = 1 to m do
15:     s_j <- (sum_{t in T(s_i) intersect T(c_j)} w(t)) / M_span   if M_span > 0 else 0.0
16: end for
17: j* <- argmax_{j in {1..m}} (s_j, -j)             // deterministic lowest-index tie-break
18: conf <- float(s_{j*})
19: if tau is not None and conf < tau then           // Layer 7 No-Evidence Abstention Gate
20:     return {None, None, None, [], abstained = True}
21: end if
22: a* <- argmin_{a in keys(q_hat)} |a - alpha|      // nearest calibrated risk level
23: s_max <- max(1.0, max_{j} s_j)
24: E_alpha <- [ j in {0..m-1} | 1.0 - (s_j / s_max) <= q_hat_{1-a*} ]
25: return {
        evidence_sentence: C[j*],
        evidence_index:    j*,
        entailment_score:  round(clamp(conf, 0.0, 1.0), 4),
        evidence_set:      E_alpha,
        abstained:         False
    }
```

Algorithm 2 formalises the `c4_tie_break` rule.

```
ALGORITHM 2: C1 x C2 x C4 Conformal Tie-Break Fusion (c4_tie_break)
──────────────────────────────────────────────────────────────────────────────────
Input:  Response d with C1 predicted spans S(d) = {s_1, ..., s_k},
        C2 conformal decisions d_C2(s_i) in {FLAG, ABSTAIN, PASS},
        C4 maximum attribution scores conf(s_i) = max_j s_{i,j},
        calibration-fitted C4 abstention threshold tau = 0.1973
Output: Response-level binary hallucination prediction Y_hat(d) in {0, 1}

1:  Y_hat(d) <- 0
2:  for each predicted span s_i in S(d) do
3:      no_evidence(s_i) <- (conf(s_i) < tau)
4:      if d_C2(s_i) == FLAG then
5:          Y_hat(d) <- 1                            // High-confidence C1+C2 flag retained
6:      else if d_C2(s_i) == ABSTAIN and no_evidence(s_i) is True then
7:          Y_hat(d) <- 1                            // C4 tie-breaker promotes unsupported abstention
8:      end if
9:  end for
10: return Y_hat(d)
```

---

## 4.5 Standards and professional practice

Design decisions adhere to international standards:
1. **ISO/IEC 25010:2023 (SQuaRE):** Governs non-functional performance requirements: low latency (`NFR-C4-02`), zero memory overhead (`NFR-C4-03`), and parity (`NFR-C4-01`).
2. **IEEE Std 1016-2009:** Structures software design descriptions across Figures 4.1, 4.2, and 4.3.
3. **IEEE Std 829 / ISO/IEC/IEEE 29119:** Guides the five-tier software testing strategy.
4. **OpenAPI 3.1.0 & Pydantic v2:** Enforces type safety on all REST endpoints.

---

# 5 IMPLEMENTATION INTEGRATION AND TESTING

## 5.1 Development environment and technology stack

Component 4 runs on Python 3.10 and TypeScript. Table 5.1 lists key dependencies.

| Layer / Role | Technology / Library | Version | Purpose in Component 4 |
| :--- | :--- | :--- | :--- |
| **Core Language** | Python | `3.10` / `3.11` | Primary implementation language for `src/c4_attribution/` and `backend/`. |
| **Numerical Computing** | NumPy | `1.26.4` | Vectorised quantile estimation, ECE binning, LAC/CRC conformal sets, and bootstrap CIs. |
| **Data Contracts** | Pydantic | `2.7.x` | Frozen schema validation (`Span`, `AnalysisResult`, `AnalyzeRequest` in `src/common/schema.py`). |
| **Offline Neural Scorers** | PyTorch + Hugging Face `transformers` | `2.2+` / `4.41+` | Offline benchmarking of `BAAI/bge-small-en-v1.5`, `DeBERTa-v3-base-mnli-fever-anli`, and `HHEM`. |
| **Backend API** | FastAPI + Uvicorn | `0.111+` | Asynchronous REST endpoints (`POST /api/analyze`, `GET /api/metrics`, `GET /api/figures/{name}`). |
| **Frontend UI** | React 18 + TypeScript + Vite + Tailwind | `5.x` | Interactive span highlighting, context sentence scroll-and-highlight, and conformal set display. |
| **Visualisation** | Matplotlib | `3.8.x` | Publication-grade vector `.pdf` and 300-DPI `.png` generation (`figures.py`, `arch_diagrams.py`). |
| **Automated Testing** | Pytest | `8.x` | Execution of $147$ C4 tests and $861$ total repository regression tests. |

*Table 5.1: Development Environment, Software Dependencies, and Pinning Strategy.*

---

## 5.2 Component implementation

The code in `src/c4_attribution/` comprises $18$ Python files. Table 5.2 outlines each module's role.

| Module File (`src/c4_attribution/`) | Layer | Key Functions / Classes | Primary Responsibility |
| :--- | :---: | :--- | :--- |
| `sentences.py` | Layer 1c | `split_sentences`, `split_sentences_variant`, `sentence_offsets`, `realign` | Implements the frozen regex splitter `(?<=[.!?])\s+`, character-offset recovery, and the passage/blank-line variant splitter. |
| `evidence_labels.py` | Layer 1 | `original_snippet`, `is_no_evidence`, `label_from_annotation`, `marker_counts` | Extracts `Original:` blocks up to the closed marker tuple `(original, generative, generated, aigc, aogc)` and detects no-evidence phrases. |
| `dataset.py` | Layer 1 | `load_responses`, `span_records`, `predicted_span_records` | Loads frozen C1 `calib` and `test` splits (`split_ids.json`) and pairs responses with gold or C1-predicted spans. |
| `evidence_audit.py` | Layer 1 & 1b | `measure`, `build_report`, `hand_check_agreement`, `_cohens_kappa` | Audits verbatim/containment rates, convention disagreement, splitter ablation, and dual-rater blind agreement ($\kappa = 0.8466$). |
| `matching.py` | Layer 2 | `match_span`, `theta_sweep`, `build_report` | Implements the $	heta = 0.50$ content-token containment gold-sentence matcher under Conventions A and B. |
| `lexical.py` | Layer 3 | `idf_overlap_scores`, `best_sentence`, `containment`, `floor_prediction` | Implements within-context IDF lexical containment and the four trivial floors (`first`, `longest`, `random`, `abstain`). |
| `prefilter.py` | Layer 4 & 4b | `score_row`, `prefilter_ceiling`, `select_operating_point` | Evaluates the 4 backends $	imes$ 2 hypothesis rules and measures the pre-filter Recall@k error ceiling. |
| `calibration.py` | Layer 5 | `TemperatureCalibrator`, `PlattCalibrator`, `IsotonicCalibrator`, `ece_mce` | Fits post-hoc probability calibrators on `calib` and scores ECE, MCE, Brier, NLL, and AUROC on `test` against the constant floor. |
| `evidence_conformal.py` | Layer 6 | `lac_quantile`, `lac_sets`, `crc_threshold`, `evaluate_alpha`, `per_task_breakdown` | Implements split conformal LAC and CRC prediction sets, $3\sigma$ coverage bands, bootstrap CIs, and task-conditional sets. |
| `abstention.py` | Layer 7 | `fit_threshold`, `evaluate_threshold`, `task_floors` | Fits the no-evidence abstention threshold ($	au = 0.1973$) on `calib` and evaluates precision, recall, and F1 against task floors. |
| `faithfulness.py` | Layer 8 | `build_report`, `audit_false_citations` | Measures how often an unthresholded attributor fabricates citations on baseless spans vs. post-abstention gate. |
| `transfer.py` | Layer 9 | `leave_one_task_out`, `evaluate_fold` | Runs Leave-One-Task-Out cross-task transfer across `qa`, `data2text`, and `summarization`. |
| `review_cost.py` | Layer 10 | `span_reading_cost`, `response_compression`, `build_report` | Computes the reviewer reading-cost curve (`4.61` vs. `19.05` sentences, `70.63%` reduction). |
| `fusion.py` | Layer 11 | `rebuild_c2_decisions`, `disagreement_cells`, `fuse`, `example_f1` | Rebuilds C2 conformal decisions (`0.8808` coverage verified) and evaluates the four C1$	imes$C2$	imes$C4 fusion rules. |
| `serving.py` | Layer 12 | `C4Attributor`, `build_artifact`, `evidence_is_safe` | Ships the dependency-free online attributor, `c4_artifact.json`, and Serving Rules 1–3. |
| `latency_benchmark.py` | Layer 13 | `run_benchmark` | Benchmarks CPU/GPU p50/p95 latency, span throughput, and RAM/VRAM footprint across all four backends (`C4-15`). |
| `uat_validation.py` | Layer 14 | `run_uat_suite`, `_ensure_dual_rater_audit` | Executes the 10 functional/non-functional UAT cases and compiles the internal reviewer verification timing study (`C4-16`). |
| `tables.py`, `figures.py`, `arch_diagrams.py` | Layer 15 | `BUILDERS`, `build_all` | Generates all 16 markdown tables (`C4-1` to `C4-16`), 7 empirical figures, and 3 Chapter 4 architecture diagrams. |

*Table 5.2: Layer-by-Layer Module Inventory of `src/c4_attribution/`.*

---

## 5.3 Integration with the other components

Component 4 interfaces with the rest of TrustRAG across three clear touchpoints:
1. **Upstream with Components 1 & 2:** Contexts segmented in C4 match those used by C1 and C2 character-for-character. In Layer 11, C4 rebuilds C2 decisions to verify recorded test coverage ($0.8808$) before running fusion.
2. **Peer with Component 3:** C3 and C4 share the regex splitter `(?<=[.!?])\s+`. C3 uses C4's sentence references to form explanations and respects C4's abstention decisions.
3. **Live FastAPI & React Integration:** `C4Attributor` is loaded into FastAPI at startup, adding $0	ext{ MB}$ memory and annotating spans via `C4Attributor.annotate`.

---

## 5.4 Verification and software testing

We apply a five-tier testing strategy comprising **$147$ unit and contract tests** across $13$ test files (`tests/test_c4_*.py`), all of which pass ($0$ failures). Table 5.3 provides the test breakdown.

| Testing Level | Test Files (`tests/`) | Number of Tests | Key Invariants Verified | Status |
| :--- | :--- | :---: | :--- | :---: |
| **Level 1: Unit Tests** | `test_c4_sentences.py`, `test_c4_evidence_labels.py`, `test_c4_matching.py`, `test_c4_lexical.py` | $54$ | Closed 5-marker vocabulary; non-truncation on `Weekends:`; IDF tokenisation; deterministic lowest-index tie-breaking; $	heta = 0.50$ matching; dual-rater Cohen's $\kappa = 0.8466$. | **54 / 54 PASS** |
| **Level 2: Component & Calibration Tests** | `test_c4_calibration.py`, `test_c4_conformal.py`, `test_c4_abstention.py`, `test_c4_faithfulness.py` | $46$ | Strict fitting on `calib` only; LAC quantile finite-sample bound; CRC vs. LAC equivalence; risk-level set nesting $\mathcal{C}_{0.20} \subseteq \mathcal{C}_{0.10} \subseteq \mathcal{C}_{0.05}$; abstention threshold $	au$ bounds. | **46 / 46 PASS** |
| **Level 3: Cross-Component Integration Tests** | `test_c4_dataset.py`, `test_c4_transfer.py`, `test_c4_review_cost.py`, `test_c4_fusion.py` | $27$ | Splitter parity across Backend, C3, and C4; C2 coverage rebuild parity ($0.8807531380753139$); 4-cell disagreement sum conservation; LOTO fold disjointness. | **27 / 27 PASS** |
| **Level 4: Serving & Contract Safety Tests** | `test_c4_serving.py`, `test_c4_tables_figures.py` | $20$ | Serving Rule 1 (no `+c4` on `model_version`); Serving Rule 2 (no `"coverage"`, `"guarantee"`, `"%"` in span strings); Serving Rule 3 (OOD withdrawal); $\le 10^{-9}$ offline/online parity; `FIGURE_ORDER` whitelist lockstep; `C4-1`..`C4-16` contiguity. | **20 / 20 PASS** |
| **Level 5: System Benchmark & UAT** | `latency_benchmark.py`, `uat_validation.py` | $10$ UAT cases | End-to-end `AnalyzeRequest` $
ightarrow$ `AnalysisResult` validation, $<50	ext{ ms}$ CPU latency ($1.312	ext{ ms}$ p50/span), $0	ext{ MB}$ memory overhead, and internal reviewer effort reduction ($69.47\%$ time saved). | **10 / 10 PASS** |

*Table 5.3: Five-Level Software Verification and Regression Test Matrix (`147` C4 Tests / `861` Suite Tests).*

---

## 5.5 Deployment and operation

* **Offline Evaluation:** Heavy neural scoring runs once offline via `prefilter.py`, saving score files to `results/c4/scores/` and calibration thresholds to `results/c4/c4_artifact.json`.
* **Live Serving:** In production, FastAPI loads `c4_artifact.json` at startup. Because `C4Attributor` uses only standard Python libraries and NumPy, launching the service requires **no additional model weights, zero extra VRAM, and adds only $1.312	ext{ ms}$ median latency per span**.

---

# 6 RESULTS AND VALIDATION

This chapter presents the empirical validation of **Component 4 (Retrieval-Attribution Alignment and Evidence Grounding)** across all sixteen research and technical evaluation tables (`Tables 6.1 [C4-1]` through `6.16 [C4-14]`) and seven empirical figures (`Figures 6.1` through `6.7`), generated directly from the committed JSON artefacts under `results/c4/`.

---

## 6.1 Research results

### 6.1.1 Layer 1: Evidence supervision audit, verbatim rates, and matching protocol (`C4-1` to `C4-3`)
Before evaluating any attribution scorer, Layer 1 (`src/c4_attribution/evidence_audit.py`) audited the human `Original:` supervision across Component 1's frozen calibration (`calib`) and test (`test`) splits. Table 6.1 (`C4-1`) reports the exact supervision counts.

| Split | Annotated Spans | With `Original:` | Share with `Original:` | Attributable (Quote Exists) | No Evidence (Denial) | Share No Evidence of `Original:` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`calib`** | $1,214$ | $787$ | $0.6483$ | $652$ | $134$ | $0.1703$ |
| **`test`** | $1,533$ | $1,048$ | $0.6836$ | $824$ | $224$ | $0.2137$ |

*Table 6.1 (C4-1): Evidence Supervision on the Frozen Calibration and Test Splits (`results/c4/analysis/evidence_audit.json`).*

Across both splits, $64.83\%$ of calibration spans ($787/1,214$) and $68.36\%$ of test spans ($1,048/1,533$) carry an explicit `Original:` note. Of the $1,048$ test spans with an `Original:` marker, $824$ ($78.63\%$) cite source content, while $224$ ($21.37\%$) state that no evidence exists in the text. To verify our regex phrase list (`src/c4_attribution/evidence_labels.py`), two team members independently audited a random sample of $n = 50$ snippets (`seed = 42`): Rater 1 agreed with $47/50$ cases ($0.9400$), and Rater 2 agreed with $46/50$ ($0.9200$), giving raw agreement $P_o = 0.9800$ and **Cohen's $\kappa = 0.8466$**. The three regex misfires (`response_id` `3082`, `8108`, and `15178`) occurred when an annotator quoted a real sentence but added a comment (e.g., `"\"lot\": true (but not stated whether it is free or not)"`).

Table 6.2 (`C4-2`) breaks down how the $1,476$ attributable quotes behave across tasks and hallucination types.

| Group / Stratum | Attributable $n$ | Verbatim in Context | Containment $\ge 0.9$ | Mean Containment | Share No Evidence |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Task: `qa`** | $111$ | $0.6306$ | $0.8559$ | $0.9163$ | **$0.7273$** |
| **Task: `data2text`** | $844$ | $0.4254$ | $0.6386$ | $0.7850$ | $0.1086$ |
| **Task: `summarization`** | $521$ | $0.6123$ | $0.7562$ | $0.8497$ | $0.2154$ |
| **Error: `Evident Conflict`** | $937$ | $0.5176$ | $0.7086$ | $0.8313$ | $0.0199$ |
| **Error: `Subtle Conflict`** | $174$ | $0.6092$ | $0.7644$ | $0.8565$ | $0.0492$ |
| **Error: `Evident Baseless Info`** | $198$ | $0.3586$ | $0.5606$ | $0.7097$ | **$0.5636$** |
| **Error: `Subtle Baseless Info`** | $167$ | $0.5150$ | $0.7186$ | $0.8289$ | $0.3042$ |
| **All Combined (`calib` + `test`)** | **$1,476$** | **$0.5068$** | **$0.6965$** | **$0.8177$** | **$0.1951$** |

*Table 6.2 (C4-2): Verbatim Rate, Content-Token Containment, and No-Evidence Share by Task and Error Type.*

Table 6.2 shows two important trends:
1. **Human evidence quotes are often non-verbatim:** Only $50.68\%$ of quotes match verbatim substrings in the text ($42.54\%$ in `data2text` because annotators convert JSON keys into natural sentences). Mean content-token containment is **$0.8177$** overall ($0.9163$ on `qa`, $0.8497$ on `summarization`, and $0.7850$ on `data2text`), confirming that token containment is a fairer matching metric than exact string match.
2. **No-evidence rates differ sharply by task:** In `qa`, **$72.73\%$** of `Original:` comments state that no evidence exists, compared to $21.54\%$ in `summarization` and $10.86\%$ in `data2text`. For error types, conflicts almost always point to a sentence ($1.99\%$ and $4.92\%$ no-evidence), whereas baseless errors frequently have no source sentence at all ($56.36\%$ and $30.42\%$).
3. **Sentence splitter check (`Layer 1c`):** Testing the variant splitter (`split_sentences_variant`) shifted the best-matching sentence index on **$0$ of $1,476$ spans ($0.0000$)** and changed the candidate pool on only $1$ span ($0.0678\%$, pool size $18.9180$ vs $18.9194$). Using the regex splitter `(?<=[.!?])\s+` therefore maintains full compatibility across components with zero loss of accuracy.

Table 6.3 (`C4-3`) shows the matching protocol results on the $1,533$ test spans at $\theta = 0.50$, along with a sensitivity sweep across $\theta \in [0.30, 0.90]$.

| Protocol / Sweep | Category / $\theta$ | Count / Conv A Matched | Share / Conv A Rate | Conv B Matched | Conv B Rate | Unmatched |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Test Split Partition ($\theta = 0.50$)** | `matched_attributable` | $582$ | $0.3796$ | — | — | — |
| | `unmatched_attributable` | $242$ | $0.1579$ | — | — | — |
| | `no_evidence` | $224$ | $0.1461$ | — | — | — |
| | `no_original_marker` | $485$ | $0.3164$ | — | — | — |
| **$\theta$ Sensitivity Sweep ($n = 824$)** | $\theta = 0.30$ | $791$ | $0.9600$ | $791$ | $0.9600$ | $33$ |
| | $\theta = 0.40$ | $767$ | $0.9308$ | $767$ | $0.9308$ | $57$ |
| | **$\theta = 0.50$ (Pre-fixed)** | **$582$** | **$0.7063$** | **$582$** | **$0.7063$** | **$242$** |
| | $\theta = 0.60$ | $562$ | $0.6820$ | $562$ | $0.6820$ | $262$ |
| | $\theta = 0.70$ | $546$ | $0.6626$ | $546$ | $0.6626$ | $278$ |
| | $\theta = 0.80$ | $534$ | $0.6481$ | $534$ | $0.6481$ | $290$ |
| | $\theta = 0.90$ | $521$ | $0.6323$ | $521$ | $0.6323$ | $303$ |

*Table 6.3 (C4-3): Attribution Matching Protocol and Containment Threshold $\theta$ Sweep on the Test Split.*

---

### 6.1.2 Layers 3, 4 & 4b: Lexical baseline, multi-backend comparison, and pre-filter ceiling (`C4-4` to `C4-6`)
Table 6.4 (`C4-4`) evaluates our within-context IDF lexical scorer against four trivial floors on the $582$ matched test spans under Convention A.

| Task | Scored Spans ($n$) | Lexical Top-1 | Lexical MRR | Floor: `first` | Floor: `longest` | Floor: `random` | Floor: `abstain` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`qa`** | $57$ | $0.6842$ | $0.7931$ | $0.0877$ | $0.1404$ | $0.0877$ | $0.0000$ |
| **`data2text`** | $291$ | **$0.9794$** | **$0.9885$** | $0.0034$ | **$0.5430$** | $0.0687$ | $0.0000$ |
| **`summarization`** | $234$ | $0.8419$ | $0.8924$ | $0.0897$ | $0.1325$ | $0.0513$ | $0.0000$ |
| **Pooled (`test`)** | **$582$** | **$0.8952$** | **$0.9307$** | **$0.0464$** | **$0.3385$** | **$0.0636$** | **$0.0000$** |

*Table 6.4 (C4-4): Lexical Attribution Baseline Against Four Trivial Floors on the Test Split (Convention A).*

Breaking down the results by task is critical. In `data2text`, `floor_longest` achieves **$0.5430$ Top-1 accuracy** ($0.3385$ overall) simply because JSON records serialise into one long string. Even against this high floor, IDF lexical containment reaches **$0.9794$ Top-1** on `data2text`, **$0.8419$** on `summarization`, and **$0.6842$** on `qa`.

Table 6.5 (`C4-5`) and **Figure 6.1** compare the four attribution backends on the test split ($n = 582$).

| Backend | Hypothesis | Test $n$ | Unconditional Top-1 | MRR | Recall@5 | Top-1 Given Gold Survived Pre-Filter | Pre-Filtered ($k=5$)? | Calib Top-1 (Conv A / Conv B, $n=431$) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`lexical` (Selected)** | **`sentence`** | **$582$** | **$0.8952$** | **$0.9307$** | **$0.9759$** | **$0.8952$** | **No** | **$0.9095$ / $0.9629$** |
| `lexical` | `span` | $582$ | $0.8952$ | $0.9307$ | $0.9759$ | $0.8952$ | No | $0.8886$ / $0.9629$ |
| `embed` (`bge-small`) | `sentence` | $582$ | $0.3007$ | $0.5068$ | $0.7818$ | $0.3007$ | No | $0.3480$ / $0.4594$ |
| `embed` (`bge-small`) | `span` | $582$ | $0.3007$ | $0.5068$ | $0.7818$ | $0.3007$ | No | $0.3480$ / $0.4594$ |
| `hhem_support` | `sentence` | $582$ | $0.6959$ | $0.7599$ | $0.7818$ | **$0.8901$** | Yes ($k=5$) | $0.7796$ / $0.9234$ |
| `hhem_support` | `span` | $582$ | $0.6959$ | $0.7599$ | $0.7818$ | **$0.8901$** | Yes ($k=5$) | $0.7796$ / $0.9234$ |
| `nli_entail` (`DeBERTa-v3`) | `sentence` | $582$ | $0.6392$ | $0.7228$ | $0.7818$ | $0.8176$ | Yes ($k=5$) | $0.7169$ / $0.8863$ |
| `nli_entail` (`DeBERTa-v3`) | `span` | $582$ | $0.6392$ | $0.7228$ | $0.7818$ | $0.8176$ | Yes ($k=5$) | $0.7169$ / $0.8863$ |

*Table 6.5 (C4-5): Attribution Backend Comparison and Hypothesis-Formation Ablations on the Test Split.*

![Figure 6.1: Attribution Backend Top-1 Accuracy against Four Trivial Floors by Task and Pre-Filter Ceiling](file:///S:/Research%20New%20Project/TrustRAG/paper/figures/c4_attribution_accuracy.png)
*Figure 6.1: Attribution Backend Top-1 Accuracy against Four Trivial Floors by Task and Pre-Filter Ceiling.*

To understand why `hhem_support` ($0.6959$) and `nli_entail` ($0.6392$) lag behind `lexical` ($0.8952$), Layer 4b (`Table 6.6 [C4-6]`) measures the pre-filter's recall ceiling across $k \in \{1, 3, 5, 10\}$ on $580$ logged test spans.

| Pre-Filter Strategy | Top-$k$ Cut-off | Scored Spans ($n$) | Recall@$k$ | Irrecoverable Error Floor ($1 - \text{Recall@}k$) |
| :--- | :---: | :---: | :---: | :---: |
| **`embedding_used` (`BAAI/bge-small-en-v1.5`)** | $k = 1$ | $580$ | $0.3000$ | $0.7000$ ($70.00\%$) |
| | $k = 3$ | $580$ | $0.6845$ | $0.3155$ ($31.55\%$) |
| | **$k = 5$ (Used by Cross-Encoders)** | **$580$** | **$0.7810$** | **$0.2190$ ($21.90\%$)** |
| | $k = 10$ | $580$ | $0.8862$ | $0.1138$ ($11.38\%$) |
| **`idf_lexical_not_used` (Within-Context IDF)** | $k = 1$ | $580$ | $0.8948$ | $0.1052$ ($10.52\%$) |
| | $k = 3$ | $580$ | $0.9603$ | $0.0397$ ($3.97\%$) |
| | **$k = 5$ (Counterfactual Comparison)** | **$580$** | **$0.9759$** | **$0.0241$ ($2.41\%$)** |
| | $k = 10$ | $580$ | $0.9966$ | $0.0034$ ($0.34\%$) |

*Table 6.6 (C4-6): Candidate Pre-Filter Recall@k and Irrecoverable Error Floor Across $k \in \{1, 3, 5, 10\}$.*

Tables 6.5 and 6.6 highlight three clear conclusions:
* Pre-filtering with `bge-small` at $k=5$ yields only **$0.7810$ Recall@5**, which means **$21.90\%$ of the gold sentences are discarded before the cross-encoders even run**.
* When the gold sentence survives the pre-filter, `hhem_support` reaches **$0.8901$** Top-1 (matching `lexical` at $0.8952$), confirming that the pre-filter is the primary bottleneck.
* In contrast, IDF lexical ranking reaches **$0.9759$ Recall@5** (an error floor of just $2.41\%$, nearly ten times lower). Furthermore, because `lexical` runs in $1.312\text{ ms}$ over the entire context pool without pre-filtering, it avoids this issue entirely.

---

### 6.1.3 Layer 5: Post-hoc attribution score calibration (`C4-7`)
Layer 5 (`src/c4_attribution/calibration.py`) evaluated score calibration across $11,033$ `(span, context_sentence)` pairs in the test split. Table 6.7 (`C4-7`) and **Figure 6.2** report the results against the constant base-rate floor ($\\hat{p} \equiv \bar{y}_{\text{cal}} = 0.0528$).

| Backend (`n_test = 11,033` pairs) | Calibration Method | ECE $\downarrow$ | MCE $\downarrow$ | Brier $\downarrow$ | NLL $\downarrow$ | AUROC $\uparrow$ | Notes |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **`lexical`** | `uncalibrated` | $0.1458$ | $0.2738$ | $0.0497$ | $0.3204$ | $0.9854$ | Raw IDF overlap ratio $\in [0,1]$ |
| | `temperature` | $0.3886$ | $0.4534$ | $0.1665$ | $0.5177$ | $0.9854$ | Single-$T$ scaling around $0.5$ distorts sparse zeros |
| | `platt` | $0.0235$ | $0.2896$ | **$0.0176$** | **$0.0680$** | $0.9854$ | Two-parameter logistic shift fits sparse base rate |
| | **`isotonic`** | **$0.0067$** | **$0.1303$** | **$0.0161$** | **$0.0620$** | **$0.9851$** | Best calibrated & lowest Brier/NLL |
| | `constant_base_rate` (Floor) | $0.0012$ | $0.0012$ | $0.0500$ | $0.2067$ | $0.5000$ | Floor: ignores input entirely |
| **`nli_entail`** | `uncalibrated` | $0.0156$ | $0.4310$ | $0.0462$ | $0.6808$ | $0.8576$ | Pre-filtered non-candidates padded as $0$ |
| | `isotonic` | **$0.0079$** | $0.3127$ | **$0.0308$** | **$0.1292$** | $0.8581$ | Beats constant Brier ($0.0500$) |
| **`hhem_support`** | `uncalibrated` | $0.0243$ | $0.1932$ | $0.0363$ | $0.6134$ | $0.8655$ | Pre-filtered non-candidates padded as $0$ |
| | `isotonic` | **$0.0077$** | **$0.0972$** | **$0.0268$** | **$0.1157$** | **$0.8675$** | Best cross-encoder calibration |

*Table 6.7 (C4-7): Post-Hoc Attribution Score Calibration (Temperature, Platt, Isotonic) vs. Constant Floor.*

![Figure 6.2: Attribution Score Reliability Diagrams and Expected Calibration Error Across Calibrators](file:///S:/Research%20New%20Project/TrustRAG/paper/figures/c4_score_calibration.png)
*Figure 6.2: Attribution Score Reliability Diagrams and Expected Calibration Error Across Calibrators.*

As shown in Table 6.7, raw lexical scores have strong discriminative ranking (**$\text{AUROC} = 0.9854$**). Isotonic regression reduces ECE by $95.4\%$ (from $0.1458$ to **$0.0067$**) and lowers Brier score to **$0.0161$** ($67.8\%$ better than the constant baseline). Temperature scaling performs poorly here because mapping near-zero lexical scores through logit transforms creates extreme negative values that a single scalar $T$ cannot balance; Platt scaling and Isotonic regression easily fix this with learned intercepts.

---

### 6.1.4 Layer 6 & Layer 10: Split conformal evidence sets, task heterogeneity, LAC vs. CRC, and reviewer effort reduction (`C4-8` to `C4-10`, `C4-13`)
Tables 6.8 (`C4-8`), 6.9 (`C4-9`), 6.10 (`C4-10`), and 6.13 (`C4-13`) evaluate conformal evidence prediction sets fitted on calibration ($n_{\text{cal}} = 431$) and tested on the held-out test split ($n_{\text{test}} = 582$).

| $\alpha$ | Target Coverage ($1-\alpha$) | Empirical Coverage | $95\%$ Bootstrap CI Low | $95\%$ Bootstrap CI High | $3\sigma$ Finite-Sample Band | Inside Band? | Mean Set Size (Sentences) | Singleton Rate | Empty Set Rate | Mean Pool Size | Reading Reduction / Span |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$0.01$** | $0.99$ | $1.0000$ | $1.0000$ | $1.0000$ | $0.0191$ | **yes** | $19.05$ | $0.0000$ | $0.0000$ | $19.1$ | $0.0000$ ($0.0\%$) |
| **$0.05$** | $0.95$ | $1.0000$ | $1.0000$ | $1.0000$ | $0.0418$ | **yes** | $19.05$ | $0.0000$ | $0.0000$ | $19.1$ | $0.0000$ ($0.0\%$) |
| **$0.10$** | **$0.90$** | **$0.9003$** | **$0.8640$** | **$0.9338$** | **$0.0575$** | **yes** | **$4.61$** | **$0.1254$** | **$0.0000$** | **$19.1$** | **$0.7063$ ($70.6\%$)** |
| **$0.15$** | $0.85$ | $0.8643$ | $0.8226$ | $0.9037$ | $0.0685$ | **yes** | $3.71$ | $0.1838$ | $0.0000$ | $19.1$ | $0.7448$ ($74.5\%$) |
| **$0.20$** | $0.80$ | $0.8196$ | $0.7745$ | $0.8621$ | $0.0767$ | **yes** | $2.81$ | $0.2509$ | $0.0000$ | $19.1$ | $0.7913$ ($79.1\%$) |
| **$0.30$** | $0.70$ | $0.7165$ | $0.6657$ | $0.7646$ | $0.0879$ | **yes** | $1.38$ | $0.5430$ | $0.0533$ | $19.1$ | $0.8909$ ($89.1\%$) |
| **$0.40$** | $0.60$ | $0.6306$ | $0.5801$ | $0.6836$ | $0.0940$ | **yes** | $0.81$ | $0.5945$ | $0.2491$ | $19.1$ | $0.9440$ ($94.4\%$) |

*Table 6.8 (C4-8 & C4-13 Combined): Split Conformal Evidence Prediction Sets and Reviewer Reading Reduction on the Test Split.*

![Figure 6.3: Conformal Evidence Set Empirical Coverage vs. Target 1 - alpha with 3-sigma Finite-Sample Band](file:///S:/Research%20New%20Project/TrustRAG/paper/figures/c4_conformal_coverage.png)
*Figure 6.3: Conformal Evidence Set Empirical Coverage vs. Target $1 - \alpha$ with $3\sigma$ Finite-Sample Band.*

![Figure 6.4: Mean Conformal Evidence Set Size vs. Candidate Context Pool Size Across Risk Levels](file:///S:/Research%20New%20Project/TrustRAG/paper/figures/c4_conformal_set_size.png)
*Figure 6.4: Mean Conformal Evidence Set Size vs. Candidate Context Pool Size Across Risk Levels.*

![Figure 6.7: Reviewer Reading-Cost Curve: Context Sentences Read per Flagged Span vs. Empirical Coverage](file:///S:/Research%20New%20Project/TrustRAG/paper/figures/c4_review_cost_curve.png)
*Figure 6.7: Reviewer Reading-Cost Curve: Context Sentences Read per Flagged Span vs. Empirical Coverage.*

Table 6.8 supports two key findings:
* **Reliable Coverage Across Risk Levels:** At all tested levels $\alpha \in \{0.01, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40\}$, empirical coverage remains inside the expected $3\sigma$ finite-sample band.
* **Target Calibration ($\\alpha = 0.10$):** At $90\%$ target coverage, test coverage is **$0.9003$** ($90.03\%$, $95\%$ CI $[0.8640, 0.9338]$), and the average conformal set shrinks from **$19.05$ sentences** to **$4.61$ sentences per span**, delivering a **$70.63\%$ reduction in reviewer reading burden**.

Table 6.9 (`C4-9`) examines performance across individual tasks at $\alpha = 0.10$.

| Task | $n_{\text{cal}}$ | $n_{\text{test}}$ | Marginal Threshold Coverage | Marginal Mean Set Size | Inside $3\sigma$ Band? | Group-Conditional Coverage | Group-Conditional Set Size |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`qa`** | $42$ | $57$ | $0.8421$ | $4.79$ | **yes** | $1.0000$ | $11.58$ |
| **`data2text`** | $209$ | $291$ | $0.9038$ | $1.75$ | **yes** | $0.8900$ | $1.59$ |
| **`summarization`** | $180$ | $234$ | $0.9103$ | $8.13$ | **yes** | $0.9017$ | $7.70$ |

*Table 6.9 (C4-9): Marginal vs. Group-Conditional Conformal Coverage and Set Size by Task at $\alpha = 0.10$.*

While all three tasks satisfy the coverage band, the average set size varies by task: $1.75$ sentences for `data2text`, $4.79$ for `qa`, and $8.13$ for `summarization` (where long news articles frequently mention the central entity across multiple sentences).

Table 6.10 (`C4-10`) compares LAC against Conformal Risk Control (CRC).

| Conformal Formalism | Empirical Coverage ($\alpha = 0.10$) | Mean Set Size (Sentences) |
| :--- | :---: | :---: |
| **Least Ambiguous Set-Valued Classifiers (LAC)** | $0.9003$ | $4.61$ |
| **Conformal Risk Control (CRC, Binary Miss Loss)** | $0.9003$ | $4.61$ |
| **Difference ($\\text{LAC} - \text{CRC}$)** | **$0.0000$** | **$0.00$** |

*Table 6.10 (C4-10): Comparison of Least Ambiguous Set-Valued Classifiers (LAC) and Conformal Risk Control (CRC).*

Because single-sentence attribution uses a binary 0/1 miss loss under Convention A, LAC and CRC yield identical sets and empirical coverage ($0.9003$, $4.61$ sentences).

---

### 6.1.5 Layers 7, 8 & 9: Calibrated no-evidence abstention, faithfulness audit, and cross-task transfer (`C4-11`, `C4-12`)
Table 6.11 (`C4-11`) and **Figure 6.5** evaluate the no-evidence abstention threshold ($\\tau = 0.1973$, fitted on $787$ calibration spans) on the $1,048$ test spans.

| Evaluation Slice / System | $n$ | No-Evidence Base Rate | Precision | Recall | F1 Score | Abstention Rate | Always-Abstain Floor F1 | Beats Task Floor? |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Pooled Test: C4 ($\tau = 0.1973$, fitted on `calib`)** | $1,048$ | $0.2137$ | **$0.3428$** | **$0.8080$** | **$0.4814$** | $0.5038$ | $0.3522$ | **yes ($+12.92$ pts)** |
| — Floor: `always_abstain` | $1,048$ | $0.2137$ | $0.2137$ | $1.0000$ | $0.3522$ | $1.0000$ | — | — |
| — Floor: `never_abstain` | $1,048$ | $0.2137$ | $0.0000$ | $0.0000$ | $0.0000$ | $0.0000$ | — | — |
| — Baseline: `nearest_sentence, no threshold` (C3) | $1,048$ | $0.2137$ | $\text{n/a}$ | **$0.0000$** | **$0.0000$** | $0.0000$ | — | — |
| **Per-Task Breakdown (`test`):** | | | | | | | | |
| — Task: **`data2text`** | $523$ | $0.1112$ | $0.2102$ | $0.8909$ | **$0.3400$** | $0.4713$ | $0.2002$ | **yes ($+13.98$ pts)** |
| — Task: **`summarization`** | $382$ | $0.2736$ | $0.3679$ | $0.7179$ | **$0.4865$** | $0.5340$ | $0.4293$ | **yes ($+5.72$ pts)** |
| — Task: **`qa`** | $143$ | **$0.7273$** | $0.7681$ | $0.8780$ | $0.8191$ | $0.8322$ | **$0.8421$** | **no ($-2.30$ pts)** |

*Table 6.11 (C4-11): Calibrated No-Evidence Abstention Performance Against Task-Specific Floors.*

![Figure 6.5: Calibrated No-Evidence Abstention Precision, Recall, and F1 vs. Task-Specific Floors](file:///S:/Research%20New%20Project/TrustRAG/paper/figures/c4_no_evidence_pr.png)
*Figure 6.5: Calibrated No-Evidence Abstention Precision, Recall, and F1 vs. Task-Specific Floors.*

Table 6.11 demonstrates:
* Unthresholded nearest-neighbour retrieval (`best_matching_sentence`) always returns a sentence, scoring **$0.0000$ recall** on no-evidence spans.
* Component 4's calibrated threshold ($\\tau = 0.1973$) catches **$80.80\%$ of baseless spans**, improving over the `always_abstain` floor by **$+12.92$ F1 points** pooled ($0.4814$ vs. $0.3522$), $+13.98$ on `data2text`, and $+5.72$ on `summarization`.
* On `qa`, where $72.73\%$ of spans have no context evidence, the trivial baseline of abstaining on every span yields an F1 of $0.8421$. Component 4 scores $0.8191$ F1 on `qa` while preserving the ability to cite evidence on the remaining $27.27\%$ of attributable cases.

Table 6.12 (`C4-12`) and **Figure 6.6** evaluate Leave-One-Task-Out (LOTO) cross-task transfer (`src/c4_attribution/transfer.py`).

| Held-Out Task | Test $n$ (Conformal) | Coverage Transferred | Coverage In-Task | Inside $3\sigma$ Band? | Set Size Transferred | Set Size In-Task | Abstention F1 Transferred | Abstention F1 In-Task | Top-1 Accuracy (Invariant) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`qa`** | $57$ | $0.8421$ | $1.0000$ | **yes** | $4.79$ | $11.58$ | $0.8182$ | $0.8421$ | $0.6842$ |
| **`data2text`** | $291$ | $0.9175$ | $0.8900$ | **yes** | $2.03$ | $1.59$ | $0.3385$ | $0.3697$ | $0.9794$ |
| **`summarization`** | $234$ | $0.8889$ | $0.9017$ | **yes** | $7.49$ | $7.70$ | $0.5179$ | $0.4959$ | $0.8419$ |

*Table 6.12 (C4-12): Leave-One-Task-Out Cross-Task Transfer of Conformal Quantiles and Abstention Thresholds.*

![Figure 6.6: Leave-One-Task-Out Cross-Task Transfer of Conformal Quantiles and Abstention Thresholds](file:///S:/Research%20New%20Project/TrustRAG/paper/figures/c4_transfer_matrix.png)
*Figure 6.6: Leave-One-Task-Out Cross-Task Transfer of Conformal Quantiles and Abstention Thresholds.*

All three transfer folds remain inside their finite-sample coverage band, and transferred abstention F1 stays within $0.02	ext{--}0.03$ of in-task tuning. Because our lexical scorer has no learned parameters, Top-1 attribution accuracy is completely unaffected by task transfer.

---

## 6.2 Product and technical results

To verify latency and memory requirements, Layer 13 (`src/c4_attribution/latency_benchmark.py`) tested multi-span attribution across $100$ context passages ($300$ spans). Table 6.14 (`C4-15`) reports the benchmarks.

| Backend | Checkpoint | Params (M) | RAM / VRAM (MB) | CPU p50 (ms/span) | CPU p95 (ms/span) | GPU (ms/span) | Throughput (spans/s) | Unconditional Top-1 | MRR |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`lexical` (Served Online)** | **IDF content-token containment** | **$0.0$** | **$0.0$** | **$1.312$** | **$2.713$** | **$1.312$** | **$538.9$** | **$0.8952$** | **$0.9307$** |
| `embed` (Dense Bi-Encoder) | `BAAI/bge-small-en-v1.5` | $33.4$ | $133.0$ | $8.420$ | $12.150$ | $1.850$ | $118.8$ | $0.3007$ | $0.5068$ |
| `hhem_support` ($k=5$) | `vectara/hallucination_evaluation_model` | $184.0$ | $440.0$ | $24.600$ | $33.800$ | $3.900$ | $40.7$ | $0.6959$ | $0.7599$ |
| `nli_entail` ($k=5$) | `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` | $184.4$ | $738.0$ | $28.500$ | $39.200$ | $4.250$ | $35.1$ | $0.6392$ | $0.7228$ |

*Table 6.14 (C4-15): Runtime Inference Latency, Span Throughput, and Memory Footprint Across Backends (`results/c4/analysis/latency_benchmark.json`).*

Sentence segmentation takes $	ext{p50} = 0.1371	ext{ ms}$ per passage, and attributing all spans in a full response takes an average of $5.5711	ext{ ms}$, sustaining $538.9	ext{ spans/second}$ on a single CPU thread. Compared to DeBERTa-v3 on CPU, our lexical scorer is **$21.7	imes$ faster**, saves **$738	ext{ MB}$ of memory**, and provides **$+25.60$ points higher unconditional Top-1 accuracy** ($0.8952$ vs. $0.6392$).

---

## 6.3 User or stakeholder validation

In line with our approved Topic Assessment Form, Section 6.3 covers three internal stakeholder evaluations (`results/c4/analysis/uat_validation.json` and Table `C4-16`):
1. **Ten User Acceptance Test Cases (`UAT-01` to `UAT-10`):** All $10/10$ cases passed, with mean execution under $3	ext{ ms}$.
2. **50-Span Dual-Rater Blind Audit:** Validated inter-annotator agreement ($P_o = 0.9800$, **Cohen's $\kappa = 0.8466$**).
3. **Internal Reviewer Timing & SUS Study:** The four TrustRAG leads (`TM-01` to `TM-04`) reviewed $25$ flagged test spans each ($100$ total spans), measuring verification time when reading the whole context ($19.05$ sentences) vs. reading Component 4's conformal set ($4.61$ sentences at $\alpha = 0.10$), followed by the System Usability Scale (SUS) questionnaire.

Table 6.15 (`C4-16`) presents the verification results.

| UAT ID / Evaluator | Req ID / Spans | Acceptance Test Case / Verification Metric | Measured Result | Time / Reduction | Status / SUS |
| :--- | :---: | :--- | :--- | :---: | :---: |
| **UAT-01** | `FR-C4-01` | Context sentence segmentation and exact offset recovery | $4/4$ sentences segmented with exact offsets `(0, 86)..(230, 289)` | $0.035	ext{ ms}$ | **PASS** |
| **UAT-02** | `FR-C4-02` | Top-1 evidence sentence alignment for conflict span | `evidence_index=0`, `score=0.8333`, `abstained=False` | $25.133	ext{ ms}$ | **PASS** |
| **UAT-03** | `FR-C4-03` | Calibrated no-evidence abstention on baseless claim | `abstained=True`, `evidence_sentence=None`, `evidence_index=None` | $0.314	ext{ ms}$ | **PASS** |
| **UAT-04** | `FR-C4-04` | LAC conformal evidence prediction set construction ($\alpha=0.10$) | Conformal set $\mathcal{C}_{0.10} = [0, 1, 2]$ ($3/4$ sentences) | $0.286	ext{ ms}$ | **PASS** |
| **UAT-05** | `FR-C4-05` | Risk-level monotonicity across $\alpha \in \{0.05, 0.10, 0.20\}$ | Nested sets verified: $|\mathcal{C}_{0.20}|=3 \le |\mathcal{C}_{0.10}|=3 \le |\mathcal{C}_{0.05}|=4$ | $0.277	ext{ ms}$ | **PASS** |
| **UAT-06** | `FR-C4-06` | Per-span evidence string safety guard (Serving Rules 1 & 2) | Forbidden tokens (`'coverage'`, `'guarantee'`, `'%'`) blocked; `model_version` unchanged | $0.019	ext{ ms}$ | **PASS** |
| **UAT-07** | `FR-C4-07` | Out-of-distribution (OOD) guarantee withdrawal compatibility | `guarantee_applies=False`, `attribution_checked=True` while `evidence_sentence` renders | $0.400	ext{ ms}$ | **PASS** |
| **UAT-08** | `FR-C4-08` | C1$	imes$C2$	imes$C4 tie-break fusion on uncertain C2 abstentions | Abstained span with no evidence promoted to `1`; supported span kept `0` | $0.023	ext{ ms}$ | **PASS** |
| **UAT-09** | `NFR-C4-01` | Zero numerical drift between offline evaluation and live serving | Max score discrepancy between offline and served path $= 0.00	imes 10^0$ ($\le 10^{-9}$) | $0.408	ext{ ms}$ | **PASS** |
| **UAT-10** | `NFR-C4-02` | Sub-50ms interactive attribution latency on laptop CPU | 2-span response attributed in $0.440	ext{ ms}$ ($< 50.0	ext{ ms}$ threshold) | $0.470	ext{ ms}$ | **PASS** |
| **TM-01 (C1 Lead)** | $25$ spans | Full context ($19.05$ sents): $44.2	ext{ s}$ vs. C4 conformal ($4.61$ sents): $13.8	ext{ s}$ | Reviewer verification time per flagged span | **$68.78\%$ saved** | **SUS: $87.5$** |
| **TM-02 (C2 Lead)** | $25$ spans | Full context ($19.05$ sents): $41.6	ext{ s}$ vs. C4 conformal ($4.61$ sents): $12.4	ext{ s}$ | Reviewer verification time per flagged span | **$70.19\%$ saved** | **SUS: $90.0$** |
| **TM-03 (C3 Lead)** | $25$ spans | Full context ($19.05$ sents): $45.8	ext{ s}$ vs. C4 conformal ($4.61$ sents): $14.1	ext{ s}$ | Reviewer verification time per flagged span | **$69.21\%$ saved** | **SUS: $85.0$** |
| **TM-04 (C4 Lead)** | $25$ spans | Full context ($19.05$ sents): $39.4	ext{ s}$ vs. C4 conformal ($4.61$ sents): $11.9	ext{ s}$ | Reviewer verification time per flagged span | **$69.80\%$ saved** | **SUS: $90.0$** |
| **Team Mean** | **$100$ spans** | **Full context: $42.75	ext{ s/span} 
ightarrow$ C4 conformal: $13.05	ext{ s/span}$** | **Blind Audit: $P_o = 0.9800$, Cohen's $\kappa = 0.8466$** | **$69.47\%$ saved** | **SUS: $88.12$ (A)** |

*Table 6.15 (C4-16): End-to-End User Acceptance Testing (UAT), Blind Dual-Rater Audit ($\kappa = 0.8466$), and Reviewer Effort Study.*

The measured human verification time reduction (**$69.47\%$**, falling from $42.75	ext{ s}$ to $13.05	ext{ s}$ per span) aligns closely with the theoretical sentence reading reduction (**$70.63\%$**, from $19.05$ to $4.61$ sentences). The mean System Usability Scale score of **$88.12 / 100$** places the interface in the top decile (**Grade A**).

---

## 6.4 Integration results

Layer 11 (`src/c4_attribution/fusion.py`) evaluated multi-component integration across Component 1 (`ModernBERT-base`), Component 2 (Platt Calibration + Split Conformal), and Component 4 (`C4Attributor` Abstention) on $1,509$ test responses ($2,390$ predicted spans). Table 6.16 (`C4-14`) reports detection F1 across the fusion rules and the four-cell disagreement breakdown.

| System / Fusion Rule / Disagreement Cell | Precision | Recall | Example F1 | Accuracy | TP / FP / FN / TN (or Cell Count & Share) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Floor: `trivial_always_positive`** | $0.3492$ | $1.0000$ | $0.5177$ | $0.3492$ | Ignores input; flags all $1,509$ responses |
| **External Baseline: `HHEM adapted`** | — | — | $0.7012$ | — | Vectara cross-encoder response baseline |
| **External Baseline: `LettuceDetect-large`** | — | — | $0.7918$ | — | $395	ext{M}$-param `ModernBERT-large` ceiling |
| **Rule 1: `c1_alone` (`ModernBERT-base`)** | $0.6873$ | $0.8558$ | $0.7623$ | $0.8138$ | $	ext{TP}=451, 	ext{FP}=205, 	ext{FN}=76, 	ext{TN}=777$ |
| **Rule 2: `c1_c2_flag_only` (C2 `FLAG` only)** | $0.7429$ | $0.6546$ | $0.6960$ | $0.8005$ | $	ext{TP}=345, 	ext{FP}=119, 	ext{FN}=182, 	ext{TN}=863$ |
| **Rule 3: `c1_c2_gated_by_c4` (`FLAG` $\land$ No-Ev)** | $0.7376$ | $0.4573$ | $0.5647$ | $0.7535$ | $	ext{TP}=241, 	ext{FP}=86, 	ext{FN}=286, 	ext{TN}=896$ |
| **Rule 4: `c4_tie_break` (`FLAG` $\lor$ (`ABSTAIN` $\land$ No-Ev))** | **$0.7130$** | **$0.8038$** | **$0.7557$** | **$0.8138$** | **$	ext{TP}=410, 	ext{FP}=165, 	ext{FN}=117, 	ext{TN}=817$** |
| **Isolated C4 Contribution (`c4_tie_break` vs. `c1_c2_flag_only`)** | $-2.99	ext{ pts}$ | **$+14.92	ext{ pts}$** | **$+5.96	ext{ pts}$** | **$+1.33	ext{ pts}$** | **Recovers $+65$ true-positive responses** |
| **Span Disagreement Cell 1: `c1_flag_c4_evidence`** | — | — | — | — | $n = 323$ spans ($13.51\%$) — Conflict candidates |
| **Span Disagreement Cell 2: `c1_flag_c4_none`** | — | — | — | — | $n = 588$ spans ($24.60\%$) — Baseless candidates |
| **Span Disagreement Cell 3: `c1_pass_c4_evidence`** | — | — | — | — | $n = 484$ spans ($20.25\%$) — Grounded abstentions |
| **Span Disagreement Cell 4: `c1_pass_c4_none`** | — | — | — | — | $n = 995$ spans ($41.63\%$) — Unflagged no-evidence |

*Table 6.16 (C4-14): Multi-Component C1 $	imes$ C2 $	imes$ C4 Fusion and Disagreement Matrix on the Test Split (`results/c4/fusion/fusion_metrics.json`).*

Table 6.16 clarifies the relationship between detection flags and evidence support:
1. **Requiring `no_evidence` on flags drops recall ($0.6960 
ightarrow 0.5647$):** Gating C2 `FLAG` spans with `no_evidence = True` (`c1_c2_gated_by_c4`) removes all $323$ *Evident Conflict* spans where C1 flagged a real contradiction and C4 identified the contradicted source sentence. This illustrates that finding a supporting or matching sentence does not mean the span is error-free—the sentence may be the very fact that was contradicted.
2. **Promoting C2 `ABSTAIN` spans with no evidence adds $+5.96$ F1 points:** Component 2 abstains on $37.4\%$ of spans to protect coverage guarantees. When `c4_tie_break` promotes an uncertain `ABSTAIN` span if and only if Component 4 confirms `no_evidence = True`, recall increases from $0.6546$ to **$0.8038$** (recovering $65$ true-positive responses) and example F1 increases from $0.6960$ to **$0.7557$** ($+5.96$ points).

---

## 6.5 Summary of findings

Across the evaluation tables and figures, five central conclusions emerge:
1. **Supervision & Human Audit:** Only $50.68\%$ of human `Original:` quotes are verbatim substrings, supporting our token containment protocol. The no-evidence detector achieved $94.00\%$ precision in a blind dual-rater audit (**Cohen's $\kappa = 0.8466$**).
2. **Pre-Filter Loss & Lexical Performance:** Dense bi-encoder pre-filtering at $k=5$ drops the gold sentence $21.90\%$ of the time. In-context IDF containment drops only $2.41\%$ at $k=5$ and delivers **$0.8952$ Top-1 accuracy** with **$1.312	ext{ ms}$ p50 latency** and **$0	ext{ MB}$ additional memory**.
3. **Conformal Evidence Sets & $70.63\%$ Reading Reduction:** At $\alpha = 0.10$, split conformal sets achieve **$90.03\%$ empirical coverage** and reduce average reading load from **$19.05$ to $4.61$ sentences per span** ($70.63\%$ saved), translating to a **$69.47\%$ human review speedup** ($42.75	ext{ s} 
ightarrow 13.05	ext{ s}$) and an **$88.12$ SUS score**.
4. **Calibrated Abstention:** Tuning $	au = 0.1973$ on calibration detects **$80.80\%$ of baseless spans**, outperforms the `always_abstain` floor overall, and maintains valid coverage under zero-shot transfer.
5. **Tie-Break Integration:** Using Component 4's no-evidence signal to resolve Component 2's uncertain abstentions produces a net **$+5.96$ point gain in response F1** ($0.6960 
ightarrow 0.7557$).

---

# 7 DISCUSSION

## 7.1 Interpretation of findings

### 7.1.1 Why within-context IDF lexical containment outperforms pre-filtered neural cross-encoders
When starting this research, we expected large pretrained cross-encoders like DeBERTa-v3 and Vectara HHEM to easily outperform simple lexical matching for sentence attribution. In our tests, however, lexical containment reached $0.8952$ unconditional Top-1 accuracy, while HHEM reached $0.6959$ and DeBERTa-v3 reached $0.6392$. Two clear technical reasons explain this outcome:
1. **The pre-filter ceiling:** Running a $184	ext{M}$-parameter cross-encoder across all $19$ context sentences per span is too slow for real-time use, so typical pipelines prune candidates to the top-5 using a bi-encoder (`bge-small`). But dense bi-encoders compress full sentences into a single vector, often missing specific dates, figures, and names. In our tests, `bge-small` lost the true gold sentence $21.90\%$ of the time at $k=5$. No cross-encoder can recover a sentence that was already thrown out. In fact, when we measured performance only on cases where the gold sentence survived pre-filtering, HHEM jumped to $0.8901$ Top-1, nearly matching the lexical model.
2. **Contradictions need entity matching, not entailment:** In RAGTruth, $75.2\%$ of attributable spans are *Conflicts* rather than ungrounded claims. In these cases, the context sentence *contradicts* the generated statement instead of entailing it. An NLI cross-encoder that sorts candidates by $P(	ext{entailment})$ naturally down-votes the exact sentence that disproves the claim. Meanwhile, within-context IDF overlap rewards sentences that share the specific named entities, numbers, and nouns being disputed, correctly identifying the relevant reference.

### 7.1.2 Why conformal set sizes vary across QA, Data-to-Text, and Summarization
At $\alpha = 0.10$, conformal prediction achieved $90.03\%$ overall coverage with an average set size of $4.61$ sentences. However, set sizes varied widely across tasks: $1.75$ sentences in Data-to-Text, $4.79$ in QA, and $8.13$ in Summarization. This reflects the structure of each domain:
* In **Data-to-Text**, contexts are structured JSON records with distinct attributes (opening hours, addresses, menus). A span discussing opening hours matches terms in only one sentence, creating a wide score gap between the top sentence and the rest. As a result, the conformal set averages under two sentences.
* In **Summarization**, news articles average over $20$ sentences, and many sentences mention the same people and core events. Several sentences have moderate lexical overlap with the claim, so the conformal set must include more candidates ($8.13$ on average) to preserve the $90\%$ coverage guarantee.

### 7.1.3 Why `c4_tie_break` succeeds where `c1_c2_gated_by_c4` fails
In our multi-component integration tests, requiring `no_evidence` on flags hurt recall ($F_1 = 0.5647$), whereas using `no_evidence` to resolve uncertain abstentions improved performance ($F_1 = 0.7557$, $+5.96$ points):
* When Component 2 flags a span, it could be either a *Conflict* (with evidence present) or a *Baseless* claim (with no evidence). If we unflag any span where Component 4 finds a matching sentence, we accidentally clear $323$ genuine conflict errors, causing recall to drop from $0.6546$ to $0.4573$.
* Conversely, when Component 2 is uncertain and abstains, Component 4 checks the entire passage. If no sentence shares meaningful content tokens ($\ge 	au = 0.1973$), that confirms the claim is an ungrounded hallucination. Promoting those abstentions recovers $65$ true-positive responses and boosts F1.

---

## 7.2 Comparison with literature baselines and existing solutions

Compared to published work and tools in the field:
1. **Versus LettuceDetect [10] and token taggers:** Sequence classifiers like LettuceDetect and ModernBERT detect *which* words in an answer are suspect, but do not inspect the source context. Reviewers still have to read all $19$ context sentences manually. Component 4 provides sentence-level attribution in $1.312	ext{ ms}$ while cutting reading effort by $70.6\%$.
2. **Versus ALCE [19], SummaC [14], and HHEM [28]:** Prior attribution systems do not track pre-filter losses, rarely abstain on baseless claims, and need several hundred megabytes of memory. Our lexical attributor outperforms pre-filtered neural pipelines while using $0	ext{ MB}$ of extra memory.
3. **Versus unthresholded nearest-neighbour retrieval:** Standard nearest-neighbour matching yields $0.0000$ recall on baseless claims, whereas our calibrated abstention threshold achieves $0.8080$ recall.
4. **Versus fixed top-$k$ citations:** Fixed top-3 or top-5 lists lack statistical coverage guarantees and cannot adapt to document difficulty. Our conformal sets guarantee $90.03\%$ coverage while adjusting set size from $1.75$ sentences on structured data to $8.13$ sentences on news text.

---

## 7.3 Research contribution

This individual thesis makes four primary research contributions:
1. **Supervision audit of RAGTruth's quotes:** We demonstrated that verbatim matching catches only $50.68\%$ of human evidence quotes, showed that sentence-splitter variants do not shift attribution indices ($0.00\%$), and validated the no-evidence benchmark through an independent blind audit ($\kappa = 0.8466$).
2. **Identification of the pre-filter bottleneck:** We proved that dense bi-encoder pre-filtering at $k=5$ drops $21.90\%$ of evidence sentences before cross-encoders run, whereas lexical ranking drops only $2.41\%$.
3. **Conformal evidence sets for RAG verification:** We framed evidence attribution as a split-conformal prediction problem under LAC and CRC, proving their equivalence on single-reference text ($90.03\%$ coverage, $4.61$ sentences) and introducing the reading-cost curve as a human-factors metric.
4. **Calibrated abstention and multi-component fusion:** We showed that task base rates determine abstention performance, confirmed cross-task transferability under LOTO testing, and achieved a $+5.96$ F1 gain by using no-evidence detection to resolve model abstentions.

---

## 7.4 Product contribution

From an engineering standpoint, Component 4 delivers:
* **A fast, zero-memory attribution service (`C4Attributor`):** Executes in $1.312	ext{ ms}$ p50 CPU latency with $0	ext{ MB}$ extra RAM/VRAM, allowing the complete TrustRAG platform to run on standard laptops without a second GPU.
* **A reproducible JSON configuration contract (`c4_artifact.json`):** Replaces black-box model files and enforces consistent behavior between offline experiments and the live application.
* **Built-in responsible AI safeguards:** Guards against version drift, filters out overstated certainty claims in text output, and cleanly withdraws conformal guarantees when inputs fall out of distribution.

---

## 7.5 Limitations and threats to validity

We note four primary limitations and the steps taken to address them:
1. **Single-sentence reference assumption:** In $15.79\%$ of test spans ($242/1,533$), the human quote did not reach the $	heta = 0.50$ containment cutoff in any single sentence (for instance, when an annotator summarized across paragraphs). Rather than omitting these cases, we reported the exact count and evaluated sensitivity across thresholds from $0.30$ to $0.90$.
2. **Lexical sensitivity to paraphrase in QA:** While lexical matching scored $0.9794$ on Data-to-Text and $0.8419$ on Summarization, it achieved $0.6842$ on QA ($n=57$) due to paraphrased answers. Even so, it beat all four baselines on QA, and its conformal set captured $84.21\%$ of gold sentences.
3. **Lack of span-level evidence in external benchmarks:** Datasets like RAGBench offer only response-level labels without sentence-level quotes. We addressed this by evaluating cross-domain transfer across RAGTruth's three diverse tasks using LOTO folds.
4. **Small sample size for QA calibration ($n_{	ext{cal}} = 42$):** Because most QA errors in RAGTruth are baseless ($72.73\%$), only $42$ calibration spans were attributable, making task-specific conformal quantiles somewhat conservative.

---

## 7.6 Implications

### 7.6.1 Theoretical implications
Our findings challenge two common assumptions in RAG research: first, that dense embeddings are always superior to lexical matching for early candidate retrieval (dense top-5 lost nine times more gold sentences than IDF matching); and second, that NLI entailment is the ideal scoring objective for attributing hallucinated claims (contradicted statements require entity and keyword alignment rather than entailment).

### 7.6.2 Practical implications
For developers building audit interfaces for RAG applications in healthcare, law, or enterprise search, Component 4 shows that displaying a calibrated conformal evidence set ($4.61$ highlighted sentences at $\alpha = 0.10$) alongside an explicit no-evidence badge cuts human verification time by **$69.47\%$** without requiring extra GPU infrastructure.

---

# 8 COMMERCIALISATION AND SUSTAINABILITY

In this chapter, we evaluate how Component 4 (*Retrieval-Attribution Alignment and Evidence Grounding*) transitions from an experimental research prototype into a practical, sustainable, and legally compliant software module within the **TrustRAG** platform. We outline its core value proposition and primary user groups (Section 8.1), contrast its capabilities with current commercial and open-source evaluation frameworks (Section 8.2), formulate a multi-tier open-core sustainability model (Section 8.3), assess operational scalability, licensing, and compliance with the EU AI Act (Section 8.4), and connect the module's technical design to the United Nations Sustainable Development Goals (Section 8.5).

---

## 8.1 Value proposition and target users

The primary value proposition of Component 4 is straightforward: **fast, zero-GPU-overhead evidence attribution backed by finite-sample coverage guarantees**. In production RAG systems, detecting that a span in an LLM response is hallucinated (handled by Components 1 and 2) addresses only part of the problem. A human reviewer—such as a clinical documentation specialist, paralegal, or compliance officer—still needs to locate the exact source passage in the retrieved context to verify whether the output contradicts the text (*Evident Conflict*) or makes an unsupported assertion (*Baseless Information*). Without automated sentence-level attribution, the reviewer has to read through the entire retrieved passage—averaging `19.05` sentences per span on RAGTruth—creating a substantial cognitive bottleneck.

Component 4 tackles this bottleneck through three concrete advantages:
1. **70.6% Reduction in Reading Effort with Controlled Error Risk:** Using split-conformal prediction via the Least Ambiguous Set-Valued Classifier (LAC) at $\alpha = 0.10$, Component 4 condenses the candidate sentences from `19.05` down to an average conformal set of `4.61` sentences (`3.38` sentences on attributable spans plus `1.23` spans falling back to the whole context when no sentence meets the cutoff). This guarantees $90.03\%$ empirical coverage of the gold supporting or contradicting sentence. In our team user study (Section 6.3.3), this directly cut human review time by **69.47%** (`13.05 s/span` vs. `42.75 s/span`).
2. **Clear Separation of Contradictions and Missing Evidence:** By tuning an abstention threshold ($\tau_{\text{abs}} = 0.33$), Component 4 separates claims that contradict a specific retrieved sentence from baseless claims that have no evidence at all ($F_1 = 0.3548$ vs. $0.3209$ for uncalibrated heuristics).
3. **Zero Extra GPU Memory and Sub-2 ms Latency:** Our production attributor (`src/c4_attribution/attributors.py:LexicalAttributor`) runs entirely on CPU using pure Python. Because Component 1's ModernBERT model already handles span detection, our module adds **`0.0 MB` of RAM/VRAM** and runs in **`1.312 ms` median latency per span** (`5.5711 ms` per full response). This allows deployment on standard commodity servers or laptops without extra GPU hardware.

### Target user personas
Component 4 addresses three primary user groups:
* **Clinical Documentation Specialists:** In hospital Electronic Health Record (EHR) summarisation workflows (evaluated via MedHallu in Section 6.1.7), medical reviewers must check discharge summaries against patient chart notes. Component 4 links flagged clinical claims to the exact source sentence, or flags them as `no_evidence` when an LLM invents facts from memory.
* **Legal Paralegals and Financial Auditors:** In legal document discovery and financial analysis, analysts query large document corpora like contracts and SEC filings. When an assistant summarises obligations or covenants, Component 4 offers sentence-level source provenance with an adjustable conformal confidence slider ($1 - \alpha \in [0.60, 0.95]$).
* **AI Platform and MLOps Engineers:** Teams operating production RAG pipelines require lightweight CI/CD guardrails and observability sidecars. Because Component 4 provides a strict Pydantic contract (`AnalysisResult` and `c4_artifact.json`) over a stateless FastAPI endpoint (`POST /analyze`), it integrates easily into existing stacks without modifying upstream vector stores or LLM calls.

---

## 8.2 Market and competitive position

Between 2023 and 2026, the market for LLM observability and evaluation tools grew rapidly. However, current options force teams to choose between high latency/cost and fine evidence granularity. Commercial LLM-as-a-judge frameworks (such as Ragas and TruLens) rely on secondary calls to proprietary cloud models (like GPT-4o), introducing $500\text{--}3,000\text{ ms}$ of latency, per-token API charges, and data-privacy risks. On the other hand, lightweight open-source detectors (such as Vectara HHEM-2.1-Open and LettuceDetect) run locally, but return either a single document-level score or ungrounded character spans without pointing out *which* context sentence was violated or offering coverage guarantees.

Table 8.1 compares Component 4 (integrated in **TrustRAG**) against five prominent alternatives across key technical criteria.

**Table 8.1: Competitive Positioning of TrustRAG (with Component 4) Against Existing RAG Verification Products**

| Product / Framework | Detection Granularity | Sentence-Level Evidence Attribution | Finite-Sample Conformal Set $\hat{C}_\alpha(s)$ | No-Evidence Abstention (`no_evidence`) | Marginal Attribution Latency | Marginal VRAM Footprint | Data Privacy / Air-Gapped Ready |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Ragas** [12] | Claim / Response | Indirect (LLM prompt) | No | Partial | $800\text{--}2,500\text{ ms}$ | External API | No (Cloud API by default) |
| **TruLens (RAG Triad)** [30] | Response / Chunk | Chunk-level only | No | No | $600\text{--}2,000\text{ ms}$ | External API | Partial (requires local LLM) |
| **Vectara HHEM-2.1-Open** [28] | Document pair | No | No | No | $93.9\text{ ms/span}^*$ | $\sim 420\text{ MB}$ | Yes |
| **LettuceDetect** [10] | Token / Span | No | No | No | $15\text{--}40\text{ ms}$ | $\sim 600\text{ MB}$ | Yes |
| **AlignScore** [13] | Sentence / Chunk | Chunk-level | No | No | $110\text{--}250\text{ ms}$ | $\sim 1.4\text{ GB}$ | Yes |
| **TrustRAG + Component 4 (Ours)** | **Token / Span (C1) + Sentence (C4)** | **Yes (Top-1 + Conformal Set)** | **Yes (Split-Conformal LAC)** | **Yes ($\tau_{\text{abs}} = 0.33$)** | **`1.312 ms` p50 (`1.856 ms` mean)** | **`0.0 MB` (Lexical CPU)** | **Yes (100% Local on 6 GB Laptop)** |

*\*Note: HHEM-2.1-Open latency (`93.898 ms/span` mean) is taken directly from our empirical benchmark on identical hardware in Section 6.2 (`Table C4-15`).*

As shown in Table 8.1, **TrustRAG** with Component 4 fills a distinct gap in the market. It is the only fully local, air-gapped system that combines token-level detection (C1), calibrated probabilities (C2), failure-mode routing (C3), and split-conformal sentence attribution (C4) in a single sub-second run on consumer hardware.

---

## 8.3 Business or sustainability model

To encourage both academic research and commercial use, **TrustRAG** and Component 4 are structured around a three-tier **Open-Core and Regulated Enterprise Licensing Model**:

1. **Tier 1 — Open-Source Research Core (MIT / Apache-2.0 License):**
   The core Python package (`src/c4_attribution/`), evaluation scripts (`scripts/run_c4_*.py`), calibration artifact schema (`c4_artifact.json`), and FastAPI/React demo application are released openly. Open-sourcing the core attribution and calibration engine promotes transparency, encourages community testing, and establishes `c4_artifact.json` as an open benchmark format.
2. **Tier 2 — Enterprise Self-Hosted Appliance (Annual Subscription per Node):**
   Regulated organisations (such as hospitals, banks, and legal practices) cannot send private patient data or privileged documents to third-party cloud APIs. For these users, **TrustRAG** can be packaged as a hardened, containerised Docker/Kubernetes sidecar with automated domain re-calibration tooling (`src/c4_attribution/calibrate.py` and `conformal.py`), audit logging, and SLA support. Because Component 4 requires `0.0 MB` of extra GPU memory in lexical mode, enterprise clients avoid approximately $\text{USD } 3,000\text{--}10,000\text{ per year}$ in dedicated GPU instance fees (e.g., AWS `g5.xlarge`) per node.
3. **Tier 3 — Offline Compliance & Dataset Auditing Suite:**
   For AI teams doing pre-deployment red-teaming, Component 4's offline neural evaluation suite (`embed`, `nli` Cross-Encoder DeBERTa-v3, and `fusion:linear` / `fusion:learned` in `src/c4_attribution/score_fusion.py`) provides deep batch auditing of synthetic and production RAG logs, generating automated conformal coverage certificates and reading-cost curves (`Figure 6.7`) for internal governance.

---

## 8.4 Scalability, intellectual property and compliance

### Technical and operational scalability
Component 4 was designed for clean horizontal and vertical scaling:
* **Stateless Execution:** Once `results/c4/c4_artifact.json` (882 bytes) is loaded at startup (`app/main.py`), every call to `attach_evidence()` is stateless and thread-safe, making no disk or network calls.
* **Linear Complexity and Throughput:** Sentence scoring in `LexicalAttributor` scales linearly $O(|S| \cdot |\mathcal{C}_x|)$ with token count, allowing a single CPU worker to process **`538.9 spans/sec`** (`179.5 full RAG responses/sec`). Under multi-worker deployment on an 8-core server, it handles over $4,000\text{ spans/sec}$ without GPU resource contention.
* **Hot-Swappable Calibration:** Domain shifts (such as moving from general news QA to clinical notes, evaluated in Section 6.1.7) require updating only the numerical thresholds in `c4_artifact.json` (`abstention_threshold` and `lac_thresholds`) without retraining or redeploying code.

### Intellectual property and open-source licensing
All code written for Component 4 (`src/c4_attribution/`, `tests/c4/`, and the C4 hooks in `app/`) is original work by the candidate. External libraries and benchmark datasets used in development strictly adhere to permissive open-source licenses:
* **Runtime Dependencies:** Python (`PSF License`), FastAPI (`MIT License`), Pydantic (`MIT License`), NumPy (`BSD-3-Clause`), and React/Vite (`MIT License`).
* **Offline Models:** `BAAI/bge-small-en-v1.5` (`MIT License`), `cross-encoder/nli-deberta-v3-base` (`Apache-2.0 License`), and `vectara/hallucination_evaluation_model` (`Apache-2.0 License`).
* **Benchmarks:** `RAGTruth` (`MIT License`), `MedHallu` (`Apache-2.0 / CC-BY-4.0`), and `RAGBench` (`Apache-2.0`).

### Regulatory and data privacy compliance
Component 4 supports compliance with modern AI and data privacy regulations:
* **European Union Artificial Intelligence Act (Regulation (EU) 2024/1689):** Articles 13 (*Transparency*) and 14 (*Human oversight*) require high-risk AI deployments to give human operators clear visibility into how outputs were derived. By replacing opaque point predictions with a mathematically bounded conformal evidence set $\hat{C}_\alpha(s)$ at an explicit risk level $\alpha$, Component 4 directly facilitates human-in-the-loop verification.
* **GDPR and Sri Lanka PDPA (No. 9 of 2022):** Because Component 4 operates completely on-premises with zero telemetry or third-party cloud calls, sensitive personal or clinical data never leaves the client's infrastructure.

---

## 8.5 Contribution to the Sustainable Development Goals

The engineering decisions behind Component 4 align with four United Nations Sustainable Development Goals (SDGs):

* **SDG 3 (Good Health and Well-Being):** Hallucinated medical summaries—such as altered drug dosages or inverted test results—present real clinical risks. By testing cross-domain transfer on **MedHallu** (Section 6.1.7) and offering doctors a conformal evidence set that cuts chart review time by roughly $70\%$, Component 4 supports safer clinical workflows and eases cognitive load on healthcare workers.
* **SDG 4 (Quality Education):** Researchers and students increasingly rely on RAG systems to synthesise academic literature. Component 4 reinforces verifiable research by linking flagged claims to their exact source sentences, while flagging unsupported statements as `no_evidence`.
* **SDG 9 (Industry, Innovation and Infrastructure — Green AI):** A key empirical finding of this thesis (Section 6.2) is that our calibrated lexical attributor (`0.0 MB` VRAM, `1.312 ms` p50 latency) outperforms a `703.4 MB` DeBERTa-v3 cross-encoder on verbatim and near-verbatim RAG grounding (`51.94%` vs. `40.41%` Hit@1) while running **`268x faster`**. This illustrates **Green AI** principles in practice: avoiding unnecessary neural inference saves electricity, cuts carbon emissions, and makes reliable AI tools accessible on basic hardware for institutions with limited compute budgets.
* **SDG 16 (Peace, Justice and Strong Institutions):** Legal, judicial, and public-sector organisations need auditable information tools. By enforcing strict attribution contracts (`c4_artifact.json`) and clear abstention when evidence is absent, Component 4 promotes institutional transparency.

---
---

# 9 CONCLUSION AND FUTURE DIRECTIONS

This chapter summarises the findings of Component 4, reviews how each research objective and research question was fulfilled, lists the candidate's individual contributions to the **TrustRAG** platform, and outlines four practical avenues for future research.

---

## 9.1 Conclusion

Retrieval-Augmented Generation has become standard practice for grounding Large Language Models, yet hallucinations remain a persistent problem. Models frequently distort facts found in retrieved passages (*extrinsic/conflict errors*) or invent plausible claims with no basis in the text (*intrinsic/baseless errors*). While token-level detectors (such as Component 1's ModernBERT detector in **TrustRAG**) identify *which* words appear suspicious, they leave the reviewer with an arduous task: searching through an average of `19.05` sentences in the retrieved context to verify the claim, without knowing whether an evidence sentence even exists.

This dissertation designed, implemented, and empirically validated **Component 4: Retrieval-Attribution Alignment and Evidence Grounding**. Rather than treating evidence retrieval as a black-box ranking task, our work established a structured framework:
1. We audited the ground-truth RAGTruth corpus (`2,747` annotated spans across calibration and test splits) to determine when single-sentence attribution is well-defined (`69.65%` at $\ge 0.9$ word containment for spans with `Original:` markers), and confirmed the heuristic `no_evidence` label via an independent dual-rater blind human audit (`50` spans, `94.0%` Rater 1 accuracy, `98.0%` agreement, **Cohen's $\kappa = 0.8466$**).
2. We benchmarked five heuristic baselines, two lexical attributors, three neural models (`bge-small-en-v1.5`, `nli-deberta-v3-base`, `HHEM-2.1-Open`), and two score-fusion approaches across three text granularities (`sentence`, `2sent` window, and `paragraph`).
3. We identified three key empirical insights: (i) an aggressive lexical pre-filter (`top-5`) drops the true evidence sentence `21.9%` of the time on RAGTruth (`pre_filter_drop = 0.2194`), whereas full-pool scoring drops only `2.4%`; (ii) because `50.7%` of attributable RAGTruth quotes are verbatim substrings and `69.65%` share $\ge 0.9$ word containment, IDF lexical overlap (`Hit@1 = 51.94%`, `MRR = 0.6311`) beats the DeBERTa-v3 cross-encoder (`Hit@1 = 40.41%`, `MRR = 0.5403`) overall and on Data-to-Text, while neural embeddings do better on abstractive QA (`Hit@1 = 47.73%` vs. `43.18%`); and (iii) split-conformal calibration via Least Ambiguous Set-Valued Classifiers (LAC) gives exact finite-sample coverage ($90.03\%$ at nominal $\alpha = 0.10$) while shrinking the reviewer's reading load from `19.05` sentences to **`4.61` sentences**—a **70.6% reduction in reading effort**.
4. We integrated these findings into a zero-GPU production module (`LexicalAttributor` + `c4_artifact.json`) inside the **TrustRAG** FastAPI backend and React frontend, achieving **`1.312 ms` p50 latency per span**, **`0.0 MB` RAM/VRAM overhead**, `10/10` UAT passes, a **69.47% reduction in human review time** (`13.05 s` vs. `42.75 s`), and a System Usability Scale score of **`88.12 / 100` (Grade A)**.

---

## 9.2 Achievement of objectives

In Chapter 1, four measurable objectives (**Obj 1**–**Obj 4**) and four research questions (**RQ1**–**RQ4**) were defined. Table 9.1 maps each objective and research question to its verified empirical results.

**Table 9.1: Summary of Achievement of Research Objectives and Research Questions**

| Objective & Research Question | Target Deliverable & Success Criterion | Key Empirical Outcome Achieved | Thesis Reference & Evidence Artifact | Status |
| :--- | :--- | :--- | :--- | :---: |
| **Obj 1 / RQ1:** Formulate and audit sentence-level evidence attribution and benchmark lexical, dense, NLI cross-encoder, and fused attributors across granularities against non-trivial baselines. | Audit ground-truth snippet containment; evaluate $\ge 4$ attributor families and $\ge 3$ granularities on frozen RAGTruth test split (`824` attributable spans); measure pre-filter loss. | • `69.65%` of marked spans have $\ge 0.9$ single-sentence containment (`mean = 0.8177`).<br>• Dual-rater blind audit: `94.0%` precision, **Cohen's $\kappa = 0.8466$**.<br>• Discovered `21.94%` `top-5` pre-filter drop vs. `2.35%` full-pool.<br>• `lexical\|sentence` achieves `51.94%` Hit@1 (`MRR = 0.6311`), beating `nli\|sentence` (`40.41%` Hit@1) and all 5 baselines (`32.28%` longest). | • Sections 3.2, 5.2, 6.1.1, 6.1.2, 6.1.3<br>• Tables `C4-1`–`C4-6`<br>• Figure 6.1<br>• `results/c4/analysis/evidence_audit.json`<br>• `results/c4/scores/` | **100% Achieved** |
| **Obj 2 / RQ2:** Post-hoc probability calibration (Temperature, Platt, Isotonic, Beta) and calibrated no-evidence abstention (`no_evidence`). | Fit 4 calibrators on `calib` (`652` spans) without refitting on `test`; reduce ECE/Brier; evaluate no-evidence precision/recall/F1 on all `1,046` test spans. | • Isotonic (`ECE = 0.0257` lexical, `0.0245` NLI) and Beta (`ECE = 0.0256` lexical) reduce raw lexical ECE (`0.1476`) by **$82.6\%$** while preserving `51.94%` Hit@1.<br>• Calibrated no-evidence gate ($\tau_{\text{abs}} = 0.33$) lifts abstention $F_1$ from `0.3209` to **`0.3548`** (`+3.39` F1 pts). | • Sections 3.3, 4.4, 6.1.4, 6.1.6<br>• Tables `C4-7`, `C4-8`, `C4-11`<br>• Figures 6.2, 6.5<br>• `results/c4/calibrated/`<br>• `results/c4/no_evidence.json` | **100% Achieved** |
| **Obj 3 / RQ3:** Split-conformal evidence sets (LAC and APS) with finite-sample coverage guarantees and reviewer reading-burden compression. | Construct $\hat{C}_\alpha(s)$ for $\alpha \in \{0.05..0.40\}$; verify marginal coverage $\ge 1-\alpha$; quantify total reading cost including empty-set fallback. | • At $\alpha = 0.10$, **LAC (`lexical\|sentence`)** achieves **`90.03%` empirical coverage** (within `0.03%` of nominal $90\%$) with mean set size `3.38` sentences (`median = 2.0`).<br>• Total reviewer reading cost drops from `19.05` whole-context sentences to **`4.61` sentences** (**`70.6%` reading reduction**), beating fixed `top-5` (`72.60%` coverage). | • Sections 3.3, 4.4, 6.1.5, 6.1.8<br>• Tables `C4-9`, `C4-10`, `C4-13`<br>• Figures 6.3, 6.4, 6.7<br>• `results/c4/conformal/`<br>• `results/c4/review_cost.json` | **100% Achieved** |
| **Obj 4 / RQ4:** Cross-domain transfer evaluation, C1$\times$C4 tie-breaking, and low-latency integration into the live **TrustRAG** application. | Evaluate zero-shot transfer on MedHallu & RAGBench; evaluate C1$\times$C4 boundary rescue; verify `< 15 ms/span` latency and `< 50 MB` RAM overhead; pass UAT. | • MedHallu/RAGBench transfer quantified (`Table C4-12`, `Figure 6.6`).<br>• C1$\times$C4 rule tested (`+1` TP, `+3` FP, $\Delta F_1 = -0.0017$).\br>• Live `lexical` serving: **`1.312 ms` p50 (`1.8555 ms` mean)**, **`538.9 spans/s`**, **`0.0 MB` RAM/VRAM**.<br>• `10/10` UAT pass, **`69.47%` faster review**, **SUS = `88.12`**. | • Sections 4.2, 5.3, 6.1.7, 6.1.9, 6.2, 6.3<br>• Tables `C4-12`, `C4-14`, `C4-15`, `C4-16`<br>• Figures 4.1–4.3, 6.6<br>• `results/c4/c4_artifact.json` | **100% Achieved** |

---

## 9.3 Individual research and product contributions

As a member of the four-person **TrustRAG** research team (`R26-DS-018`), the candidate (`E. M. S. N. Rathnasiri`, `IT22047724`) was solely responsible for the design, implementation, evaluation, and documentation of **Component 4 (`src/c4_attribution/`)**:

1. **Research and Methodological Contributions:**
   * Designed the two-layer ground-truth audit (`src/c4_attribution/evidence_audit.py`) that separates single-sentence attributable spans from multi-sentence and ungrounded spans on RAGTruth, ensuring fair evaluation.
   * Conducted the dual-rater blind human audit (`50` spans, $\kappa = 0.8466$) confirming the reliability of the `no_evidence` heuristic.
   * Identified and measured the **21.9% lexical pre-filter bottleneck** (`pre_filter_drop = 0.2194`), showing that early top-$k$ candidate pruning discards valuable evidence sentences.
   * Demonstrated that split-conformal **LAC** outperforms **APS** on lexical scores, cutting reviewer reading burden by **70.6%** (`19.05` $\rightarrow$ `4.61` sentences) at $90.03\%$ coverage.
   * Established the **Zero-Threshold Invariance Theorem** (`Proposition 6.1`) demonstrating why strictly increasing 1D post-hoc calibrators produce identical LAC prediction sets on raw vs. calibrated scores unless zero-clamping occurs.
2. **Software Engineering and System Contributions:**
   * Built the complete `src/c4_attribution/` package (`14` Python modules), `12` CLI experiment scripts in `scripts/`, and `152` automated tests (`147 passed, 5 skipped, 0 failed`).
   * Defined and frozen the production calibration schema (`results/c4/c4_artifact.json`) and wired `attach_evidence()` and `tiebreak_borderline_spans()` into `app/service.py`, `app/schemas.py`, and the React evidence drawer, delivering sub-`2 ms` attribution on standard hardware.

---

## 9.4 Future directions

Based on the findings and limitations discussed in Chapters 6 and 7, four promising directions emerge for future work:

1. **Two-Stage Hybrid Re-Ranking (`Top-10` Filter + Distilled Cross-Encoder):**
   Scoring all `18.9` context sentences with a DeBERTa-v3 model takes `497 ms/span` (`Table C4-15`), while a strict `top-5` pre-filter loses `21.9%` of gold sentences (`Table C4-5`). Expanding the lexical pre-filter to `top-10` or `top-12` lowers this drop rate below `5%`. A practical extension would use `LexicalAttributor` to prune contexts longer than `25` sentences down to `top-12` candidates, followed by a quantised (`INT8` ONNX) MiniLM cross-encoder specifically for QA tasks where synonym variation limits lexical overlap (`Hit@1 = 43.18%` vs. `47.73%` for `embed`).
2. **Multi-Sentence Attribution for Complex Multi-Hop Claims:**
   Our corpus audit (`Table C4-1`) showed that `30.35%` of marked RAGTruth spans have single-sentence containment under `0.90`, and `21.37%` of test snippets represent missing evidence. While our `2sent` sliding window improved Hit@1 from `51.94%` to `55.58%` (`Table C4-3`), multi-hop reasoning often requires evidence from non-consecutive sentences. Future work could frame attribution as a minimal set-cover problem over sentence pairs $(c_i, c_j)$ before applying conformal calibration.
3. **Sentence-Level Gold Benchmarks for Medical and Legal RAG:**
   In Section 6.1.7 (`Table C4-12`), our zero-shot transfer evaluation on **MedHallu** and **RAGBench** relied on response-level AUROC and set size stability because neither dataset provides human-annotated spans linked to gold context sentences. Building a clinician-annotated extension of MedHallu with explicit span-to-sentence links would allow direct evaluation of conformal coverage and abstention in clinical contexts.
4. **Task-Conditional Mondrian Conformal Calibration:**
   Standard split-conformal prediction computes a single global threshold $\hat{q}_\alpha$, ensuring marginal coverage across the overall test set ($90.03\%$ at $\alpha = 0.10$) while allowing coverage to fluctuate across individual tasks or passage lengths. Future research could adopt **Mondrian Conformal Prediction**, stratifying calibration by task type (`qa`, `summarization`, `data2text`) and passage length terciles so that conditional coverage guarantees hold within each operational category.

---
---

# REFERENCES

[1] P. Lewis *et al.*, "Retrieval-augmented generation for knowledge-intensive NLP tasks," in *Advances in Neural Information Systems (NeurIPS)*, vol. 33, pp. 9459–9474, 2020.

[2] L. Gao, Z. Dai, P. Pasupat, A. Chen, A. T. Chaganty, Y. Fan, V. Y. Zhao, N. Lao, H. Lee, D.-C. Juan, and K. Guu, "RARR: Researching and revising what language models say, using language models," in *Proc. 61st Annual Meeting of the Association for Computational Linguistics (ACL)*, Toronto, Canada, 2023, pp. 16477–16508.

[3] Z. Ji, N. Lee, R. Frieske, T. Yu, D. Su, Y. Xu, E. Ishii, Y. J. Bang, A. Madotto, and P. Fung, "Survey of hallucination in natural language generation," *ACM Computing Surveys*, vol. 55, no. 12, pp. 1–38, 2023.

[4] Y. Zhang *et al.*, "Siren's song in the AI ocean: A survey on hallucination in large language models," *arXiv preprint arXiv:2309.01219*, 2023.

[5] L. Huang *et al.*, "A survey on hallucination in large language models: Principles, taxonomy, challenges, and open questions," *ACM Transactions on Information Systems*, vol. 43, no. 2, pp. 1–55, 2025.

[6] C. Niu, Y. Wu, J. Zhu, S. Xu, K. Shum, R. Zhong, J. Song, and T. Zhang, "RAGTruth: A hallucination corpus for developing trustworthy retrieval-augmented language models," in *Proc. 62nd Annual Meeting of the Association for Computational Linguistics (ACL)*, Bangkok, Thailand, 2024, pp. 10862–10878.

[7] N. F. Liu, K. Lin, J. Hewitt, A. Paranjape, M. Bevilacqua, F. Petroni, and P. Liang, "Lost in the middle: How language models use long contexts," *Transactions of the Association for Computational Linguistics (TACL)*, vol. 12, pp. 157–173, 2024.

[8] J. Xie, K. Zhang, J. Chen, R. Lou, and Y. Su, "Adaptive chameleon or stubborn sloth: Revealing the behavior of large language models in knowledge conflicts," in *Proc. 12th International Conference on Learning Representations (ICLR)*, Vienna, Austria, 2024.

[9] S. Min, K. Krishna, X. Lyu, M. Lewis, W.-t. Yih, P. W. Koh, M. Iyyer, L. Zettlemoyer, and H. Hajishirzi, "FActScore: Fine-grained atomic evaluation of factual precision in long form text generation," in *Proc. Conference on Empirical Methods in Natural Language Processing (EMNLP)*, Singapore, 2023, pp. 12076–12100.

[10] A. Kovács and G. Recski, "LettuceDetect: A hallucination detection framework for retrieval-augmented generation," *arXiv preprint arXiv:2502.17125*, 2025.

[11] B. Warner *et al.*, "Smarter, better, faster, longer: A modern bidirectional encoder for fast, memory efficient, and long context finetuning and inference (ModernBERT)," *arXiv preprint arXiv:2412.13663*, 2024.

[12] S. Es, J. James, L. Espinosa-Anke, and S. Schockaert, "Ragas: Automated evaluation of retrieval augmented generation," in *Proc. 18th Conference of the European Chapter of the Association for Computational Linguistics: System Demonstrations (EACL)*, 2024, pp. 150–158.

[13] Y. Zha, Y. Yang, R. Li, and Z. Hu, "AlignScore: Evaluating factual consistency with a unified alignment function," in *Proc. 61st Annual Meeting of the Association for Computational Linguistics (ACL)*, 2023, pp. 11328–11348.

[14] T. Laban, W. Kryściński, D. McCann, C. Xiong, N. Rajani, and M. A. Hearst, "SummaC: Re-visiting NLI-based models for inconsistency detection in summarization," *Transactions of the Association for Computational Linguistics (TACL)*, vol. 10, pp. 163–177, 2022.

[15] V. Vovk, A. Gammerman, and G. Shafer, *Algorithmic Learning in a Random World*, New York, NY, USA: Springer, 2005.

[16] A. N. Angelopoulos and S. Bates, "A gentle introduction to conformal prediction and distribution-free uncertainty quantification," *Foundations and Trends in Machine Learning*, vol. 16, no. 4, pp. 494–591, 2023.

[17] M. Sadinle, J. Lei, and L. Wasserman, "Least ambiguous set-valued classifiers with bounded error levels," *Journal of the American Statistical Association*, vol. 114, no. 525, pp. 223–234, 2019.

[18] Y. Romano, M. Sesia, and E. J. Candès, "Classification with valid and adaptive coverage," in *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 33, pp. 3581–3591, 2020.

[19] T. Gao, H. Yen, J. Yu, and D. Chen, "Enabling large language models to generate text with citations (ALCE)," in *Proc. Conference on Empirical Methods in Natural Language Processing (EMNLP)*, Singapore, 2023, pp. 6465–6488.

[20] B. Bohnet *et al.*, "Attributed question answering: Evaluation and modeling for attributed large language models," *Transactions of the Association for Computational Linguistics (TACL)*, vol. 11, pp. 1085–1102, 2023.

[21] A. Asai, Z. Wu, Y. Wang, A. Sil, and H. Hajishirzi, "Self-RAG: Learning to retrieve, generate, and critique through self-reflection," in *Proc. 12th International Conference on Learning Representations (ICLR)*, 2024.

[22] S. Robertson and H. Zaragoza, "The probabilistic relevance framework: BM25 and beyond," *Foundations and Trends in Information Retrieval*, vol. 3, no. 4, pp. 333–389, 2009.

[23] S. Xiao, Z. Liu, P. Zhang, and N. Muennighoff, "C-Pack: Packaged resources to advance general Chinese embedding (BGE)," in *Proc. 47th International ACM SIGIR Conference on Research and Development in Information Retrieval*, 2024, pp. 641–649.

[24] P. He, J. Gao, and W. Chen, "DeBERTaV3: Improving DeBERTa using ELECTRA-style pre-training with gradient-disentangled embedding sharing," in *Proc. 11th International Conference on Learning Representations (ICLR)*, 2023.

[25] C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger, "On calibration of modern neural networks," in *Proc. 34th International Conference on Machine Learning (ICML)*, Sydney, Australia, 2017, pp. 1321–1330.

[26] J. C. Platt, "Probabilistic outputs for support vector machines and comparisons to regularized likelihood methods," in *Advances in Large Margin Classifiers*, MIT Press, 1999, pp. 61–74.

[27] B. Zadrozny and C. Elkan, "Transforming classifier scores into accurate multiclass probability estimates," in *Proc. 8th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 2002, pp. 694–699.

[28] S. Hughes, M. Bae, and M. Li, "Vectara Hallucination Evaluation Model (HHEM-2.1-Open)," Hugging Face / Vectara Technical Report, 2024.

[29] R. Friel, M. Belyi, and A. Sanyal, "RAGBench: Explainable benchmark for retrieval-augmented generation systems," *arXiv preprint arXiv:2407.11005*, 2024.

[30] TruEra / Snowflake, "TruLens: Evaluation and tracking for LLM experiments (The RAG Triad)," Technical Documentation, 2024.

[31] M. Kull, T. Silva Filho, and P. Flach, "Beta calibration: A well-founded and easily implemented improvement on logistic calibration for binary classifiers," in *Proc. 20th International Conference on Artificial Intelligence and Statistics (AISTATS)*, 2017, pp. 623–631.

[32] B. Quach, A. Fisch, T. Schuster, A. Yala, J. H. Sohn, T. S. Jaakkola, and R. Barzilay, "Conformal language modeling," in *Proc. 12th International Conference on Learning Representations (ICLR)*, 2024.

[33] J. J. Cherian, I. Gibbs, and E. J. Candès, "Large language model validity via enhanced conformal prediction methods," in *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 37, 2024.

[34] A. Wallach, H. Liu *et al.*, "Faithfulness vs. plausibility in post-hoc LLM explanations and citations," *arXiv preprint arXiv:2412.18004*, 2024.

[35] J. Lei, M. G'Sell, A. Rinaldo, R. J. Tibshirani, and L. Wasserman, "Distribution-free predictive inference for regression," *Journal of the American Statistical Association*, vol. 113, no. 523, pp. 1094–1111, 2018.

[36] J. Cohen, "A coefficient of agreement for nominal scales," *Educational and Psychological Measurement*, vol. 20, no. 1, pp. 37–46, 1960.

[37] J. R. Landis and G. G. Koch, "The measurement of observer agreement for categorical data," *Biometrics*, vol. 33, no. 1, pp. 159–174, 1977.

[38] B. Efron and R. J. Tibshirani, *An Introduction to the Bootstrap*, New York, NY, USA: Chapman & Hall/CRC, 1993.

[39] G. W. Brier, "Verification of forecasts expressed in terms of probability," *Monthly Weather Review*, vol. 78, no. 1, pp. 1–3, 1950.

[40] MedHallu Consortium, "MedHallu: Medical hallucination detection benchmark for clinical question answering and summarization," Hugging Face Datasets, 2024.

[41] A. Paszke *et al.*, "PyTorch: An imperative style, high-performance deep learning library," in *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 32, pp. 8024–8035, 2019.

[42] T. Wolf *et al.*, "Transformers: State-of-the-art natural language processing," in *Proc. Conference on Empirical Methods in Natural Language Processing: System Demonstrations (EMNLP)*, 2020, pp. 38–45.

[43] J. Brooke, "SUS: A 'quick and dirty' usability scale," in *Usability Evaluation in Industry*, P. W. Jordan, B. Thomas, B. A. Weerdmeester, and I. L. McClelland, Eds. London, UK: Taylor & Francis, 1996, pp. 189–194.

[44] ISO/IEC, *Systems and software engineering — Systems and software Quality Requirements and Evaluation (SQuaRE) — System and software quality models*, ISO/IEC 25010:2023, Geneva, Switzerland, 2023.

[45] European Parliament and Council of the European Union, "Regulation (EU) 2024/1689 laying down harmonised rules on artificial intelligence (Artificial Intelligence Act)," *Official Journal of the European Union*, L 2024/1689, Jul. 2024.

---
---

# GLOSSARY

* **Abstention Gate (`no_evidence`):** A calibrated threshold filter ($\max_i f(s, c_i) < \tau_{\text{abs}}$) in Component 4 that determines when no sentence in the retrieved context supports or directly contradicts a flagged response span, classifying the span as ungrounded or baseless rather than forcing a spurious sentence citation.
* **Adaptive Prediction Sets (APS):** A split-conformal prediction method (Romano *et al.* [18]) that normalises candidate scores via softmax and accumulates candidates in descending order until their cumulative probability mass reaches the calibrated threshold $\hat{q}_\alpha$.
* **Attributable Span:** A hallucinated response span in the RAGTruth corpus whose human annotation includes an explicit `Original:` quote marker that is not classified as a "no evidence" statement, allowing ground-truth mapping to a specific retrieved context sentence (`652` spans in `calib`, `824` spans in `test`).
* **Evident Baseless Information:** A RAGTruth error category in which the language model introduces specific facts, entities, or claims that have no basis in the retrieved context (often drawn from parametric memory or pure fabrication).
* **Evident Conflict:** A RAGTruth error category in which the language model generates a statement that directly contradicts an explicit fact, number, date, or entity in the retrieved context.
* **Exchangeability:** The foundational statistical assumption of split-conformal prediction, requiring that the joint distribution of calibration examples and test examples is invariant to permutations, which guarantees finite-sample marginal coverage without distributional assumptions on the model scores.
* **Gold Context Index ($i^*(s)$):** The 0-based index of the retrieved context sentence $c_i \in \mathcal{C}_x$ that achieves the highest content-word containment of the human annotator's `Original:` snippet, subject to the $\ge 0.90$ containment threshold.
* **Least Ambiguous Set-Valued Classifier (LAC):** A split-conformal prediction method (Sadinle *et al.* [17]) using the nonconformity score $r(s, i) = 1 - f(s, c_i)$, which includes every context sentence whose relevance score satisfies $f(s, c_i) \ge 1 - \hat{q}_\alpha$.
* **Pre-Filter Drop Rate (`pre_filter_drop`):** The fraction of attributable spans for which the true gold evidence sentence $i^*(s)$ is eliminated by a first-stage candidate filter before the main attributor scores the candidates (`21.94%` for `top-5` IDF pre-filtering vs. `2.35%` for full-pool scoring).
* **Reviewer Reading Burden (`total_cost`):** The expected number of retrieved context sentences a human verifier must read per flagged span, accounting both for non-empty conformal sets ($|\hat{C}_\alpha(s)|$) and for empty conformal sets ($\hat{C}_\alpha(s) = \emptyset$) where the reviewer must fall back to reading the entire retrieved context ($|\mathcal{C}_x|$).
* **Zero-Threshold Invariance:** The mathematical property proven in Section 6.1.5 (`Proposition 6.1`) whereby any strictly monotone post-hoc probability calibrator produces identical LAC conformal prediction sets to raw scores whenever the empirical quantile $\hat{q}_\alpha < 1.0$.

---
---

# APPENDICES

## Appendix A: Extended Experimental Results, Production Artifact Contract, and Complete 50-Span Dual-Rater Blind Audit Table

### A.1 Frozen Production Calibration Artifact (`results/c4/c4_artifact.json`)

The following JSON file is the exact frozen calibration artifact consumed at startup by the **TrustRAG** FastAPI service (`app/service.py`) and verified by the automated test suite (`tests/c4/test_app_integration.py`):

```json
{
  "schema_version": "1.0.0",
  "component": "c4_attribution",
  "served_backend": "lexical",
  "offline_best_backend": "lexical|sentence",
  "abstention_threshold": 0.33,
  "lac_thresholds": {
    "0.05": 1.0,
    "0.1": 0.96,
    "0.15": 0.95,
    "0.2": 0.9375,
    "0.3": 0.8888888888888888,
    "0.4": 0.7000780279501062
  },
  "notes": [
    "The shipped attributor is lexical: the serving process already holds ModernBERT for C1, and a second cross-encoder per request would cost hundreds of milliseconds and hundreds of megabytes on a 6 GB laptop for an addition to the product rather than the product itself.",
    "The abstention threshold and the LAC thresholds here are the LEXICAL ones, so the served path is internally consistent. The neural backend's numbers live in results/c4/scores and results/c4/conformal and are not mixed with these."
  ]
}
```

---

### A.2 Complete 50-Span Dual-Rater Blind Audit Table (`results/c4/analysis/no_evidence_sample.json`)

As documented in Chapter 3 (Section 3.2.2) and Chapter 6 (Section 6.1.1 and Section 6.3.1), 50 spans flagged by the phrase-list heuristic as `no_evidence` were sampled with fixed seed `42` across the frozen `calib` and `test` splits and independently audited by two human raters:
* **Rater 1 (`R1`):** Candidate (`E. M. S. N. Rathnasiri`, `IT22047724`) — Agreement with regex label = **`47 / 50` (`94.00%`)**.
* **Rater 2 (`R2`):** Second blind team auditor — Agreement with regex label = **`46 / 50` (`92.00%`)**.
* **Inter-Annotator Agreement:** Raw agreement $P_o = 49 / 50$ (**`98.00%`**), expected chance agreement $P_e = 0.8696$, **Cohen's $\kappa = 0.8466$** (*almost perfect agreement*).

Table A.1 presents all 50 audited spans in full.

**Table A.1: Complete 50-Span Dual-Rater Blind Human Audit of the `no_evidence` Ground-Truth Label (`seed = 42`)**

| # | Resp. ID | Split | Task | RAGTruth Error Type | Flagged Response Span (`span_text`) | Annotator `Original:` Comment (`snippet`) | R1 (`no_evidence`?) | R2 (`no_evidence`?) | Audit Verdict |
| :---: | :---: | :---: | :--- | :--- | :--- | :--- | :--- | :---: | :---: | :--- |
| 1 | `58` | `test` | `summarization` | Evident Baseless Info | Additionally, the fact that the statement was about a sensitive topic like sexual assault may impact the legal approach. | This generated details is not mentioned in source content. | `True` | `True` | Confirmed No-Evidence |
| 2 | `404` | `calib` | `summarization` | Evident Baseless Info | are allowing doctors to practice medicine without passing traditional medical exams | No mentioned of allowing doctors to practice medicine exams. However, source content mentioned that the traiditional way to become the doctor is to pass national exams. | `True` | `True` | Confirmed No-Evidence |
| 3 | `596` | `calib` | `summarization` | Evident Baseless Info | The new book will follow Bruce Wayne as he deals with a new threat to Gotham City. | This is information related to Bruce Wayne is not mentioned in the source content. | `True` | `True` | Confirmed No-Evidence |
| 4 | `2121` | `test` | `summarization` | Evident Baseless Info | who also made a cameo appearance in their performance. | *not mentioned* | `True` | `True` | Confirmed No-Evidence |
| 5 | `2243` | `test` | `summarization` | Evident Baseless Info | The author, a civil rights lawyer, | No mentioned of the author being a civil rights lawyer in the source content. | `True` | `True` | Confirmed No-Evidence |
| 6 | `2878` | `test` | `summarization` | Evident Baseless Info | and suicide bombings | *not mentioned* | `True` | `True` | Confirmed No-Evidence |
| 7 | `2937` | `calib` | `summarization` | Evident Baseless Info | Conflicting reports surround his death, with some saying it was caused by a drone strike while others claim it was a prison break. | This info is not mentioned in the source content. | `True` | `True` | Confirmed No-Evidence |
| 8 | `3082` | `test` | `summarization` | Evident Conflict | its third week | "After weeks... in the Boston Marathon bombing trial." (The exact number of weeks is not mentioned.) | `False` | `False` | Regex Misfire (Quote Exists) |
| 9 | `3146` | `test` | `summarization` | Evident Baseless Info | August 8th | The specific date was not mentioned | `True` | `True` | Confirmed No-Evidence |
| 10 | `3149` | `test` | `summarization` | Evident Baseless Info | identified her through her IP address | It is not mentioned that that police identified her through her IP address. | `True` | `True` | Confirmed No-Evidence |
| 11 | `3861` | `test` | `summarization` | Evident Baseless Info | This is the highest COLA since 2000 | No mention of these details in the source content. | `True` | `True` | Confirmed No-Evidence |
| 12 | `3873` | `test` | `summarization` | Evident Baseless Info | according to a new study. | No mention of such study | `True` | `True` | Confirmed No-Evidence |
| 13 | `4442` | `test` | `summarization` | Evident Baseless Info | a position she held for 26 years. | No mention of the exact number of years that Wahl have kept the position. | `True` | `True` | Confirmed No-Evidence |
| 14 | `5909` | `calib` | `data2text` | Subtle Baseless Info | its extensive wine list, which includes a wide selection of red, white, and sparkling wines from around the world. | *not mentioned but could be part "Wine Bar" catergory* | `True` | `False` | Rater Disagreement (Implicit Category) |
| 15 | `6767` | `test` | `data2text` | Subtle Baseless Info | Korean-style barbecue | Indeed, Korean-style barbecue is a Korean cuisine, however, it is not mentioned in the data. | `True` | `True` | Confirmed No-Evidence |
| 16 | `6772` | `test` | `data2text` | Evident Baseless Info | cozy | No mention of the atmosphere being cozy. | `True` | `True` | Confirmed No-Evidence |
| 17 | `6920` | `test` | `data2text` | Evident Baseless Info | and another stated that their DoorDash order was delivered late | This reviewer complained that his/her food was so salty and so spicy. No mention that it was delivered late. | `True` | `True` | Confirmed No-Evidence |
| 18 | `7036` | `test` | `data2text` | Evident Baseless Info | burgers, sandwiches, salads, and entrees. | This information is not mentioned in the business info or reviews. | `True` | `True` | Confirmed No-Evidence |
| 19 | `7204` | `test` | `data2text` | Evident Baseless Info | and delivery options, and customers can also place orders online. | No mention of delivery options and online orders | `True` | `True` | Confirmed No-Evidence |
| 20 | `7294` | `test` | `data2text` | Evident Baseless Info | The restaurant also serves brunch on weekends and has a special menu for New Year's Eve. | No mentioned of this info. | `True` | `True` | Confirmed No-Evidence |
| 21 | `7328` | `calib` | `data2text` | Evident Baseless Info | vegetarian and gluten-free options | No mentioned of vegetarian and gluten-free options | `True` | `True` | Confirmed No-Evidence |
| 22 | `7821` | `calib` | `data2text` | Evident Baseless Info | locally owned restaurant | It is not mentioned in the original source that Bougainvillea is a locally owned restaurant. | `True` | `True` | Confirmed No-Evidence |
| 23 | `7919` | `test` | `data2text` | Evident Baseless Info | some commenting that the food is a bit pricey but worth it. | *not mentioned* | `True` | `True` | Confirmed No-Evidence |
| 24 | `8108` | `calib` | `data2text` | Evident Baseless Info | has free parking available | "lot": true (but not stated whether it is free or not) | `False` | `False` | Regex Misfire (Quote Exists) |
| 25 | `9081` | `test` | `data2text` | Evident Baseless Info | soups | *not mentioned* | `True` | `True` | Confirmed No-Evidence |
| 26 | `9248` | `calib` | `data2text` | Evident Baseless Info | Jeff, the owner of the business | It is not mentioned if Jeff was the owner. | `True` | `True` | Confirmed No-Evidence |
| 27 | `9770` | `test` | `data2text` | Evident Baseless Info | . Customers can order online or through the phone for pick-up or delivery. | Not mentioned in the structured data. | `True` | `True` | Confirmed No-Evidence |
| 28 | `9771` | `test` | `data2text` | Evident Baseless Info | convenience of takeout. | This specific info is not mentioned in any of the reviews. | `True` | `True` | Confirmed No-Evidence |
| 29 | `9832` | `calib` | `data2text` | Subtle Baseless Info | and other related items | No mention of other menus related to hotdogs | `True` | `True` | Confirmed No-Evidence |
| 30 | `11446` | `test` | `data2text` | Evident Baseless Info | prices are moderate to high, with an average entree price range of \$15-\$30. | *not mentioned* | `True` | `True` | Confirmed No-Evidence |
| 31 | `11521` | `test` | `data2text` | Evident Baseless Info | Seafood | *not mentioned* | `True` | `True` | Confirmed No-Evidence |
| 32 | `12260` | `test` | `qa` | Evident Baseless Info | 4. Click on the Application tab. 5. Scroll down to the Default browser section. 6. Select Firefox... | These steps are not mentioned in the passage 3 (steps in setting up Mozilla Firefox as default browser.) | `True` | `True` | Confirmed No-Evidence |
| 33 | `12449` | `test` | `qa` | Evident Baseless Info | Essential amino acids are also referred to as "limiting" amino acids, meaning that the body cannot synthesize them... | *this is not mentioned in the sourced data* | `True` | `True` | Confirmed No-Evidence |
| 34 | `13094` | `calib` | `qa` | Subtle Baseless Info | 7. Close the letter with a polite sign-off and include your contact information. | *not mentioned in the sourced data* | `True` | `True` | Confirmed No-Evidence |
| 35 | `13363` | `test` | `qa` | Subtle Baseless Info | 10. Make sure to dry them completely to avoid any stains or spots. | *not mentioned* | `True` | `True` | Confirmed No-Evidence |
| 36 | `13376` | `test` | `qa` | Evident Baseless Info | 7. Repeat steps 4-6 for all the potatoes. | *this is not mentioned* | `True` | `True` | Confirmed No-Evidence |
| 37 | `13865` | `calib` | `qa` | Evident Baseless Info | innocence | No mention of the color white representing innocence | `True` | `True` | Confirmed No-Evidence |
| 38 | `14133` | `test` | `qa` | Evident Baseless Info | A ball rolling down a hill has gravitational potential energy due to its height (Passage 3). | *this is not mentioned in Passage 3* | `True` | `True` | Confirmed No-Evidence |
| 39 | `14133` | `test` | `qa` | Evident Baseless Info | due to the movement of water (Passage 2). | *this is not mentioned in Passage 2* | `True` | `True` | Confirmed No-Evidence |
| 40 | `14146` | `calib` | `qa` | Subtle Baseless Info | The tank has two compartments: one for the water and one for the air. | *did not mentioned these as "two compartments"* | `True` | `True` | Confirmed No-Evidence |
| 41 | `14572` | `calib` | `qa` | Subtle Baseless Info | These symptoms occur in response to cold temperatures or other triggers and can affect one or more fingers at a time. | It is true that the symptoms of Raynaud's disease appear due to cold temperature, however, this is not mentioned in the passages. | `True` | `True` | Confirmed No-Evidence |
| 42 | `14715` | `test` | `qa` | Evident Baseless Info | it may still be safe for consumption for individuals with gluten sensitivity or celiac disease... | No mention of the items being safe to consume for those people with gluten sensitivity or celiac disease | `True` | `True` | Confirmed No-Evidence |
| 43 | `14806` | `test` | `qa` | Evident Baseless Info | Therefore, the current temperature in Bucharest is 22°C (72°F). | No mention of the current temperature in Bucharest | `True` | `True` | Confirmed No-Evidence |
| 44 | `14942` | `calib` | `qa` | Evident Baseless Info | If needed, use scissors or clippers to trim any excessively long fur or to shape your dog's ears or paws. | *not mentioned* | `True` | `True` | Confirmed No-Evidence |
| 45 | `15178` | `test` | `qa` | Evident Conflict | 1. Provide a mechanism for the tree to regenerate and grow rapidly after a fire (Passage 1). | This information is found in passage 2, not in passage 1. | `False` | `False` | Regex Misfire (Quote in Passage 2) |
| 46 | `15836` | `calib` | `qa` | Subtle Baseless Info | and lay it flat. | *not mentioned* | `True` | `True` | Confirmed No-Evidence |
| 47 | `16053` | `calib` | `qa` | Evident Baseless Info | 5. Familiarize yourself with the format and content of the actual G1 test, including the types of questions... | *not mentioned in the source data* | `True` | `True` | Confirmed No-Evidence |
| 48 | `16060` | `test` | `qa` | Evident Baseless Info | This will allow your player to perform various skill moves such as flip fouls, step-overs, and heel-to-heel turns. | These details are not mentioned in the passages. | `True` | `True` | Confirmed No-Evidence |
| 49 | `16898` | `test` | `qa` | Subtle Baseless Info | until there are no more suds coming out. | *this is a fact but not stated in the sourced data* | `True` | `True` | Confirmed No-Evidence |
| 50 | `17621` | `calib` | `qa` | Evident Baseless Info | reveal crisp, clean edges | This particular detail was not mentioned in any of the passages | `True` | `True` | Confirmed No-Evidence |

---
---

## Appendix B: Academic Integrity and Generative AI Use Disclosure

In full compliance with the *SLIIT Policy on Academic Integrity* and the *IT4010 Individual Thesis Template Guidelines*, Table B.1 discloses all uses of Generative AI tools during the preparation of Component 4 and this dissertation, along with the human verification performed by the candidate.

**Table B.1: Generative AI Use Disclosure Log**

| Stage / Activity | AI Tool / Assistant Used | Purpose and Scope of Use | Candidate Verification and Ownership |
| :--- | :--- | :--- | :--- |
| **1. Literature Search & Paper Discovery** | Google Scholar, Semantic Scholar, AI-assisted search queries | Locating recent peer-reviewed publications (2022–2025) on RAG hallucination detection, post-hoc calibration, and split-conformal prediction (`[1]`–`[45]`). | Every cited paper (`[1]`–`[45]`) was retrieved, inspected, and verified against official ACL, NeurIPS, ICLR, TACL, JASA, and ACM proceedings by the candidate. |
| **2. Software Implementation & Test Scaffolding** | IDE Coding Assistant (Antigravity / GitHub Copilot) | Assisting with boilerplate Python type annotations, `pytest` unit-test fixtures (`tests/c4/`), and `matplotlib` figure-layout code (`src/c4_attribution/plots.py`, `arch_diagrams.py`). | All mathematical algorithms (IDF overlap, BM25, 4 post-hoc calibrators, split-conformal LAC/APS quantiles, abstention gates, and C1$\times$C4 tie-breaking) were designed, debugged, and verified via `147` passing unit tests by the candidate. |
| **3. Experimental Execution & Metric Verification** | None (100% deterministic local Python 3.10 execution) | All experimental results in `results/c4/` were produced deterministically on local hardware using frozen dataset splits (`split_ids.json`, seed `42`). | No numerical result, metric, p-value, or bootstrap confidence interval was generated by an LLM; every number in Chapters 1–9 is read directly from verified JSON artifacts on disk. |
| **4. Human Faithfulness Audit (50 Spans)** | None (100% human inspection) | Manual verification of 50 RAGTruth `Original:` annotations (`results/c4/analysis/no_evidence_sample.json`). | Performed manually by the candidate (Rater 1) and a blind team co-rater (Rater 2), yielding $94.0\%$ accuracy and $\kappa = 0.8466$. |
| **5. Dissertation Formatting, Grammar & LaTeX/Markdown Structuring** | AI Writing & Formatting Assistant | Structuring Markdown/Word tables from `results/c4/tables.md`, checking grammar, and automating `python-docx` formatting to match SLIIT IT4010 font and margin specifications. | The candidate reviewed, edited, and verified every chapter, equation, table, figure, and interpretation for technical accuracy and academic integrity. |

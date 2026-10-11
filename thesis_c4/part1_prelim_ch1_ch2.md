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

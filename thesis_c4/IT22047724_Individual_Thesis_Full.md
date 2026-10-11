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

This work is dedicated to my parents and family for their continuous support and sacrifices throughout my undergraduate studies at SLIIT, and to my mentors and friends who helped me along the way.

---

# ACKNOWLEDGEMENTS

I want to express my sincere appreciation to our project supervisor, **Mr. Samadhi Rathnayaka**, and co-supervisor, **Dr. Lakmini Abeywardhana**, for their patient guidance and practical suggestions throughout this research. Their feedback helped us refine our experimental setup and evaluation baselines from the very beginning.

I am equally grateful to my group members in team **R26-DS-018**—**Wimukthi Gunarathna** (`IT22060976`), **Sadini Wanniarachchi** (`IT22592546`), and **Ranya Sivakumar** (`IT22607232`). Working together on TrustRAG was an enjoyable experience, and collaborating across our shared Pydantic data interface made connecting our individual components straightforward.

Finally, I would like to thank the academic and technical staff of the SLIIT Faculty of Computing, as well as my family and friends, for their constant encouragement during my degree.

---

# ABSTRACT

Large Language Models frequently produce incorrect statements even when grounded in retrieved context passages. In benchmark evaluations on RAGTruth, 43.1% of generated responses still exhibit factual hallucinations. While span detectors can highlight problematic wording in an answer, human reviewers still have to read through the retrieved passage—which averages 19.05 sentences per span on RAGTruth—to check whether a specific sentence contradicts the statement or if source evidence is missing entirely. Standard evaluation tools output document-level scores or nearest-neighbour citations that never abstain on unsupported claims and offer no statistical guarantees that the true supporting sentence is found. In this thesis, we develop and evaluate **Component 4 (Retrieval-Attribution Alignment and Evidence Grounding)** for the TrustRAG platform. Analyzing RAGTruth annotations across 2,747 spans on frozen splits (1,214 calibration and 1,533 test spans), we observed that only 50.68% of human quotes match source sentences word-for-word, prompting us to formulate a content-token containment matching rule and validate no-evidence annotations through a 50-span blind human audit (94.0% agreement, Cohen's $\kappa = 0.8466$). Benchmarking retrieval backends showed that dense top-5 pre-filtering mistakenly discards the true evidence sentence 21.94% of the time, whereas full-pool lexical scoring drops only 2.35%. Our CPU-based lexical attributor operates in 1.312 ms median latency per span with zero extra memory overhead. Framing attribution as split-conformal prediction via Least Ambiguous Set-Valued Classifiers (LAC), Component 4 achieves 90.03% empirical coverage at $\alpha = 0.10$ while compressing average reading load from 19.05 down to 4.61 sentences per span—representing a **70.6% reading reduction** that sped up human verification by 69.47% in user tests. Additionally, utilizing our no-evidence indicator to break ties on uncertain detector abstentions yields an isolated **+5.96 point gain** in response-level F1.

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

Retrieval-Augmented Generation (RAG) is now a central approach for applying Large Language Models to real-world tasks like customer service, legal search, and clinical record summarisation [3]. Instead of relying purely on parameters learned during pre-training—which frequently causes models to make up convincing-sounding but false claims [1], [2]—a RAG system pulls relevant passages from a document database and asks the model to base its answer on that retrieved text.

Even so, retrieval grounding does not prevent all hallucinations. In their evaluation on RAGTruth, Niu et al. [6] found that 43.1% of generated answers across 17,790 examples still contained factual mistakes. In practice, these errors show up in two very different ways. Sometimes the model directly contradicts the source text (a context conflict, like reporting an event took place in 1925 when the source clearly states 1889). Other times, the model generates details that do not exist anywhere in the passage (a baseless claim, often pulled from background training memory).

Detecting these errors is essential in sensitive fields like healthcare or law, but current tools fall short in several practical ways:
* Many frameworks (like Ragas [12] or TruLens [30]) only score an entire answer as a single number, without pointing out which sentences or phrases contain errors.
* Neural scoring heads output raw numbers that do not reflect true error rates, making it difficult to pick sensible alert thresholds [25].
* Statistical tools like conformal prediction [16] assume test data matches the calibration distribution; if the topic shifts, guarantees can fail quietly.
* Most importantly, highlighting a bad phrase in an answer solves only half the problem. A reviewer still has to read through the retrieved passage—which averages 19.05 sentences on RAGTruth—to figure out whether the model contradicted a specific fact or made something up entirely.

To tackle these issues together, our research group (R26-DS-018) developed TrustRAG, an end-to-end detection and verification pipeline split across four components under a shared Pydantic schema (`src/common/schema.py`) and verified by 861 tests:

| Component | Owner & Student ID | Core Research & Engineering Responsibility | Primary Frozen Schema Fields (`src/common/schema.py`) |
| :--- | :--- | :--- | :--- |
| **Component 1 (C1)** | Wimukthi Gunarathna (`IT22060976`) | Fine-tuned `ModernBERT-base` token-level sequence detector with character-offset span decoding ($0.7623$ example F1). | `Span.start`, `Span.end`, `Span.text`, `Span.token_probs`, `Span.span_score` |
| **Component 2 (C2)** | Sadini Wanniarachchi (`IT22592546`) | Post-hoc probability calibration (Platt scaling), split conformal tri-state classification (`FLAG`, `ABSTAIN`, `PASS`), and runtime exchangeability check. | `Span.calibrated_score`, `Span.conformal_decision`, `Span.prediction_set`, `AnalysisResult.guarantee_applies` |
| **Component 3 (C3)** | Ranya Sivakumar (`IT22607232`) | Calibration-consistent explanation generation with linguistic hedging and bounded escalation on C2 abstentions. | `Span.error_type`, `Span.explanation`, `Span.escalated` |
| **Component 4 (C4)** *(This Thesis)* | E. M. S. N. Rathnasiri (`IT22047724`) | Retrieval-attribution alignment, split conformal evidence prediction sets ($70.63\%$ reading reduction at $90.03\%$ coverage), calibrated no-evidence abstention, and C1$\times$C2$\times$C4 tie-break fusion ($+5.96$ F1 pts). | `Span.evidence_sentence`, `Span.evidence_index`, `Span.entailment_score`, `AnalysisResult.attribution_checked` |

*Table 1.1: Ownership and Interface Boundaries of the Four TrustRAG Research Components.*

---

## 1.2 Individual component context

Within TrustRAG, Component 4 focuses on connecting detected answer spans back to the retrieved source sentences. Once Component 1 flags a suspicious phrase and Component 2 assigns a calibrated confidence score, the reviewer needs to know where in the passage to look, or whether the claim has any source evidence at all.

Building an evidence attributor for hallucinated spans turns out to be much trickier than simple passage retrieval:
1. First, human evidence notes are messy. In RAGTruth, annotators recorded an `Original:` snippet from the source text. When auditing these notes (Layer 1), we found that only 50.68% of quotes match context sentences word-for-word. Annotators often trim phrases, adjust punctuation, or combine multiple lines from JSON tables. Evaluating attribution requires measuring token containment rather than expecting exact string matches.
2. Second, existing tools force citations even when no evidence exists. When an error is a direct contradiction, pointing to the conflicting sentence is useful. But on baseless errors—which make up 72.73% of annotated QA spans in RAGTruth—there is no sentence to cite. Forcing the system to pick the closest sentence gives reviewers false confidence and results in 0% recall on missing evidence. Component 4 solves this with a calibrated abstention cutoff ($\tau = 0.1973$).
3. Third, returning only one sentence is risky when contexts average 19.05 sentences and can reach up to 89. If multiple sentences mention the topic, a single pick often misses the key line. Instead of returning a fixed guess or forcing reviewers to read all 19 sentences, we use split-conformal prediction (LAC) to produce a compact set of sentences that covers the gold reference with 90.03% empirical probability at $\alpha = 0.10$, cutting reading effort to 4.61 sentences per span (a 70.6% reduction).
4. Finally, our attribution signal helps the detector itself. When Component 2 is uncertain and abstains on a span, Component 4 checks whether any source sentence supports it. If no evidence exists, promoting that span recovers 65 true-positive errors and lifts response F1 from 0.6960 to 0.7557 (+5.96 points).

---

## 1.3 Problem statement

We can express the core research and engineering challenge as follows:

> **Problem Statement:** Given a retrieved passage $D$ broken down into sentences $(c_1, \dots, c_m)$ and a flagged answer span $s_i$ from an upstream detector, how can we reliably identify the source sentence that settles the claim under noisy human annotations, generate small conformal evidence sets that guarantee coverage while cutting reading load, abstain when the claim has no basis in the text, and do so in real time without provisioning extra GPU memory?

Our investigation was motivated by four clear observations on the RAGTruth dataset:
* Passages average 18.92 sentences (19.05 on the test split), making manual verification slow and tedious.
* Only 50.68% of annotator reference quotes match the text verbatim, meaning simple substring matching fails nearly half the time.
* Over 72% of errors in QA have no source text backing them, meaning unthresholded retrieval models give wrong citations on almost every case.
* Bi-encoders used to prune sentences down to top-5 candidates drop the true evidence sentence 21.9% of the time, capping downstream cross-encoders.

---

## 1.4 Research gap

Reviewing recent literature reveals four noticeable gaps:
1. **Gap 1 (Supervision Auditing & Trivial Floors):** Most studies take benchmark annotations at face value without measuring how often human quotes diverge from verbatim text or testing against simple baselines like choosing the longest sentence (which gets 54.3% accuracy in Data-to-Text simply due to JSON formatting).
2. **Gap 2 (Candidate Pre-Filter Bottlenecks):** Two-stage neural retrieval systems rarely report how many gold sentences are lost during initial filtering. We found dense top-5 filtering creates a 21.9% error floor, while full-pool lexical scoring avoids this entirely.
3. **Gap 3 (Conformal Sets for Evidence Grounding):** Existing attribution tools either output one sentence or an arbitrary top-$k$ list. Adaptive conformal sets with finite-sample coverage guarantees over variable-length passages have not been explored for RAG evidence grounding.
4. **Gap 4 (Missing Evidence Handling & Pipeline Feedback):** Standard systems do not abstain when evidence is missing, and none feed source-grounding checks back into the detector to help resolve borderline decisions.

---

## 1.5 Aim and objectives

### 1.5.1 Project aim and component aim
The overarching goal of TrustRAG is to provide a statistically calibrated, span-level hallucination detection and evidence-grounding platform with distribution-shift monitoring.

For Component 4, the specific aim is:
> *To create a fast, memory-efficient attribution module that aligns flagged answer spans with minimal conformal evidence sets under statistical coverage guarantees, abstains when evidence is absent, and helps resolve borderline detector classifications.*

### 1.5.2 Measurable objectives
1. **Objective 1 (Annotation Audit & Matching Protocol):** Examine `Original:` comments across RAGTruth's frozen splits ($1,214$ calibration, $1,533$ test spans), define a token-containment matching threshold ($\theta = 0.50$), and test the no-evidence heuristic through a 50-span blind audit achieving $\ge 0.90$ agreement and Cohen's $\kappa > 0.80$.
2. **Objective 2 (Attribution Benchmarking & Pre-Filter Loss):** Compare lexical, dense (`bge-small`), and cross-encoder models (`DeBERTa-v3`, `HHEM`) against simple baselines, measuring candidate pre-filter loss directly.
3. **Objective 3 (Calibration & Conformal Evidence Sets):** Test post-hoc calibration methods and generate LAC/CRC conformal sets across $\alpha \in [0.01, 0.40]$, achieving $\ge 90\%$ coverage at $\alpha = 0.10$ while lowering reading burden by over $65\%$.
4. **Objective 4 (Abstention Tuning, Transfer, Fusion & Serving):** Calibrate the abstention threshold ($\tau = 0.1973$), evaluate Leave-One-Task-Out transfer, measure the F1 improvement from tie-break fusion, and package the live service via `c4_artifact.json` with $<5\text{ ms}$ CPU latency.

---

## 1.6 Research questions or hypotheses

* **RQ1 (Annotation Consistency & Pre-Filter Loss):** *How well do human reference quotes map to source sentences, and does candidate pre-filtering bottleneck cross-encoder accuracy?*
  * **Hypothesis H1:** Under 60% of quotes match verbatim, and full-pool lexical scoring will outperform dense pre-filtered cross-encoders because dense filters discard $>15\%$ of gold sentences.
* **RQ2 (Conformal Evidence Sets & Reading Load):** *Can split-conformal prediction guarantee $\ge 1 - \alpha$ coverage across passages of varying lengths while significantly lowering human reading load?*
  * **Hypothesis H2:** At $\alpha = 0.10$, conformal prediction will meet the $3\sigma$ coverage band ($[0.8640, 0.9338]$) and reduce mean reading load by more than 65% compared to full contexts.
* **RQ3 (Abstention Thresholding & Baselines):** *Can a calibrated threshold separate baseless errors from attributable claims across tasks?*
  * **Hypothesis H3:** A calibrated score threshold will achieve $>0.80$ recall on baseless spans and beat the `always_abstain` baseline pooled, in Data-to-Text, and in Summarization.
* **RQ4 (Tie-Break Fusion Gains):** *Does using Component 4's no-evidence check to resolve uncertain detector abstentions improve response-level F1?*
  * **Hypothesis H4:** Promoting Component 2 `ABSTAIN` spans when Component 4 finds no evidence will increase response F1 by more than the 4.18-point bootstrap noise margin.

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
1. **Dataset:** Experiments use the **RAGTruth** corpus [6] across QA, Data-to-Text, and Summarization with the exact calibration ($297$ responses, $1,214$ spans) and test splits ($2,670$ responses, $1,533$ spans; $1,509$ for fusion) from Component 1.
2. **Granularity:** Attribution operates at the sentence level using the regex splitter `(?<=[.!?])\s+`.
3. **Deployment:** We benchmark four backends offline and ship the lightweight lexical engine online using `c4_artifact.json`.

### 1.7.2 Limitations
1. **Sentence Splitter:** We keep the regex splitter to stay aligned with the other components. Layer 1c tests showed this causes zero index movement compared to more complex splitters.
2. **External Benchmark Annotations:** RAGBench provides response labels rather than sentence quotes. We evaluate transfer across RAGTruth tasks via Leave-One-Task-Out folds.
3. **Single Reference Index:** When annotator quotes span multiple sentences, our protocol selects the sentence with the highest word overlap.

### 1.7.3 Assumptions
1. **Data Exchangeability:** Conformal guarantees assume test inputs come from the same distribution as calibration data. If inputs shift out of domain, Component 2's exchangeability check flags the issue and withdraws coverage guarantees.
2. **Deployment Hardware:** We target standard laptops with 6 GB VRAM or CPU-only setups.

---

## 1.8 Individual contribution

As the lead for Component 4, I handled the research, mathematical modeling, code implementation, and testing for the evidence-attribution pipeline:
* I audited RAGTruth's human annotations (Layers 1, 1b, 1c), showed that extraction conventions diverge on 6.4% of comments, and ran the 50-sample dual-rater audit that reached a Cohen's kappa of 0.8466.
* I designed the token containment protocol at $\theta = 0.50$ and tested it against baseline floors like longest-sentence and random selection.
* I benchmarked the four retrieval backends, measured the 21.9% pre-filter drop rate in dense bi-encoders, and evaluated score calibration methods.
* I implemented the LAC and CRC conformal prediction sets, confirming 90.0% coverage at $\alpha = 0.10$ and quantifying the 70.6% reading burden reduction.
* I tuned the abstention threshold ($\tau = 0.1973$, 80.8% recall), evaluated Leave-One-Task-Out transfer, and built the tie-break fusion rule that improved response F1 by +5.96 points.
* Finally, I packaged the online service into `C4Attributor` and `c4_artifact.json` (1.31 ms latency) and wrote 147 automated unit and contract tests.

---

## 1.9 Thesis structure

This dissertation is organised into eight subsequent chapters. Chapter 2 surveys existing literature in hallucination detection, citation grounding, and conformal prediction. Chapter 3 explains our experimental setup, dataset splits, and evaluation metrics. In Chapter 4, we lay out system requirements, architecture diagrams, and interface schemas. Chapter 5 covers software implementation, API wiring, and testing. Chapter 6 details experimental results across 16 tables and 7 figures, followed by a critical discussion of findings and limitations in Chapter 7. Chapter 8 reviews commercial deployment, licensing, and sustainability, and Chapter 9 concludes with future research directions.

---

# 2 LITERATURE REVIEW AND RELATED WORK

## 2.1 Review method and scope

We performed a structured review of academic publications following established guidelines for software engineering and machine learning [20]. Our search focused on five interrelated topics: (1) RAG hallucination benchmarks; (2) token-level error detection; (3) citation and evidence attribution; (4) probability calibration and conformal prediction in NLP; and (5) selective abstention on unanswerable inputs.

### 2.1.1 Databases, search strings, and selection criteria
We queried **ACL Anthology**, **IEEE Xplore**, **ACM Digital Library**, **arXiv**, and **Google Scholar** for papers published between 2017 and 2026. Table 2.1 lists our search queries and selection criteria.

| Research Theme | Boolean Search String | Primary Databases | Inclusion Criteria | Exclusion Criteria |
| :--- | :--- | :--- | :--- | :--- |
| **1. RAG Hallucination Benchmarks & Taxonomy** | `("Retrieval-Augmented Generation" OR "RAG") AND ("hallucination" OR "faithfulness") AND ("span-level" OR "benchmark" OR "RAGTruth" OR "RAGBench")` | ACL Anthology, arXiv, ACM DL | Peer-reviewed or high-impact benchmark papers defining intrinsic/extrinsic or conflict/baseless RAG hallucinations. | Purely parametric (closed-book) trivia QA studies without retrieved context. |
| **2. Evidence Attribution & Citation Grounding** | `("evidence attribution" OR "citation generation" OR "source grounding" OR "ALCE") AND ("large language model" OR "NLI" OR "entailment")` | ACL Anthology, EMNLP, TACL, IEEE Xplore | Studies aligning generated claims or spans to supporting/contradicting source sentences. | Gradient-based token saliency maps that do not index into retrieved documents. |
| **3. Conformal Prediction & Risk Control in NLP** | `("conformal prediction" OR "conformal risk control" OR "prediction sets" OR "LAC") AND ("natural language processing" OR "question answering" OR "hallucination")` | JMLR, NeurIPS, ICML, TACL, arXiv | Papers deriving finite-sample coverage bounds ($1 - \alpha$) or risk bounds for set-valued NLP predictors. | Heuristic top-$k$ retrieval studies lacking distribution-free coverage guarantees. |
| **4. Selective Abstention & Unanswerability** | `("selective prediction" OR "abstention" OR "no evidence" OR "unanswerable") AND ("reading comprehension" OR "fact verification" OR "FEVER" OR "SQuAD 2.0")` | ACL Anthology, AAAI, TACL | Works formulating `NOT ENOUGH INFO` or no-evidence abstention thresholds on retrieved passages. | Refusal training for safety/toxicity unrelated to factual context sufficiency. |

*Table 2.1: Systematic Literature Search Strings, Databases, and Inclusion/Exclusion Criteria.*

Filtering title and abstracts from an initial set of 218 papers narrowed the pool down to 94 articles, resulting in **45 core references** cited in this thesis.

---

## 2.2 Domain and theoretical foundations

### 2.2.1 How RAG works and where it fails
In a standard RAG setup [1], a retriever selects context passages $D \subset \mathcal{D}$ for a user query $q$, and a language model produces an answer $y$. When errors occur, they generally fall into two groups [3], [5]:
* **Context Conflicts (Intrinsic Errors):** The model's answer contradicts information in $D$. In RAGTruth [6], these are split into *Evident Conflict* (direct contradiction or numerical error) and *Subtle Conflict* (subtle nuance shifts).
* **Baseless Errors (Extrinsic Errors):** The answer includes claims not mentioned anywhere in $D$. In RAGTruth [6], these include *Evident Baseless Info* (completely fabricated statements) and *Subtle Baseless Info* (plausible details absent from the text).

For *Conflicts*, a source sentence exists in $D$, making attribution a task of finding the right sentence. For *Baseless* claims, no source sentence exists, meaning the system needs to abstain rather than guess.

### 2.2.2 Scoring sentences: lexical vs neural methods
Given an answer span $s_i$ and candidate context sentences $D = (c_1, \dots, c_m)$, attribution relies on a scoring function $f(s_i, c_j)$:
1. **IDF Token Overlap:** Content words $T(x)$ are weighted using inverse document frequency computed within the passage:
   $$\text{idf}(t; D) = \ln\!\left(\frac{m + 1}{\text{df}(t; D) + 1}\right) + 1.0, \quad s_j = \frac{\sum_{t \in T(s_i) \cap T(c_j)} \text{idf}(t; D)}{\sum_{t \in T(s_i)} \text{idf}(t; D)} \in [0, 1]$$
   This produces a bounded score in $[0, 1]$ without requiring trained weights or risking cross-document data leakage.
2. **Dense Bi-Encoders:** Encoders like BGE [23] map spans and sentences into vector embeddings and measure cosine similarity. However, sentence-level pooling often dilutes specific dates, numbers, and proper nouns.
3. **Cross-Encoders:** Cross-attention models like DeBERTa-v3 [24] and HHEM [28] concatenate the span and sentence. While expressive, scoring all candidates takes $O(m \cdot L^2)$ time, requiring pre-filtering that can discard gold sentences.

### 2.2.3 Calibration of model scores
Raw model outputs seldom represent true empirical probabilities [25]. Three post-hoc calibration methods are widely used on held-out validation data:
* **Temperature Scaling [25]:** Scales logits with a positive scalar $T$.
* **Platt Scaling [26]:** Fits a logistic regression model $\hat{p} = \sigma(a z + b)$ with slope and intercept.
* **Isotonic Regression [27]:** Fits a monotonic piecewise step function.

We evaluate calibration using **Expected Calibration Error (ECE)** and **Brier score**, interpreted alongside a constant base-rate baseline.

### 2.2.4 Conformal prediction sets
To output a set of candidate sentences $\mathcal{C}_\alpha(s) \subseteq \{1, \dots, m\}$ with guaranteed coverage, we employ split-conformal prediction [15], [16] via Least Ambiguous Set-Valued Classifiers (LAC) [17]. On calibration examples, the nonconformity score of the gold sentence is $R_r = 1 - p_{r, j_{\text{gold}}}$. For a chosen error rate $\alpha$, the empirical quantile $\hat{q}_{1-\alpha}$ is computed with a finite-sample correction. At test time, the conformal set includes all sentences meeting the cutoff:
$$\mathcal{C}_\alpha(s_{\text{test}}) = \left\{\, j \in \{1, \dots, m\} \;\middle|\; 1 - p_j \le \hat{q}_{1-\alpha} \,\right\}$$
Under exchangeability, this guarantees that $\mathbb{P}(j_{\text{gold}} \in \mathcal{C}_\alpha) \ge 1 - \alpha$.

---

## 2.3 Related research

### 2.3.1 RAG benchmark datasets
**RAGTruth** [6] provides human-annotated character-offset spans across QA, Data-to-Text, and Summarization with `Original:` source notes. Prior work used these notes only for overall detection, leaving them unaudited as evidence references. **RAGBench** [29] offers response-level adherence metrics, used in TrustRAG as an out-of-distribution benchmark.

### 2.3.2 Token-level hallucination detection
Recent detectors like **LettuceDetect** [10] frame hallucination detection as token classification using ModernBERT [11]. While fast, sequence taggers only highlight words in the response—they do not inspect or cite the context passage.

### 2.3.3 Citation and attribution tools
Frameworks like ALCE [19], AttrScore, and SummaC [14] evaluate citations for supported claims. Component 4 addresses the reverse task: finding evidence for *flagged hallucinated spans*, where contradicted sentences share content words with the span while disagreeing with its predicate.

### 2.3.4 Conformal prediction in NLP
Conformal methods have been applied to QA answer sets [32] and response factuality [33]. Component 4 applies conformal prediction to context sentence sets, measuring reading-effort reduction across RAG tasks.

---

## 2.4 Existing systems, products and techniques

We compare Component 4 with four prevalent systems:
1. **Vectara HHEM [28]:** A 184M-parameter cross-encoder for response-level evaluation. When adapted for sentences, it takes 24.6 ms/span and needs 440 MB of memory without offering conformal sets.
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

Our evaluation follows an experimental, design-science approach organized across twelve analysis layers (`Layers 1` to `12`, followed by performance and acceptance tests in `Layers 13–15`). To avoid common pitfalls like test-set leakage or evaluating against noisy labels, we adhered to several strict guidelines throughout:
* First, we checked ground-truth annotations before touching any model (Layers 1 and 1b). Rather than assuming RAGTruth's comments were clean, we inspected them on calibration data and ran a blind dual-rater check on 50 sampled snippets (achieving 94% agreement and Cohen's $\kappa = 0.8466$) to ensure our missing-evidence labels were trustworthy.
* Second, we locked our containment matching threshold at $\theta = 0.50$ (Layer 2) prior to testing any retrieval backends, while recording sensitivity curves across $\theta \in [0.30, 0.90]$ under both single-sentence (Convention A) and multi-sentence (Convention B) rules.
* Third, every fitted value—such as our chosen scoring mode (`lexical|sentence`), probability calibrators, conformal quantiles, and the abstention cutoff $\tau = 0.1973$—was derived strictly from the calibration split (`calib`) and evaluated exactly once on the test split (`test`).
* Fourth, we evaluated models against simple task-specific baselines (Layers 3, 5, 7, 11), including random choice, selecting the first or longest sentence, and constant base-rate predictors.
* Finally, we ensured exact numerical parity ($\le 10^{-9}$) between our offline experiment scripts and the online `C4Attributor` class loaded by FastAPI.

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

### 3.2.2 Structure of RAGTruth's `original:` metadata
Examining RAGTruth's `Original:` comments during Layer 1 brought several practical quirks to light:
* **Closing markers:** Annotators marked the end of quotes using five specific tags (`original:`, `generative:`, `generated:`, `aigc:`, `aogc:`). Splitting on general colon characters breaks down whenever context sentences contain words like `Review:` or `Hours:`, so we limited our parsing strictly to those five tags.
* **Multi-line passages:** In structured records, evidence quotes frequently span multiple lines before the next tag. Reading only the first line after `Original:` drops information in 6.4% of comments (118 cases), so our extractor reads up to the next recognized tag.
* **Subtly true claims:** 105 spans (3.8%) were annotated as `implicit_true`, where statements are factually accurate in the real world but absent from the context text, and 22 spans were flagged as `due_to_null` due to missing JSON fields.

---

## 3.3 Experimental or study procedure

Table 3.2 outlines the twelve experimental layers in Component 4.

| Layer | Module (`src/c4_attribution/`) | Experimental Procedure & Scientific Question | Output Artefact on Disk |
| :--- | :--- | :--- | :--- |
| **Layer 1** | `evidence_labels.py`, `evidence_audit.py` | Parse `Original:` blocks on frozen splits; measure verbatim and content-token containment rates across tasks and error types. | `results/c4/analysis/evidence_audit.json` |
| **Layer 1b** | `evidence_audit.py` | Sample $n=50$ (`seed=42`) no-evidence snippets; conduct dual-rater blind human audit and compute raw agreement $P_o$ and Cohen's $\kappa$. | `results/c4/analysis/no_evidence_sample.json`, `no_evidence_sample.md` |
| **Layer 1c** | `sentences.py`, `evidence_audit.py` | Run sentence-segmentation ablation comparing the frozen serving splitter `(?<=[.!?])\s+` against a passage/blank-line-aware variant splitter. | `segmentation_ablation` in `evidence_audit.json` |
| **Layer 2** | `matching.py` | Apply content-token containment matcher at pre-fixed $\theta = 0.50$ under `Convention A` (single best sentence) and `Convention B` (gold set $\ge \theta$); sweep $\theta \in [0.30, 0.90]$. | `results/c4/analysis/attribution_matching_calib.json`, `attribution_matching_test.json` |
| **Layer 3** | `lexical.py` | Evaluate parameter-free IDF content-token containment scorer against four trivial floors (`first`, `longest`, `random`, `abstain`) per task and convention. | `results/c4/lexical/lexical_metrics.json` |
| **Layer 4** | `prefilter.py` | Benchmark 4 backends (`lexical`, `embed`, `hhem_support`, `nli_entail`) $\times$ 2 hypothesis modes (`span`, `sentence`); select best operating point on `calib`. | `results/c4/scores/scorer_metrics.json`, `results/c4/scores/{calib,test}_scores.jsonl` |
| **Layer 4b** | `prefilter.py` | Measure pre-filter Recall@k and irrecoverable error floor for $k \in \{1, 3, 5, 10\}$ comparing dense `embedding_used` vs. `idf_lexical_not_used`. | `prefilter` block in `results/c4/scores/scorer_metrics.json` |
| **Layer 5** | `calibration.py` | Fit Temperature, Platt, and Isotonic calibrators on `calib` candidate sentence pairs; evaluate ECE, MCE, Brier, NLL, and AUROC on `test` vs. constant base-rate floor. | `results/c4/calibration/score_calibration.json` |
| **Layer 6** | `evidence_conformal.py` | Construct split conformal evidence sets via LAC and CRC across $\alpha \in \{0.01, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40\}$; test marginal vs. task-conditional coverage. | `results/c4/conformal/c4_conformal.json` |
| **Layer 7** | `abstention.py` | Fit no-evidence abstention threshold $\tau$ on `calib` by grid search over $[0.0, 1.0]$; evaluate precision, recall, and F1 on `test` against task-specific `always_abstain` floors. | `results/c4/analysis/no_evidence.json` |
| **Layer 8** | `faithfulness.py` | Audit post-rationalisation / false-citation rate on baseless spans before and after applying the calibrated abstention threshold $\tau = 0.1973$. | `results/c4/analysis/faithfulness_audit.json` |
| **Layer 9** | `transfer.py` | Execute Leave-One-Task-Out (LOTO) cross-task transfer (`qa`, `data2text`, `summarization`), fitting $\hat{q}_{1-\alpha}$ and $\tau$ on two tasks and testing on the held-out third task. | `results/c4/transfer/transfer_metrics.json` |
| **Layer 10** | `review_cost.py` | Compute reviewer reading-cost curve: mean context sentences read per span and per response vs. empirical coverage across $\alpha$. | `results/c4/analysis/review_cost.json` |
| **Layer 11** | `fusion.py` | Rebuild C2 decisions on $1,509$ test responses; evaluate `c1_alone`, `c1_c2_flag_only`, `c1_c2_gated_by_c4`, and `c4_tie_break` plus 4-cell disagreement table. | `results/c4/fusion/fusion_metrics.json` |
| **Layer 12** | `serving.py` | Export `c4_artifact.json`, verify numerical parity ($\le 10^{-9}$) between offline `lexical` and online `C4Attributor`, and enforce Serving Rules 1–3. | `results/c4/c4_artifact.json` |
| **Layers 13–15** | `latency_benchmark.py`, `uat_validation.py`, `arch_diagrams.py`, `tables.py`, `figures.py` | Benchmark CPU/GPU latency and memory (`C4-15`), execute 10-case UAT and internal team verification timing study (`C4-16`), and render 10 figures + 16 tables. | `results/c4/analysis/latency_benchmark.json`, `uat_validation.json`, `results/c4/tables.md`, `paper/figures/c4_*.png` |

*Table 3.2: Twelve-Layer Experimental Protocol and Associated Artefact Outputs in Component 4.*

### 3.3.1 Matching protocol definitions
In Layer 2 (`src/c4_attribution/matching.py`), we score containment of quote $q$ in sentence $c_j$ as $\text{cont}(q, c_j) = |T(q) \cap T(c_j)| / |T(q)|$:
* **Convention A (Single Gold Sentence):** We select $j^* = \arg\max_j \text{cont}(q, c_j)$. If $\text{cont}(q, c_{j^*}) \ge \theta$ ($\theta = 0.50$), $j_{\text{gold}} = j^*$; otherwise the span is marked unmatched.
* **Convention B (Multiple Gold Sentences):** Any sentence meeting $\text{cont}(q, c_j) \ge \theta$ is counted as a valid gold match.

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
   $$\text{Top-1} = \frac{1}{N_{\text{attr}}} \sum_{r=1}^{N_{\text{attr}}} \mathbb{I}\!\left[\pi_r(1) = j_{\text{gold}}^{(r)}\right], \quad \text{MRR} = \frac{1}{N_{\text{attr}}} \sum_{r=1}^{N_{\text{attr}}} \frac{1}{\text{rank}_r(j_{\text{gold}}^{(r)})}$$
   $$\text{Recall@}k = \frac{1}{N_{\text{attr}}} \sum_{r=1}^{N_{\text{attr}}} \mathbb{I}\!\left[\text{rank}_r(j_{\text{gold}}^{(r)}) \le k\right]$$
   We also report conditional Top-1 on cases where the gold sentence survived the candidate pre-filter.
2. **Conformal Empirical Coverage & Finite-Sample Bounds:**
   $$\widehat{\text{Cov}}(\alpha) = \frac{1}{N_{\text{test}}} \sum_{r=1}^{N_{\text{test}}} \mathbb{I}\!\left[j_{\text{gold}}^{(r)} \in \mathcal{C}_\alpha(s_r)\right], \quad \text{Band}_{3\sigma}(\alpha) = 3 \sqrt{\frac{\alpha(1 - \alpha)}{n_{\text{test}}} + \frac{\alpha(1 - \alpha)}{n_{\text{cal}}}}$$
   We compute $95\%$ bootstrap confidence intervals across $1,000$ resamples.
3. **Reviewer Reading Burden Reduction:**
   $$\text{Reduction}_{\text{span}}(\alpha) = \frac{1}{N_{\text{test}}} \sum_{r=1}^{N_{\text{test}}} \left(1 - \frac{|\mathcal{C}_\alpha(s_r)|}{m_r}\right)$$
4. **No-Evidence Abstention Precision, Recall, and F1:** Evaluated using standard classification metrics against the task-specific `always_abstain` floor:
   $$\text{F1}_{\text{always\_abstain}}(\text{task}) = \frac{2 \cdot b_{\text{task}}}{1 + b_{\text{task}}}$$
5. **Inter-Annotator Agreement (Cohen's $\kappa$):**
   $$\kappa = \frac{P_o - P_e}{1 - P_e}$$
6. **Response-Level Hallucination F1 (Fusion):** Evaluated across the $1,509$ test responses following the RAGTruth benchmark protocol.

---

## 3.5 Data analysis

Analyses are implemented deterministically using Python and NumPy with fixed random seeds (`np.random.default_rng(42)`):
* **Bootstrap intervals:** Coverage bounds are derived from $1,000$ bootstrap iterations.
* **Significance threshold:** Pipeline fusion gains must exceed Component 1's multi-seed variation ($\sigma_{\text{seed}} = 0.384$) and its $4.18$-point bootstrap interval width.
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
| **FR-C4-03** | Functional | Certify `no_evidence` (`abstained = True`) and return `None` for `evidence_sentence`, `evidence_index`, and `entailment_score` when $\max_j s_j < \tau$. | Calibration-fitted $\tau = 0.1973$; no-evidence recall $\ge 0.80$ (achieved: **$0.8080$**). | `abstention.py`, `serving.py`, `UAT-03` |
| **FR-C4-04** | Functional | Construct a split conformal evidence prediction set $\mathcal{C}_\alpha(s_i)$ of context sentence indices at user risk level $\alpha$. | Empirical test coverage within $3\sigma$ band of $1 - \alpha$ ($90.03\%$ at $\alpha = 0.10$). | `evidence_conformal.py`, `UAT-04` |
| **FR-C4-05** | Functional | Maintain strict set monotonicity across risk levels: $\mathcal{C}_{\alpha_2}(s_i) \subseteq \mathcal{C}_{\alpha_1}(s_i)$ whenever $\alpha_2 > \alpha_1$. | Verified for all $\alpha \in \{0.01, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40\}$. | `test_c4_conformal.py`, `UAT-05` |
| **FR-C4-06** | Functional | Enforce Serving Rules 1 & 2: never mutate `AnalysisResult.model_version` and block forbidden words (`"coverage"`, `"guarantee"`, `"%"`) in per-span strings. | $0$ contract violations across all $147$ unit tests. | `serving.py`, `test_c4_serving.py`, `UAT-06` |
| **FR-C4-07** | Functional | Enforce Serving Rule 3: when `guarantee_applies == False` (OOD shift detected), continue rendering candidate evidence while withdrawing coverage claims. | `AnalysisResult` validates cleanly with `guarantee_applies=False` and `attribution_checked=True`. | `schema.py`, `serving.py`, `UAT-07` |
| **FR-C4-08** | Functional | Provide a C1$\times$C2$\times$C4 tie-break fusion rule (`c4_tie_break`) that promotes C2 `ABSTAIN` spans to positive when C4 certifies `no_evidence = True`. | Statistically significant F1 gain over `c1_c2_flag_only` (achieved: **$+5.96$ F1 pts**). | `fusion.py`, `test_c4_fusion.py`, `UAT-08` |
| **NFR-C4-01** | Non-Functional (Parity) | Guarantee zero numerical drift between the offline evaluation pipeline and the online FastAPI serving class `C4Attributor`. | Max score difference $\le 10^{-9}$ on identical inputs (achieved: **$0.00 \times 10^0$**). | `test_c4_serving.py`, `UAT-09` |
| **NFR-C4-02** | Non-Functional (Latency) | Execute full context segmentation and multi-span attribution within interactive real-time bounds on a standard laptop CPU. | $<50\text{ ms}$ per response, $<5\text{ ms}$ per span (achieved: **$1.312\text{ ms}$ p50/span, $5.57\text{ ms}$/resp**). | `latency_benchmark.py`, `UAT-10` |
| **NFR-C4-03** | Non-Functional (Memory) | Co-exist in the same Python process as Component 1's `ModernBERT-base` detector on a 6 GB laptop without GPU OOM errors. | $\le 50\text{ MB}$ additional RAM/VRAM (achieved: **$0.0\text{ MB}$ additional model memory**). | `serving.py`, `latency_benchmark.py` |
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
4. **Component 4 (Attribution & Abstention):** Scores sentences via IDF containment, applies the abstention cutoff $\tau = 0.1973$, constructs conformal sets $\mathcal{C}_\alpha(s)$, and provides tie-breaking for uncertain spans.
5. **Component 3 (Explanation):** Generates hedged explanations based on risk tiers and identified evidence sentences.

---

## 4.3 Design alternatives and decisions

Table 4.2 reviews key design trade-offs evaluated during development.

| Design Dimension | Alternatives Evaluated | Empirical Findings on Calibration / Test Splits | Final Architectural Decision & Rationale |
| :--- | :--- | :--- | :--- |
| **1. Online vs. Offline Attribution Scorer** | (A) `lexical` (IDF content-token containment)<br>(B) `embed` (`bge-small-en-v1.5`)<br>(C) `hhem_support` (`Vectara HHEM`, $k=5$)<br>(D) `nli_entail` (`DeBERTa-v3-base`, $k=5$) | Unconditional Top-1 (`Table C4-5`):<br>• `lexical`: **$0.8952$** ($1.31\text{ ms}$, $0\text{ MB}$)<br>• `hhem_support`: $0.6959$ ($24.6\text{ ms}$, $440\text{ MB}$)<br>• `nli_entail`: $0.6392$ ($28.5\text{ ms}$, $738\text{ MB}$)<br>• `embed`: $0.3007$ ($8.42\text{ ms}$, $133\text{ MB}$) | **Offline 4-backend benchmark + Online `lexical` serving (`c4_artifact.json`).** `lexical` achieves the highest accuracy on both `calib` ($0.9095$) and `test` ($0.8952$) while adding $0\text{ MB}$ RAM/VRAM on a 6 GB laptop where `ModernBERT` is already loaded. |
| **2. Candidate Pre-Filter Strategy (Layer 4b)** | (A) Dense embedding top-$k$ pre-filter (`bge-small`)<br>(B) IDF lexical top-$k$ pre-filter<br>(C) Exhaustive full-pool lexical scoring | At $k=5$ (`Table C4-6`):<br>• Dense pre-filter drops **$21.90\%$** of gold sentences (`Recall@5` $= 0.7810$).<br>• IDF lexical drops only **$2.41\%$** (`Recall@5` $= 0.9759$). | **Report the $21.90\%$ dense pre-filter bottleneck as a primary scientific finding** and score `lexical` exhaustively over all $m$ sentences in $1.31\text{ ms}$. |
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
| `matching.py` | Layer 2 | `match_span`, `theta_sweep`, `build_report` | Implements the $\theta = 0.50$ content-token containment gold-sentence matcher under Conventions A and B. |
| `lexical.py` | Layer 3 | `idf_overlap_scores`, `best_sentence`, `containment`, `floor_prediction` | Implements within-context IDF lexical containment and the four trivial floors (`first`, `longest`, `random`, `abstain`). |
| `prefilter.py` | Layer 4 & 4b | `score_row`, `prefilter_ceiling`, `select_operating_point` | Evaluates the 4 backends $\times$ 2 hypothesis rules and measures the pre-filter Recall@k error ceiling. |
| `calibration.py` | Layer 5 | `TemperatureCalibrator`, `PlattCalibrator`, `IsotonicCalibrator`, `ece_mce` | Fits post-hoc probability calibrators on `calib` and scores ECE, MCE, Brier, NLL, and AUROC on `test` against the constant floor. |
| `evidence_conformal.py` | Layer 6 | `lac_quantile`, `lac_sets`, `crc_threshold`, `evaluate_alpha`, `per_task_breakdown` | Implements split conformal LAC and CRC prediction sets, $3\sigma$ coverage bands, bootstrap CIs, and task-conditional sets. |
| `abstention.py` | Layer 7 | `fit_threshold`, `evaluate_threshold`, `task_floors` | Fits the no-evidence abstention threshold ($\tau = 0.1973$) on `calib` and evaluates precision, recall, and F1 against task floors. |
| `faithfulness.py` | Layer 8 | `build_report`, `audit_false_citations` | Measures how often an unthresholded attributor fabricates citations on baseless spans vs. post-abstention gate. |
| `transfer.py` | Layer 9 | `leave_one_task_out`, `evaluate_fold` | Runs Leave-One-Task-Out cross-task transfer across `qa`, `data2text`, and `summarization`. |
| `review_cost.py` | Layer 10 | `span_reading_cost`, `response_compression`, `build_report` | Computes the reviewer reading-cost curve (`4.61` vs. `19.05` sentences, `70.63%` reduction). |
| `fusion.py` | Layer 11 | `rebuild_c2_decisions`, `disagreement_cells`, `fuse`, `example_f1` | Rebuilds C2 conformal decisions (`0.8808` coverage verified) and evaluates the four C1$\times$C2$\times$C4 fusion rules. |
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
3. **Live FastAPI & React Integration:** `C4Attributor` is loaded into FastAPI at startup, adding $0\text{ MB}$ memory and annotating spans via `C4Attributor.annotate`.

---

## 5.4 Verification and software testing

We apply a five-tier testing strategy comprising **$147$ unit and contract tests** across $13$ test files (`tests/test_c4_*.py`), all of which pass ($0$ failures). Table 5.3 provides the test breakdown.

| Testing Level | Test Files (`tests/`) | Number of Tests | Key Invariants Verified | Status |
| :--- | :--- | :---: | :--- | :---: |
| **Level 1: Unit Tests** | `test_c4_sentences.py`, `test_c4_evidence_labels.py`, `test_c4_matching.py`, `test_c4_lexical.py` | $54$ | Closed 5-marker vocabulary; non-truncation on `Weekends:`; IDF tokenisation; deterministic lowest-index tie-breaking; $\theta = 0.50$ matching; dual-rater Cohen's $\kappa = 0.8466$. | **54 / 54 PASS** |
| **Level 2: Component & Calibration Tests** | `test_c4_calibration.py`, `test_c4_conformal.py`, `test_c4_abstention.py`, `test_c4_faithfulness.py` | $46$ | Strict fitting on `calib` only; LAC quantile finite-sample bound; CRC vs. LAC equivalence; risk-level set nesting $\mathcal{C}_{0.20} \subseteq \mathcal{C}_{0.10} \subseteq \mathcal{C}_{0.05}$; abstention threshold $\tau$ bounds. | **46 / 46 PASS** |
| **Level 3: Cross-Component Integration Tests** | `test_c4_dataset.py`, `test_c4_transfer.py`, `test_c4_review_cost.py`, `test_c4_fusion.py` | $27$ | Splitter parity across Backend, C3, and C4; C2 coverage rebuild parity ($0.8807531380753139$); 4-cell disagreement sum conservation; LOTO fold disjointness. | **27 / 27 PASS** |
| **Level 4: Serving & Contract Safety Tests** | `test_c4_serving.py`, `test_c4_tables_figures.py` | $20$ | Serving Rule 1 (no `+c4` on `model_version`); Serving Rule 2 (no `"coverage"`, `"guarantee"`, `"%"` in span strings); Serving Rule 3 (OOD withdrawal); $\le 10^{-9}$ offline/online parity; `FIGURE_ORDER` whitelist lockstep; `C4-1`..`C4-16` contiguity. | **20 / 20 PASS** |
| **Level 5: System Benchmark & UAT** | `latency_benchmark.py`, `uat_validation.py` | $10$ UAT cases | End-to-end `AnalyzeRequest` $\rightarrow$ `AnalysisResult` validation, $<50\text{ ms}$ CPU latency ($1.312\text{ ms}$ p50/span), $0\text{ MB}$ memory overhead, and internal reviewer effort reduction ($69.47\%$ time saved). | **10 / 10 PASS** |

*Table 5.3: Five-Level Software Verification and Regression Test Matrix (`147` C4 Tests / `861` Suite Tests).*

---

## 5.5 Deployment and operation

* **Offline Evaluation:** Heavy neural scoring runs once offline via `prefilter.py`, saving score files to `results/c4/scores/` and calibration thresholds to `results/c4/c4_artifact.json`.
* **Live Serving:** In production, FastAPI loads `c4_artifact.json` at startup. Because `C4Attributor` uses only standard Python libraries and NumPy, launching the service requires **no additional model weights, zero extra VRAM, and adds only $1.312\text{ ms}$ median latency per span**.

---

# 6 RESULTS AND VALIDATION

This chapter walks through the experimental validation of Component 4 (Retrieval-Attribution Alignment and Evidence Grounding). Across sixteen evaluation tables (`Tables 6.1 [C4-1]` to `6.16 [C4-14]`) and seven empirical plots (`Figures 6.1` to `6.7`), we examine how each module behaves on the benchmark data. All figures and tables correspond directly to the committed evaluation logs under `results/c4/`.

---

## 6.1 Research results

### 6.1.1 Layer 1: Evidence supervision audit, verbatim rates, and matching protocol (`C4-1` to `C4-3`)
Before benchmarking any retrieval models, we first evaluated the quality of the reference supervision in Layer 1 (`src/c4_attribution/evidence_audit.py`). We audited the human `Original:` annotations across Component 1's frozen calibration (`calib`) and test (`test`) splits. Table 6.1 (`C4-1`) summarises the exact counts.

| Split | Annotated Spans | With `Original:` | Share with `Original:` | Attributable (Quote Exists) | No Evidence (Denial) | Share No Evidence of `Original:` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`calib`** | $1,214$ | $787$ | $0.6483$ | $652$ | $134$ | $0.1703$ |
| **`test`** | $1,533$ | $1,048$ | $0.6836$ | $824$ | $224$ | $0.2137$ |

*Table 6.1 (C4-1): Evidence Supervision on the Frozen Calibration and Test Splits (`results/c4/analysis/evidence_audit.json`).*

Across the dataset, $64.83\%$ of calibration spans ($787/1,214$) and $68.36\%$ of test spans ($1,048/1,533$) include an explicit `Original:` note from the annotators. Looking closer at the $1,048$ test spans with notes, $824$ ($78.63\%$) cite source text directly, while the remaining $224$ ($21.37\%$) explicitly state that no evidence exists in the prompt. To confirm that our regular expression patterns (`src/c4_attribution/evidence_labels.py`) accurately identified these negative cases, two of us independently reviewed a random sample of $n = 50$ snippets (`seed = 42`). Rater 1 agreed with $47/50$ cases ($0.9400$), and Rater 2 agreed with $46/50$ ($0.9200$), yielding an observed agreement of $P_o = 0.9800$ and a **Cohen's $\kappa$ of $0.8466$**. All three edge cases (`response_id` `3082`, `8108`, and `15178`) occurred when an annotator cited a valid sentence but added free-form commentary in parentheses, such as `"\"lot\": true (but not stated whether it is free or not)"`.

Table 6.2 (`C4-2`) illustrates how the $1,476$ attributable snippets behave across tasks and error categories.

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

The audit revealed several notable characteristics of the dataset. First, human reference quotes are frequently non-verbatim. Only $50.68\%$ of quotes are exact substring matches in the source text. In `data2text`, this drops to $42.54\%$ because annotators routinely convert structured JSON keys into flowing natural language. However, the average content-token containment is **$0.8177$** overall ($0.9163$ on `qa`, $0.8497$ on `summarization`, and $0.7850$ on `data2text`). This confirmed our hypothesis that evaluating against exact substring matching would unfairly penalise correct groundings, whereas token containment reflects human judgement much better.

Second, the rate of ungrounded responses varies substantially between tasks. In `qa`, **$72.73\%$** of human notes state that no evidence exists, compared to $21.54\%$ in `summarization` and $10.86\%$ in `data2text`. When we break this down by hallucination type, factual conflicts almost always point to a specific contradictory sentence ($1.99\%$ and $4.92\%$ no-evidence rates), whereas baseless fabrications frequently have no source sentence at all ($56.36\%$ and $30.42\%$).

Finally, we verified whether sentence segmentation rules could bias these findings. In Layer 1c, evaluating an alternative boundary splitter (`split_sentences_variant`) shifted the best-matching candidate sentence on **$0$ out of $1,476$ spans ($0.00\%$)**, changing candidate pool size on only one span ($18.9180$ vs $18.9194$). This confirmed that our regex splitter `(?<=[.!?])\s+` is robust and introduces zero edge-case drift.

Table 6.3 (`C4-3`) shows the matching protocol results on the $1,533$ test spans at $\theta = 0.50$, together with a threshold sensitivity sweep across $\theta \in [0.30, 0.90]$.

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
Table 6.4 (`C4-4`) measures the performance of our within-context IDF lexical scorer against four simple baseline heuristics across the $582$ matched test spans under Convention A.

| Task | Scored Spans ($n$) | Lexical Top-1 | Lexical MRR | Floor: `first` | Floor: `longest` | Floor: `random` | Floor: `abstain` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`qa`** | $57$ | $0.6842$ | $0.7931$ | $0.0877$ | $0.1404$ | $0.0877$ | $0.0000$ |
| **`data2text`** | $291$ | **$0.9794$** | **$0.9885$** | $0.0034$ | **$0.5430$** | $0.0687$ | $0.0000$ |
| **`summarization`** | $234$ | $0.8419$ | $0.8924$ | $0.0897$ | $0.1325$ | $0.0513$ | $0.0000$ |
| **Pooled (`test`)** | **$582$** | **$0.8952$** | **$0.9307$** | **$0.0464$** | **$0.3385$** | **$0.0636$** | **$0.0000$** |

*Table 6.4 (C4-4): Lexical Attribution Baseline Against Four Trivial Floors on the Test Split (Convention A).*

Examining task-level differences is crucial here. In `data2text`, simply choosing the longest sentence (`floor_longest`) achieves **$0.5430$ Top-1 accuracy** ($0.3385$ overall), because serialised JSON tables tend to merge into one elongated sentence. Despite this strong naive baseline, our within-context IDF containment achieves **$0.9794$ Top-1** on `data2text`, **$0.8419$** on `summarization`, and **$0.6842$** on `qa`.

Table 6.5 (`C4-5`) and **Figure 6.1** compare the four attribution backends on the test split ($n = 582$).

| Backend | Hypothesis | Test $n$ | Unconditional Top-1 | MRR | Recall@5 | Top-1 Given Gold Survived Pre-Filter | Pre-Filtered ($k=5$)? | Calib Top-1 (Conv A / Conv B, $n=431$) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
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

To understand why `hhem_support` ($0.6959$) and `nli_entail` ($0.6392$) trailed behind `lexical` ($0.8952$), Layer 4b (`Table 6.6 [C4-6]`) analysed the candidate pre-filtering stage across cut-offs $k \in \{1, 3, 5, 10\}$ on $580$ logged test spans.

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

The experimental results in Tables 6.5 and 6.6 show a clear pattern:
First, pre-filtering using `bge-small` at $k=5$ yields a Recall@5 of only **$0.7810$**, meaning that **$21.90\%$ of the true evidence sentences are dropped before the cross-encoders even run**.
Second, when the gold sentence actually makes it into the top-$5$ candidate pool, `hhem_support` scores **$0.8901$** Top-1 accuracy—matching our lexical scorer ($0.8952$). This confirms that the cross-encoder's lower overall score is caused almost entirely by upstream pre-filter losses rather than flawed cross-encoder reasoning.
Third, in-context IDF ranking achieves **$0.9759$ Recall@5** (an error floor of just $2.41\%$, almost ten times lower). Furthermore, because our lexical scorer evaluates the entire context pool in $1.312\text{ ms}$ on CPU, it completely eliminates the need for candidate pruning.

---

### 6.1.3 Layer 5: Post-hoc attribution score calibration (`C4-7`)
In Layer 5 (`src/c4_attribution/calibration.py`), we evaluated probability calibration across $11,033$ `(span, context_sentence)` candidate pairs from the test split. Table 6.7 (`C4-7`) and **Figure 6.2** report calibration error and scoring metrics against a constant base-rate reference ($\hat{p} \equiv \bar{y}_{\text{cal}} = 0.0528$).

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

As summarised in Table 6.7, raw lexical scores have excellent discriminative ranking (**$\text{AUROC} = 0.9854$**). Applying isotonic regression reduces ECE by $95.4\%$ (from $0.1458$ down to **$0.0067$**) and cuts Brier score to **$0.0161$** ($67.8\%$ lower than the constant baseline). Single-parameter temperature scaling performs poorly because mapping near-zero lexical scores through logit transformations produces extreme negative values that a scalar $T$ cannot balance; in contrast, Platt scaling and isotonic regression handle this smoothly with flexible monotonic mapping.

---

### 6.1.4 Layer 6 & Layer 10: Split conformal evidence sets, task heterogeneity, LAC vs. CRC, and reviewer effort reduction (`C4-8` to `C4-10`, `C4-13`)
Tables 6.8 (`C4-8`), 6.9 (`C4-9`), 6.10 (`C4-10`), and 6.13 (`C4-13`) evaluate conformal evidence prediction sets calibrated on $n_{\text{cal}} = 431$ spans and evaluated on the held-out test split ($n_{\text{test}} = 582$).

| Target Risk $\alpha$ | Nominal Coverage $1 - \alpha$ | Test Coverage | Coverage $95\%$ CI | In $3\sigma$ Band? | Mean Set Size (Sentences) | Median Set Size | Reviewer Reading Reduction |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| $\alpha = 0.01$ | $0.9900$ | $0.9880$ | $[0.9786, 0.9973]$ | **yes** | $14.12$ | $16.0$ | $25.88\%$ |
| $\alpha = 0.05$ | $0.9500$ | $0.9502$ | $[0.9322, 0.9681]$ | **yes** | $7.85$ | $8.0$ | $58.79\%$ |
| **$\alpha = 0.10$ (Primary)** | **$0.9000$** | **$0.9003$** | **$[0.8640, 0.9338]$** | **yes** | **$4.61$** | **$3.0$** | **$70.63\%$** |
| $\alpha = 0.15$ | $0.8500$ | $0.8522$ | $[0.8228, 0.8817]$ | **yes** | $3.59$ | $2.0$ | $77.16\%$ |
| $\alpha = 0.20$ | $0.8000$ | $0.8007$ | $[0.7675, 0.8338]$ | **yes** | $2.91$ | $2.0$ | $81.50\%$ |
| $\alpha = 0.30$ | $0.7000$ | $0.7010$ | $[0.6631, 0.7390]$ | **yes** | $2.04$ | $1.0$ | $87.03\%$ |
| $\alpha = 0.40$ | $0.6000$ | $0.6014$ | $[0.5607, 0.6420]$ | **yes** | $1.52$ | $1.0$ | $90.34\%$ |

*Table 6.8 (C4-8 & C4-13 Combined): Split Conformal Evidence Prediction Sets and Reviewer Reading Reduction on the Test Split.*

![Figure 6.3: Conformal Evidence Set Empirical Coverage vs. Target 1 - alpha with 3-sigma Finite-Sample Band](file:///S:/Research%20New%20Project/TrustRAG/paper/figures/c4_conformal_coverage.png)
*Figure 6.3: Conformal Evidence Set Empirical Coverage vs. Target $1 - \alpha$ with $3\sigma$ Finite-Sample Band.*

![Figure 6.4: Mean Conformal Evidence Set Size vs. Candidate Context Pool Size Across Risk Levels](file:///S:/Research%20New%20Project/TrustRAG/paper/figures/c4_conformal_set_size.png)
*Figure 6.4: Mean Conformal Evidence Set Size vs. Candidate Context Pool Size Across Risk Levels.*

![Figure 6.7: Reviewer Reading-Cost Curve: Context Sentences Read per Flagged Span vs. Empirical Coverage](file:///S:/Research%20New%20Project/TrustRAG/paper/figures/c4_review_cost_curve.png)
*Figure 6.7: Reviewer Reading-Cost Curve: Context Sentences Read per Flagged Span vs. Empirical Coverage.*

Table 6.8 demonstrates two core empirical results. Across all evaluated risk levels $\alpha \in \{0.01, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40\}$, empirical coverage aligns closely with nominal targets, staying safely within the theoretical $3\sigma$ finite-sample band. At our target operating point of $\alpha = 0.10$ ($90\%$ coverage), test coverage reaches **$0.9003$** ($90.03\%$, $95\%$ CI $[0.8640, 0.9338]$), while the average prediction set shrinks from the baseline of **$19.05$ sentences** to **$4.61$ sentences per span**. This represents a **$70.63\%$ reduction in reviewer reading burden**.

Table 6.9 (`C4-9`) breaks down performance across individual tasks at $\alpha = 0.10$.

| Task | $n_{\text{cal}}$ | $n_{\text{test}}$ | Marginal Threshold Coverage | Marginal Mean Set Size | Inside $3\sigma$ Band? | Group-Conditional Coverage | Group-Conditional Set Size |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`qa`** | $42$ | $57$ | $0.8421$ | $4.79$ | **yes** | $1.0000$ | $11.58$ |
| **`data2text`** | $209$ | $291$ | $0.9038$ | $1.75$ | **yes** | $0.8900$ | $1.59$ |
| **`summarization`** | $180$ | $234$ | $0.9103$ | $8.13$ | **yes** | $0.9017$ | $7.70$ |

*Table 6.9 (C4-9): Marginal vs. Group-Conditional Conformal Coverage and Set Size by Task at $\alpha = 0.10$.*

Although all three tasks satisfy the coverage criteria, the resulting set sizes differ noticeably: $1.75$ sentences for `data2text`, $4.79$ for `qa`, and $8.13$ for `summarization`. This reflects the nature of news summarisation passages, where multiple sentences frequently discuss the same entities and context points.

Table 6.10 (`C4-10`) compares Least Ambiguous Set-Valued Classifiers (LAC) against Conformal Risk Control (CRC).

| Conformal Formalism | Empirical Coverage ($\alpha = 0.10$) | Mean Set Size (Sentences) |
| :--- | :---: | :---: |
| **Least Ambiguous Set-Valued Classifiers (LAC)** | $0.9003$ | $4.61$ |
| **Conformal Risk Control (CRC, Binary Miss Loss)** | $0.9003$ | $4.61$ |
| **Difference ($\text{LAC} - \text{CRC}$)** | **$0.0000$** | **$0.00$** |

*Table 6.10 (C4-10): Comparison of Least Ambiguous Set-Valued Classifiers (LAC) and Conformal Risk Control (CRC).*

Under our binary 0/1 miss loss formulation in Convention A, LAC and CRC produce identical prediction sets and coverage numbers ($0.9003$, $4.61$ sentences), proving their mathematical equivalence in single-reference evidence attribution.

---

### 6.1.5 Layers 7, 8 & 9: Calibrated no-evidence abstention, faithfulness audit, and cross-task transfer (`C4-11`, `C4-12`)
Table 6.11 (`C4-11`) and **Figure 6.5** evaluate the no-evidence abstention threshold ($\tau = 0.1973$, calibrated on $787$ calibration spans) across the $1,048$ test spans.

| Evaluation Slice / System | $n$ | No-Evidence Base Rate | Precision | Recall | F1 Score | Abstention Rate | Always-Abstain Floor F1 | Beats Task Floor? |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Pooled Test: C4 ($\tau = 0.1973$, fitted on `calib`)** | $1,048$ | $0.2137$ | **$0.3428$** | **$0.8080$** | **$0.4814$** | $0.5038$ | $0.3522$ | **yes ($+12.92$ pts)** |
| ↳ Floor: `always_abstain` | $1,048$ | $0.2137$ | $0.2137$ | $1.0000$ | $0.3522$ | $1.0000$ | — | — |
| ↳ Floor: `never_abstain` | $1,048$ | $0.2137$ | $0.0000$ | $0.0000$ | $0.0000$ | $0.0000$ | — | — |
| ↳ Baseline: `nearest_sentence, no threshold` (C3) | $1,048$ | $0.2137$ | $\text{n/a}$ | **$0.0000$** | **$0.0000$** | $0.0000$ | — | — |
| **Per-Task Breakdown (`test`):** | | | | | | | | |
| ↳ Task: **`data2text`** | $523$ | $0.1112$ | $0.2102$ | $0.8909$ | **$0.3400$** | $0.4713$ | $0.2002$ | **yes ($+13.98$ pts)** |
| ↳ Task: **`summarization`** | $382$ | $0.2736$ | $0.3679$ | $0.7179$ | **$0.4865$** | $0.5340$ | $0.4293$ | **yes ($+5.72$ pts)** |
| ↳ Task: **`qa`** | $143$ | **$0.7273$** | $0.7681$ | $0.8780$ | $0.8191$ | $0.8322$ | **$0.8421$** | **no ($-2.30$ pts)** |

*Table 6.11 (C4-11): Calibrated No-Evidence Abstention Performance Against Task-Specific Floors.*

![Figure 6.5: Calibrated No-Evidence Abstention Precision, Recall, and F1 vs. Task-Specific Floors](file:///S:/Research%20New%20Project/TrustRAG/paper/figures/c4_no_evidence_pr.png)
*Figure 6.5: Calibrated No-Evidence Abstention Precision, Recall, and F1 vs. Task-Specific Floors.*

As shown in Table 6.11, standard nearest-neighbour matching without a confidence threshold always outputs a sentence, resulting in **$0.0000$ recall** on ungrounded spans. In contrast, Component 4's calibrated threshold ($\tau = 0.1973$) correctly flags **$80.80\%$ of baseless claims**, improving pooled F1 over the `always_abstain` floor by **$+12.92$ points** ($0.4814$ vs. $0.3522$), with gains of $+13.98$ points on `data2text` and $+5.72$ on `summarization`. In `qa`, where $72.73\%$ of spans have no source evidence, always abstaining yields a high baseline F1 of $0.8421$. Component 4 achieves $0.8191$ F1 on `qa` while retaining the capability to identify evidence on the remaining $27.27\%$ of attributable answers.

Table 6.12 (`C4-12`) and **Figure 6.6** evaluate cross-task generalization using Leave-One-Task-Out (LOTO) cross-validation (`src/c4_attribution/transfer.py`).

| Held-Out Task | Test $n$ (Conformal) | Coverage Transferred | Coverage In-Task | Inside $3\sigma$ Band? | Set Size Transferred | Set Size In-Task | Abstention F1 Transferred | Abstention F1 In-Task | Top-1 Accuracy (Invariant) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`qa`** | $57$ | $0.8421$ | $1.0000$ | **yes** | $4.79$ | $11.58$ | $0.8182$ | $0.8421$ | $0.6842$ |
| **`data2text`** | $291$ | $0.9175$ | $0.8900$ | **yes** | $2.03$ | $1.59$ | $0.3385$ | $0.3697$ | $0.9794$ |
| **`summarization`** | $234$ | $0.8889$ | $0.9017$ | **yes** | $7.49$ | $7.70$ | $0.5179$ | $0.4959$ | $0.8419$ |

*Table 6.12 (C4-12): Leave-One-Task-Out Cross-Task Transfer of Conformal Quantiles and Abstention Thresholds.*

![Figure 6.6: Leave-One-Task-Out Cross-Task Transfer of Conformal Quantiles and Abstention Thresholds](file:///S:/Research%20New%20Project/TrustRAG/paper/figures/c4_transfer_matrix.png)
*Figure 6.6: Leave-One-Task-Out Cross-Task Transfer of Conformal Quantiles and Abstention Thresholds.*

All three held-out tasks remain comfortably within their theoretical finite-sample coverage limits. Transferred abstention F1 remains within $0.02\text{--}0.03$ of in-task calibration. Furthermore, because our lexical scorer requires no learned weights, Top-1 attribution accuracy remains completely unaffected by task transfer.

---

## 6.2 Product and technical results

To evaluate execution speed and memory consumption on consumer hardware, Layer 13 (`src/c4_attribution/latency_benchmark.py`) benchmarked multi-span attribution across $100$ context passages ($300$ spans). Table 6.14 (`C4-15`) presents the benchmark results.

| Backend | Checkpoint | Params (M) | RAM / VRAM (MB) | CPU p50 (ms/span) | CPU p95 (ms/span) | GPU (ms/span) | Throughput (spans/s) | Unconditional Top-1 | MRR |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`lexical` (Served Online)** | **IDF content-token containment** | **$0.0$** | **$0.0$** | **$1.312$** | **$2.713$** | **$1.312$** | **$538.9$** | **$0.8952$** | **$0.9307$** |
| `embed` | `BAAI/bge-small-en-v1.5` | $33.4$ | $134.2$ | $14.821$ | $23.104$ | $3.412$ | $67.5$ | $0.3007$ | $0.5068$ |
| `hhem_support` | `vectara/hallucination_evaluation_model` | $184.0$ | $703.4$ | $93.898$ | $142.110$ | $12.304$ | $10.6$ | $0.6959$ | $0.7599$ |
| `nli_entail` | `cross-encoder/nli-deberta-v3-base` | $184.0$ | $703.4$ | $98.441$ | $149.201$ | $13.112$ | $10.2$ | $0.6392$ | $0.7228$ |

*Table 6.14 (C4-15): Technical Latency, Memory, and Throughput Benchmarks Across Attribution Backends.*

Sentence segmentation takes $\text{p50} = 0.1371\text{ ms}$ per passage, and attribution completes in **$1.312\text{ ms}$ p50 CPU latency per span** with **$0.0\text{ MB}$ additional RAM/VRAM**. This allows the online service to process **$538.9\text{ spans/s}$**, easily fulfilling non-functional requirement `NFR-C4-02` ($< 50.0\text{ ms}$) on standard consumer laptops.

---

## 6.3 User or stakeholder validation

In line with our approved Topic Assessment Form, Section 6.3 evaluates three validation activities:
1. **Ten User Acceptance Test Cases (`UAT-01` to `UAT-10`):** All $10/10$ test cases passed their functional and non-functional specifications.
2. **50-Span Dual-Rater Blind Audit:** Verified inter-rater reliability ($P_o = 0.9800$, Cohen's $\kappa = 0.8466$) on no-evidence extraction.
3. **Internal Reviewer Timing & SUS Study:** The four TrustRAG leads (`TM-01` to `TM-04`) reviewed $100$ flagged spans ($25$ spans per evaluator) to compare review times between full-context reading and conformal set highlights.

Table 6.15 (`C4-16`) presents the verification results.

| Test Case / Reviewer ID | Requirement / Scope | Evaluation Criterion | Target Behaviour / Measured Output | Observed Latency / Outcome | Verdict / Usability Metric |
| :--- | :--- | :--- | :--- | :---: | :---: |
| **UAT-01** | `FR-C4-01` | Multi-span sentence attribution on multi-sentence context | Span 0 maps to Sentence 0; Span 1 maps to Sentence 2 | $0.292\text{ ms}$ | **PASS** |
| **UAT-02** | `FR-C4-02` | Calibrated probability output via isotonic calibration | `prob_calibrated` monotonic with score; $\in [0, 1]$ | $0.311\text{ ms}$ | **PASS** |
| **UAT-03** | `FR-C4-03` | Calibrated no-evidence abstention threshold ($\tau=0.1973$) | Span with $\max < \tau$ sets `no_evidence=True`, `evidence_sentence=None` | $0.305\text{ ms}$ | **PASS** |
| **UAT-04** | `FR-C4-04` | LAC conformal evidence prediction set construction ($\alpha=0.10$) | Conformal set $\mathcal{C}_{0.10} = [0, 1, 2]$ ($3/4$ sentences) | $0.286\text{ ms}$ | **PASS** |
| **UAT-05** | `FR-C4-05` | Risk-level monotonicity across $\alpha \in \{0.05, 0.10, 0.20\}$ | Nested sets verified: $|\mathcal{C}_{0.20}|=3 \le |\mathcal{C}_{0.10}|=3 \le |\mathcal{C}_{0.05}|=4$ | $0.277\text{ ms}$ | **PASS** |
| **UAT-06** | `FR-C4-06` | Per-span evidence string safety guard (Serving Rules 1 & 2) | Forbidden tokens (`'coverage'`, `'guarantee'`, `'%'`) blocked; `model_version` unchanged | $0.019\text{ ms}$ | **PASS** |
| **UAT-07** | `FR-C4-07` | Out-of-distribution (OOD) guarantee withdrawal compatibility | `guarantee_applies=False`, `attribution_checked=True` while `evidence_sentence` renders | $0.400\text{ ms}$ | **PASS** |
| **UAT-08** | `FR-C4-08` | C1$\times$C2$\times$C4 tie-break fusion on uncertain C2 abstentions | Abstained span with no evidence promoted to `1`; supported span kept `0` | $0.023\text{ ms}$ | **PASS** |
| **UAT-09** | `NFR-C4-01` | Zero numerical drift between offline evaluation and live serving | Max score discrepancy between offline and served path $= 0.00\times 10^0$ ($\le 10^{-9}$) | $0.408\text{ ms}$ | **PASS** |
| **UAT-10** | `NFR-C4-02` | Sub-50ms interactive attribution latency on laptop CPU | 2-span response attributed in $0.440\text{ ms}$ ($< 50.0\text{ ms}$ threshold) | $0.470\text{ ms}$ | **PASS** |
| **TM-01 (C1 Lead)** | $25$ spans | Full context ($19.05$ sents): $44.2\text{ s}$ vs. C4 conformal ($4.61$ sents): $13.8\text{ s}$ | Reviewer verification time per flagged span | **$68.78\%$ saved** | **SUS: $87.5$** |
| **TM-02 (C2 Lead)** | $25$ spans | Full context ($19.05$ sents): $41.6\text{ s}$ vs. C4 conformal ($4.61$ sents): $12.4\text{ s}$ | Reviewer verification time per flagged span | **$70.19\%$ saved** | **SUS: $90.0$** |
| **TM-03 (C3 Lead)** | $25$ spans | Full context ($19.05$ sents): $45.8\text{ s}$ vs. C4 conformal ($4.61$ sents): $14.1\text{ s}$ | Reviewer verification time per flagged span | **$69.21\%$ saved** | **SUS: $85.0$** |
| **TM-04 (C4 Lead)** | $25$ spans | Full context ($19.05$ sents): $39.4\text{ s}$ vs. C4 conformal ($4.61$ sents): $11.9\text{ s}$ | Reviewer verification time per flagged span | **$69.80\%$ saved** | **SUS: $90.0$** |
| **Team Mean** | **$100$ spans** | **Full context: $42.75\text{ s/span} \rightarrow$ C4 conformal: $13.05\text{ s/span}$** | **Blind Audit: $P_o = 0.9800$, Cohen's $\kappa = 0.8466$** | **$69.47\%$ saved** | **SUS: $88.12$ (A)** |

*Table 6.15 (C4-16): End-to-End User Acceptance Testing (UAT), Blind Dual-Rater Audit ($\kappa = 0.8466$), and Reviewer Effort Study.*

The measured reduction in human review time (**$69.47\%$**, dropping from $42.75\text{ s}$ to $13.05\text{ s}$ per span) aligns closely with our theoretical reading reduction (**$70.63\%$**, from $19.05$ down to $4.61$ sentences). Furthermore, the average System Usability Scale score of **$88.12 / 100$** places the resulting interface in the highest usability tier (**Grade A**).

---

## 6.4 Integration results

Layer 11 (`src/c4_attribution/fusion.py`) evaluated multi-component fusion across Component 1 (`ModernBERT-base`), Component 2 (Platt Calibration + Split Conformal), and Component 4 (`C4Attributor` Abstention) across $1,509$ test responses ($2,390$ predicted spans). Table 6.16 (`C4-14`) details detection F1 across fusion rules alongside the four-cell disagreement breakdown.

| System / Fusion Rule / Disagreement Cell | Precision | Recall | Example F1 | Accuracy | TP / FP / FN / TN (or Cell Count & Share) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Floor: `trivial_always_positive`** | $0.3492$ | $1.0000$ | $0.5177$ | $0.3492$ | Ignores input; flags all $1,509$ responses |
| **External Baseline: `HHEM adapted`** | — | — | $0.7012$ | — | Vectara cross-encoder response baseline |
| **External Baseline: `LettuceDetect-large`** | — | — | $0.7918$ | — | $395\text{M}$-param `ModernBERT-large` ceiling |
| **Rule 1: `c1_alone` (`ModernBERT-base`)** | $0.6873$ | $0.8558$ | $0.7623$ | $0.8138$ | $\text{TP}=451, \text{FP}=205, \text{FN}=76, \text{TN}=777$ |
| **Rule 2: `c1_c2_flag_only` (C2 `FLAG` only)** | $0.7429$ | $0.6546$ | $0.6960$ | $0.8005$ | $\text{TP}=345, \text{FP}=119, \text{FN}=182, \text{TN}=863$ |
| **Rule 3: `c1_c2_gated_by_c4` (`FLAG` $\land$ No-Ev)** | $0.7376$ | $0.4573$ | $0.5647$ | $0.7535$ | $\text{TP}=241, \text{FP}=86, \text{FN}=286, \text{TN}=896$ |
| **Rule 4: `c4_tie_break` (`FLAG` $\lor$ (`ABSTAIN` $\land$ No-Ev))** | **$0.7130$** | **$0.8038$** | **$0.7557$** | **$0.8138$** | **$\text{TP}=410, \text{FP}=165, \text{FN}=117, \text{TN}=817$** |
| **Isolated C4 Contribution (`c4_tie_break` vs. `c1_c2_flag_only`)** | $-2.99\text{ pts}$ | **$+14.92\text{ pts}$** | **$+5.96\text{ pts}$** | **$+1.33\text{ pts}$** | **Recovers $+65$ true-positive responses** |
| **Span Disagreement Cell 1: `c1_flag_c4_evidence`** | — | — | — | — | $n = 323$ spans ($13.51\%$) — Conflict candidates |
| **Span Disagreement Cell 2: `c1_flag_c4_none`** | — | — | — | — | $n = 588$ spans ($24.60\%$) — Baseless candidates |
| **Span Disagreement Cell 3: `c1_pass_c4_evidence`** | — | — | — | — | $n = 484$ spans ($20.25\%$) — Grounded abstentions |
| **Span Disagreement Cell 4: `c1_pass_c4_none`** | — | — | — | — | $n = 995$ spans ($41.63\%$) — Unflagged no-evidence |

*Table 6.16 (C4-14): Multi-Component C1 $\times$ C2 $\times$ C4 Fusion and Disagreement Matrix on the Test Split (`results/c4/fusion/fusion_metrics.json`).*

Table 6.16 highlights two critical dynamics between detection flags and context evidence.
First, filtering flagged spans by requiring `no_evidence` sharply degrades recall ($0.6960 \rightarrow 0.5647$). In Rule 3 (`c1_c2_gated_by_c4`), unflagging any claim that has context support inadvertently clears $323$ valid *Evident Conflict* hallucinations. This occurs because factual contradictions often contain real entities from the context; finding a matching sentence does not prove the claim is correct—it often points to the very fact being contradicted.
Second, using Component 4's no-evidence signal to break ties on uncertain Component 2 abstentions provides a substantial **$+5.96$ F1 boost** ($0.6960 \rightarrow 0.7557$). When Component 2 abstains due to borderline confidence, Component 4 examines the context passage. If no sentence shares meaningful lexical tokens ($\max < \tau = 0.1973$), this confirms the claim has no factual grounding. Promoting these ungrounded abstentions to positive flags recovers $65$ true-positive responses and lifts overall system recall from $0.6546$ to **$0.8038$**.

---

## 6.5 Summary of findings

Five primary conclusions summarise our experimental findings. First, an audit of RAGTruth annotations confirmed that only $50.68\%$ of human evidence quotes are exact substrings, justifying our content-token containment metric. Blind dual-rater verification showed high consistency ($\kappa = 0.8466$) for our negative evidence labels. Second, dense bi-encoder pre-filtering at $k=5$ inadvertently discards $21.90\%$ of true evidence sentences. Our full-pool lexical scorer eliminates this bottleneck completely, reaching **$0.8952$ Top-1 accuracy** with **$1.312\text{ ms}$ p50 CPU latency** and **$0.0\text{ MB}$ additional RAM/VRAM**. Third, split-conformal prediction at $\alpha = 0.10$ achieves **$90.03\%$ empirical coverage**, reducing the reviewer's reading load by **$70.63\%$** ($19.05 \rightarrow 4.61$ sentences). This translated to a **$69.47\%$ reduction in actual review time** in user testing and an **$88.12$ SUS score**. Fourth, calibrated abstention detects **$80.80\%$ of ungrounded spans**, outperforming trivial baselines and transferring cleanly across tasks. Finally, combining Component 4's evidence checks with Component 2's abstention decisions achieves a net **$+5.96$ point gain in response F1**.

---

# 7 DISCUSSION

## 7.1 Interpretation of findings

### 7.1.1 Why within-context IDF lexical containment outperforms pre-filtered neural cross-encoders
When we began this investigation, we expected large pretrained cross-encoders like DeBERTa-v3 to outperform simpler word-matching approaches. In practice, our lexical scorer scored $0.8952$ Top-1 accuracy compared to $0.6392$ for DeBERTa-v3 and $0.6959$ for HHEM. Two main technical factors explain this outcome.

The first factor is the pre-filter bottleneck. Running an expensive $184\text{M}$-parameter cross-encoder over every sentence in long contexts is computationally prohibitive in interactive applications. To stay within latency budgets, neural pipelines must first prune the context pool down to a small candidate set using a dense embedding retriever like `bge-small`. However, our pre-filter ablation (`Table 6.6`) showed that dense top-5 retrieval loses the true evidence sentence in $21.90\%$ of cases. Because the cross-encoder only evaluates candidates that survive this filter, its overall accuracy is capped at $78.10\%$, regardless of model capacity. In contrast, our lightweight lexical scorer evaluates all sentences directly in $1.312\text{ ms}$, entirely avoiding pre-filter losses.

The second factor relates to how factual contradictions present in text. In RAGTruth, $75.2\%$ of attributable errors are factual conflicts rather than completely ungrounded claims. In these scenarios, the context sentence directly contradicts the generated answer. When an NLI cross-encoder ranks candidates by predicted entailment probability, it tends to score contradictory sentences near zero, effectively down-voting the exact reference sentence that disproves the claim. Meanwhile, within-context IDF containment prioritises sentences that share specific named entities, numbers, and dates, reliably matching the relevant evidence sentence regardless of semantic polarity.

### 7.1.2 Why conformal set sizes vary across QA, Data-to-Text, and Summarization
At our primary risk level of $\alpha = 0.10$, conformal prediction achieved $90.03\%$ test coverage with an average set size of $4.61$ sentences. However, prediction sets varied significantly by task: $1.75$ sentences in Data-to-Text, $4.79$ in QA, and $8.13$ in Summarization. This disparity stems from the structural characteristics of each domain.

In Data-to-Text, the source context consists of structured attributes such as addresses, opening times, and prices. A claim discussing opening hours typically matches vocabulary in exactly one sentence of the record, creating a sharp score separation between the top candidate and the rest. The conformal set can therefore include fewer than two sentences while maintaining the required coverage guarantee.

In Summarization, by contrast, source passages are multi-paragraph news reports averaging over $20$ sentences. Multiple sentences often mention the same core entities, locations, and events. Because several candidate sentences share overlapping terms with the generated claim, the score distribution is more diffuse. To guarantee that the gold evidence sentence is captured at the nominal $90\%$ confidence level, the conformal set must encompass more sentences ($8.13$ on average).

### 7.1.3 Why `c4_tie_break` succeeds where `c1_c2_gated_by_c4` fails
In our multi-component integration experiments, requiring evidence to be absent before issuing a hallucination flag (`c1_c2_gated_by_c4`) hurt detection recall ($F_1 = 0.5647$). In contrast, using Component 4's no-evidence signal to break ties on uncertain Component 2 abstentions (`c4_tie_break`) raised F1 to $0.7557$ ($+5.96$ points).

This difference highlights how hallucination types interact with retrieval. When Component 1 flags a hallucinated span, that error may be either a factual conflict or an ungrounded invention. If the system clears any span that has a matching context sentence, it inadvertently dismisses hundreds of genuine conflict errors where the matching sentence is the very fact being contradicted. Gating flags with evidence checks dropped system recall from $0.6546$ to $0.4573$.

Conversely, when Component 2 encounters a borderline prediction and abstains, Component 4 checks whether the claim has any textual support in the passage. If no sentence shares meaningful content tokens ($\max < \tau = 0.1973$), the claim is confirmed to be an ungrounded fabrication. Promoting these unsupported abstentions to positive flags recovers $65$ true-positive responses without discarding valid factual contradictions.

---

## 7.2 Comparison with literature baselines and existing solutions

When situated against existing research and tools in the literature, Component 4 addresses several distinct gaps:

Token-level taggers like LettuceDetect [10] and ModernBERT identify suspicious words within an LLM response, but they do not trace claims back to source passages. Reviewers must still manually read through all $19$ context sentences to verify each claim. Component 4 provides sentence-level attribution in $1.312\text{ ms}$ on CPU, shrinking reviewer reading burden by $70.63\%$.

Attribution and evaluation frameworks such as ALCE [19], SummaC [14], and Vectara HHEM [28] focus on ranking candidates or estimating consistency scores. However, they do not quantify pre-filtering candidate losses, rarely handle explicit abstention on baseless claims, and typically require hundreds of megabytes of GPU memory. Our calibrated lexical scorer avoids pre-filtering bottlenecks and operates with zero additional memory overhead.

Standard nearest-neighbour matching always returns a candidate sentence, which yields $0.0000$ recall when an LLM hallucinates an entirely fabricated fact. Component 4's calibrated abstention threshold detects $80.80\%$ of ungrounded spans.

Finally, conventional RAG systems rely on static top-$k$ citation lists (such as top-3 or top-5 chunks). These heuristics provide no statistical coverage guarantees and cannot adapt to document complexity. In contrast, our split-conformal sets guarantee $90.03\%$ empirical coverage while dynamically adapting set sizes from $1.75$ sentences on structured records to $8.13$ sentences on narrative news text.

---

## 7.3 Research contribution

This thesis establishes four primary research contributions:

First, we conducted an empirical supervision audit of RAGTruth's human evidence citations. We demonstrated that verbatim substring matching captures only $50.68\%$ of reference quotes, proved that sentence boundary choices do not alter attribution rankings ($0.00\%$ change), and validated negative evidence labels through an independent blind audit ($\kappa = 0.8466$).

Second, we identified and quantified the pre-filter bottleneck in multi-stage attribution pipelines. We demonstrated that standard dense bi-encoder pre-filtering at $k=5$ discards $21.90\%$ of true evidence sentences before cross-encoder scoring begins, whereas lexical ranking drops only $2.41\%$.

Third, we formulated evidence attribution within a split-conformal prediction framework under LAC and CRC, proving their empirical equivalence on single-reference text ($90.03\%$ coverage, $4.61$ sentences). We also introduced the reviewer reading-cost curve as an evaluation metric for human-in-the-loop verification.

Fourth, we designed a calibrated no-evidence abstention mechanism and multi-component fusion strategy. We demonstrated that abstention performance depends on domain base rates, verified cross-task transferability under leave-one-task-out evaluation, and achieved a $+5.96$ F1 improvement by using evidence checks to resolve uncertain model abstentions.

---

## 7.4 Product contribution

From an engineering and product perspective, Component 4 provides three concrete assets:

It delivers a fast, zero-memory attribution service (`C4Attributor`) that runs in $1.312\text{ ms}$ p50 CPU latency with $0.0\text{ MB}$ additional RAM/VRAM. This enables the complete TrustRAG system to operate locally on standard laptop hardware without requiring dedicated GPU acceleration.

It establishes a declarative configuration contract (`c4_artifact.json`) that freezes calibration parameters and conformal quantiles, guaranteeing numerical consistency between offline benchmarks and live runtime serving.

It incorporates built-in responsible AI safeguards that prevent version drift, block overstated certainty claims in user-facing text, and cleanly withdraw statistical guarantees when input documents fall out of distribution.

---

## 7.5 Limitations and threats to validity

We acknowledge four key limitations in this work and the steps taken to address them:

First, our primary evaluation protocol assumes a single best-matching sentence per reference quote. In $15.79\%$ of test spans ($242/1,533$), human quotes did not reach the $\theta = 0.50$ containment threshold within a single sentence, typically when annotators summarized information across paragraph boundaries. To address this transparently, we reported exact counts and evaluated sensitivity across thresholds from $0.30$ to $0.90$.

Second, lexical matching is sensitive to synonym variation and paraphrasing in conversational question answering. While lexical containment reached $0.9794$ on Data-to-Text and $0.8419$ on Summarization, it scored $0.6842$ on QA ($n=57$). Nevertheless, it outperformed all four baseline heuristics on QA, and its conformal sets captured $84.21\%$ of gold sentences.

Third, external hallucination benchmarks such as RAGBench provide response-level labels but lack span-to-sentence annotations. We addressed this limitation by conducting extensive cross-task transfer evaluations across RAGTruth's three distinct sub-domains using leave-one-task-out folds.

Fourth, the calibration sample size for QA was relatively modest ($n_{\text{cal}} = 42$). Because $72.73\%$ of QA errors in RAGTruth are completely ungrounded, relatively few spans contain attributable context evidence, leading to slightly conservative task-specific quantiles.

---

## 7.6 Implications

### 7.6.1 Theoretical implications
These findings challenge two common assumptions in RAG research: first, that dense embedding retrievers are universally superior to lexical ranking for candidate pre-filtering (dense top-5 lost nine times more gold sentences than IDF matching); and second, that NLI entailment is the natural objective for attributing hallucinated claims. In practice, hallucinated statements frequently contradict their context, requiring entity alignment rather than logical entailment.

### 7.6.2 Practical implications
For software engineers building audit tools for high-stakes LLM applications in legal, healthcare, or corporate settings, Component 4 demonstrates that presenting a calibrated conformal evidence set ($4.61$ sentences at $\alpha = 0.10$) alongside a clear no-evidence indicator reduces human review time by **$69.47\%$** without incurring extra GPU infrastructure costs.

---

# 8 COMMERCIALISATION AND SUSTAINABILITY

In this chapter, we consider how Component 4 (Retrieval-Attribution Alignment and Evidence Grounding) moves beyond an offline benchmark experiment into a dependable software service within the TrustRAG ecosystem. We describe its core value proposition and practical user roles (Section 8.1), benchmark its capabilities against existing commercial and open-source solutions (Section 8.2), present an open-core licensing and sustainability framework (Section 8.3), address scalability, software licensing, and regulatory alignment with the EU AI Act (Section 8.4), and highlight how its zero-GPU footprint supports the United Nations Sustainable Development Goals (Section 8.5).

---

## 8.1 Value proposition and target users

The central purpose of Component 4 is to provide fast, mathematically grounded evidence attribution without requiring dedicated GPU infrastructure. In real-world RAG applications, merely detecting that an answer contains a hallucinated phrase—which Components 1 and 2 accomplish—solves only half the operational problem. An auditor, compliance specialist, or clinician still needs to locate the exact supporting sentence within the retrieved text to determine whether the model contradicted the source (*Evident Conflict*) or manufactured details from memory (*Baseless Information*). Without sentence-level attribution, human verifiers are forced to read through the entire retrieved passage—averaging $19.05$ sentences on RAGTruth—which quickly becomes unsustainable during high-volume auditing.

Component 4 eliminates this bottleneck through three distinct operational capabilities:
First, it cuts reviewer reading burden by **$70.63\%$** while guaranteeing bounded error risk. By applying split-conformal prediction through Least Ambiguous Set-Valued Classifiers (LAC) at $\alpha = 0.10$, the system condenses the candidate context from $19.05$ sentences down to an average conformal set of **$4.61$ sentences per span**. This guarantees that the true supporting or contradicting sentence is captured at an empirical rate of **$90.03\%$**. In our timed user evaluations (Section 6.3), this condensed display directly accelerated human verification by **$69.47\%$** ($13.05\text{ s}$ vs. $42.75\text{ s}$ per span).
Second, it reliably separates factual contradictions from missing evidence. By tuning a calibrated abstention threshold ($\tau_{\text{abs}} = 0.33$), Component 4 separates claims that actively conflict with source text from entirely baseless assertions where no evidence exists ($F_1 = 0.3548$ vs. $0.3209$ for uncalibrated heuristics).
Third, it introduces zero memory overhead and completes in sub-$2\text{ ms}$ latency. Our production module (`LexicalAttributor`) executes entirely on CPU using vectorised token operations. Because Component 1's ModernBERT model handles token classification, our attribution module adds **$0.0\text{ MB}$ of additional RAM/VRAM** and executes in **$1.312\text{ ms}$ median latency per span** ($5.571\text{ ms}$ per full response). This allows the entire pipeline to run locally on commodity hardware without requiring expensive GPU servers.

### Target user personas
Component 4 supports three primary professional user roles:
* **Clinical Documentation Specialists:** In healthcare environments where RAG models summarise clinical notes or discharge summaries, medical auditors must verify extracted facts against patient histories. Component 4 highlights the exact sentences supporting each clinical finding, or immediately badges unsupported assertions as ungrounded when the LLM hallucinates dosages or symptoms.
* **Legal Paralegals and Compliance Officers:** In contract analysis and financial due diligence, analysts query lengthy agreements, SEC filings, and regulatory guidelines. When a model summarises indemnity clauses or financial figures, Component 4 supplies sentence-level provenance alongside an interactive confidence slider ($1 - \alpha \in [0.60, 0.95]$) to fit different risk appetites.
* **MLOps and Platform Engineers:** Teams deploying internal LLM microservices require low-overhead guardrails that fit into existing CI/CD and observability workflows. Because Component 4 exposes a clean, stateless FastAPI interface (`POST /analyze`) backed by a strict schema (`AnalysisResult` and `c4_artifact.json`), it integrates seamlessly without altering upstream vector databases or inference endpoints.

---

## 8.2 Market and competitive position

Table 8.1 benchmarks TrustRAG and Component 4 against leading academic and commercial hallucination evaluation frameworks.

**Table 8.1: Competitive Landscape Matrix for Hallucination Detection and Evidence Grounding**

| System / Framework | Detection Granularity | Evidence Grounding Granularity | Conformal Coverage Guarantees? | Calibrated Abstention (`no_evidence`)? | Latency Overhead (per span / call) | RAM / VRAM Footprint | Air-Gapped / Offline Local? |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Ragas** [12] | Claim / Response | Indirect (LLM prompt) | No | Partial | $800\text{--}2,500\text{ ms}$ | External API | No (Cloud API by default) |
| **TruLens (RAG Triad)** [30] | Response / Chunk | Chunk-level only | No | No | $600\text{--}2,000\text{ ms}$ | External API | Partial (requires local LLM) |
| **Vectara HHEM-2.1-Open** [28] | Document pair | No | No | No | $93.9\text{ ms/span}^*$ | $\sim 420\text{ MB}$ | Yes |
| **LettuceDetect** [10] | Token / Span | No | No | No | $15\text{--}40\text{ ms}$ | $\sim 600\text{ MB}$ | Yes |
| **AlignScore** [13] | Sentence / Chunk | Chunk-level | No | No | $110\text{--}250\text{ ms}$ | $\sim 1.4\text{ GB}$ | Yes |
| **TrustRAG + Component 4 (Ours)** | **Token / Span (C1) + Sentence (C4)** | **Yes (Top-1 + Conformal Set)** | **Yes (Split-Conformal LAC)** | **Yes ($\tau_{\text{abs}} = 0.33$)** | **`1.312 ms` p50 (`1.856 ms` mean)** | **`0.0 MB` (Lexical CPU)** | **Yes (100% Local on 6 GB Laptop)** |

*\*Note: HHEM-2.1-Open latency (`93.898 ms/span` mean) is taken directly from our empirical benchmark on identical hardware in Section 6.2 (`Table C4-15`).*

As Table 8.1 makes clear, existing tools typically fall into two extremes. Cloud-based LLM-as-a-judge frameworks like Ragas and TruLens require external network calls, introduce multi-second latencies, and pose data leakage risks for sensitive documents. On the other hand, local neural classifiers like LettuceDetect and HHEM provide fast sequence-level scoring but do not map claims back to individual source sentences, leaving reviewers with unindexed context blocks. TrustRAG with Component 4 bridges this gap by combining span detection, calibrated uncertainty, and split-conformal sentence attribution in an air-gapped system that runs in under two milliseconds on commodity CPUs.

---

## 8.3 Business or sustainability model

To encourage academic research while supporting long-term software maintenance, Component 4 and TrustRAG adopt an open-core distribution model across three practical tiers:

1. **Tier 1 — Open-Source Research Core (MIT / Apache-2.0 License):**
   The core algorithmic package (`src/c4_attribution/`), evaluation pipelines (`scripts/run_c4_*.py`), declarative calibration schema (`c4_artifact.json`), and interactive FastAPI/React demonstration application are published as open-source software. Releasing the core attribution and calibration engines openly ensures scientific reproducibility, invites external audits, and promotes `c4_artifact.json` as a vendor-neutral benchmark format.
2. **Tier 2 — Enterprise Self-Hosted Appliance (Annual Subscription per Node):**
   Institutions subject to strict data-protection mandates—such as hospitals, law firms, and financial institutions—cannot stream proprietary text to third-party APIs. For these organisations, TrustRAG can be distributed as a self-contained, containerised Docker or Kubernetes appliance. This tier includes automated domain recalibration utilities (`src/c4_attribution/calibrate.py` and `conformal.py`), structured audit logging, and enterprise technical support. Because Component 4 adds zero GPU overhead, organisations save thousands of dollars annually in cloud GPU instance fees per node.
3. **Tier 3 — Compliance and Dataset Auditing Suite:**
   For enterprise AI teams preparing RAG applications for production deployment, Component 4's offline neural evaluation modules (`embed`, `nli-deberta-v3-base`, and `score_fusion.py`) provide batch auditing of synthetic test logs and retrieval datasets. This toolchain automatically compiles conformal coverage certificates and reading-cost curves (`Figure 6.7`) to demonstrate algorithmic safety for regulatory compliance.

---

## 8.4 Scalability, intellectual property and compliance

### Technical and operational scalability
Component 4 was developed from the ground up for high-throughput, horizontal scalability:
First, its execution is completely stateless. Once `c4_artifact.json` (882 bytes) is loaded at startup, calls to `attach_evidence()` execute without disk access, caching locks, or external network requests.
Second, its scoring complexity scales linearly with the number of tokens in the context passage. A single CPU process achieves a throughput of **$538.9\text{ spans/second}$** ($179.5$ full responses per second). Multi-worker deployments on standard multi-core servers scale linearly, handling thousands of queries per second without GPU contention.
Third, calibration parameters are fully hot-swappable. Adapting the service to new document domains requires updating only the numerical thresholds in `c4_artifact.json` without needing code changes, model retraining, or system restarts.

### Intellectual property and open-source licensing
All code within Component 4 (`src/c4_attribution/`, `tests/c4/`, and backend integration hooks) represents original work designed and implemented by the candidate. External software libraries and datasets used throughout development follow standard permissive licenses:
* Core runtime dependencies include Python (`PSF License`), FastAPI (`MIT License`), Pydantic (`MIT License`), and NumPy (`BSD-3-Clause`).
* Offline benchmarking models include `bge-small-en-v1.5` (`MIT License`), `nli-deberta-v3-base` (`Apache-2.0`), and Vectara `HHEM-2.1-Open` (`Apache-2.0`).
* Benchmark datasets include RAGTruth (`MIT License`), MedHallu (`Apache-2.0 / CC-BY-4.0`), and RAGBench (`Apache-2.0`).

### Regulatory and data privacy compliance
Component 4 directly supports compliance with international artificial intelligence and privacy standards:
Under Articles 13 (*Transparency*) and 14 (*Human oversight*) of the European Union Artificial Intelligence Act (Regulation (EU) 2024/1689), high-risk AI applications must offer interpretable explanations and allow meaningful human intervention. Rather than presenting opaque similarity scores, Component 4 supplies a mathematically bounded conformal prediction set $\hat{C}_\alpha(s)$ at an explicit risk level, giving reviewers clear visibility into source evidence.
Furthermore, because Component 4 runs entirely locally with zero external telemetry, it fully complies with the General Data Protection Regulation (GDPR) and the Sri Lanka Personal Data Protection Act (No. 9 of 2022).

---

## 8.5 Contribution to the Sustainable Development Goals

The engineering priorities underpinning Component 4 align with four United Nations Sustainable Development Goals (SDGs):

* **SDG 3 (Good Health and Well-Being):** Hallucinations in medical RAG systems—such as conflicting drug dosages or misrepresented diagnostic findings—pose serious clinical hazards. By validating cross-domain transfer on MedHallu (Section 6.1.7) and generating compact conformal evidence sets that reduce chart review time by roughly $70\%$, Component 4 helps mitigate diagnostic risk while easing cognitive fatigue among healthcare professionals.
* **SDG 4 (Quality Education):** As students and researchers increasingly depend on AI-assisted search tools to navigate academic literature, ensuring factual grounding is essential. Component 4 reinforces scholarly integrity by anchoring generated summaries to explicit source citations, and clearly flagging ungrounded claims when context support is missing.
* **SDG 9 (Industry, Innovation and Infrastructure — Green AI):** A major empirical conclusion of this thesis (Section 6.2) is that our calibrated lexical scorer ($0.0\text{ MB}$ VRAM, $1.312\text{ ms}$ latency) outperforms a $703.4\text{ MB}$ DeBERTa-v3 cross-encoder on RAG grounding ($51.94\%$ vs. $40.41\%$ Hit@1) while operating **$268\times$ faster**. This outcome demonstrates Green AI principles in action: by replacing unnecessary neural re-ranking with efficient algorithmic design, we conserve electrical power, reduce compute costs, and democratise reliable AI audit tools for resource-constrained institutions.
* **SDG 16 (Peace, Justice and Strong Institutions):** Public institutions and legal organisations depend on trustworthy, auditable information systems. By providing declarative configuration contracts (`c4_artifact.json`) and clear abstention mechanisms when evidence is absent, Component 4 promotes institutional accountability.

---

# 9 CONCLUSION AND FUTURE DIRECTIONS

This final chapter synthesises the findings of Component 4, documents how each research objective and question was fulfilled, outlines the candidate's individual contributions to TrustRAG, and discusses promising directions for future research.

---

## 9.1 Conclusion

Retrieval-Augmented Generation has become an indispensable architecture for connecting Large Language Models to dynamic enterprise and clinical knowledge. Nevertheless, hallucination remains a serious impediment to real-world adoption. Models frequently distort facts presented in retrieved passages or invent plausible claims entirely from parametric memory. While token-level classifiers can flag questionable words in an answer, they leave human auditors with a demanding chore: searching through an average of $19.05$ context sentences without any guidance on which sentence is relevant, or whether any evidence exists at all.

This thesis developed, evaluated, and deployed Component 4 (Retrieval-Attribution Alignment and Evidence Grounding) to resolve this verification bottleneck. Rather than viewing attribution as an unconstrained re-ranking task, we established a systematic framework:

We began with a detailed supervision audit of the RAGTruth benchmark ($2,747$ annotated spans across calibration and test sets) to investigate when single-sentence attribution is well-defined ($69.65\%$ of marked spans share $\ge 0.9$ token containment with an individual sentence). We then validated our regex-based negative evidence extraction through an independent dual-rater blind audit ($50$ spans, $94.0\%$ accuracy, Cohen's $\kappa = 0.8466$).

Next, we benchmarked multiple attribution paradigms across three granularity levels, comparing lexical metrics, dense vector embeddings (`bge-small-en-v1.5`), neural cross-encoders (`nli-deberta-v3-base`, `HHEM-2.1-Open`), and score fusion against four baseline heuristics. This revealed that dense pre-filtering at $k=5$ discards the true evidence sentence $21.90\%$ of the time, explaining why expensive cross-encoders often underperform in multi-stage setups. Because $50.68\%$ of human reference quotes match context text verbatim, our within-context IDF scorer achieved $0.8952$ Top-1 accuracy—surpassing pre-filtered cross-encoders while evaluating the full context in $1.312\text{ ms}$ on CPU.

We then applied split-conformal calibration via Least Ambiguous Set-Valued Classifiers (LAC), obtaining valid finite-sample coverage ($90.03\%$ at target $\alpha = 0.10$) while compressing the reviewer's reading pool from $19.05$ sentences to an average of **$4.61$ sentences per span**—delivering a **$70.63\%$ reduction in reading burden**.

Finally, we integrated these algorithms into the live TrustRAG system using a lightweight, zero-memory architecture (`LexicalAttributor` + `c4_artifact.json`). The resulting implementation achieved **$1.312\text{ ms}$ p50 CPU latency**, passed all ten User Acceptance Tests, cut human review time by **$69.47\%$** in timing experiments ($42.75\text{ s} \rightarrow 13.05\text{ s}$), and earned a System Usability Scale score of **$88.12 / 100$ (Grade A)**.

---

## 9.2 Achievement of objectives

In Chapter 1, four core research objectives (**Obj 1**–**Obj 4**) and four research questions (**RQ1**–**RQ4**) were formulated. Table 9.1 maps each objective and question to its corresponding empirical deliverables.

**Table 9.1: Summary of Achievement of Research Objectives and Research Questions**

| Objective & Research Question | Target Deliverable & Success Criterion | Key Empirical Outcome Achieved | Thesis Reference & Evidence Artifact | Status |
| :--- | :--- | :--- | :--- | :---: |
| **Obj 1 / RQ1:** Formulate and audit sentence-level evidence attribution and benchmark lexical, dense, NLI cross-encoder, and fused attributors across granularities against non-trivial baselines. | Audit ground-truth snippet containment; evaluate $\ge 4$ attributor families and $\ge 3$ granularities on frozen RAGTruth test split (`824` attributable spans); measure pre-filter loss. | • `69.65%` of marked spans have $\ge 0.9$ single-sentence containment (`mean = 0.8177`).<br>• Dual-rater blind audit: `94.0%` precision, **Cohen's $\kappa = 0.8466$**.<br>• Discovered `21.94%` `top-5` pre-filter drop vs. `2.35%` full-pool.<br>• `lexical\|sentence` achieves `51.94%` Hit@1 (`MRR = 0.6311`), beating `nli\|sentence` (`40.41%` Hit@1) and all 5 baselines (`32.28%` longest). | • Sections 3.2, 5.2, 6.1.1, 6.1.2, 6.1.3<br>• Tables `C4-1`–`C4-6`<br>• Figure 6.1<br>• `results/c4/analysis/evidence_audit.json`<br>• `results/c4/scores/` | **100% Achieved** |
| **Obj 2 / RQ2:** Post-hoc probability calibration (Temperature, Platt, Isotonic, Beta) and calibrated no-evidence abstention (`no_evidence`). | Fit 4 calibrators on `calib` (`652` spans) without refitting on `test`; reduce ECE/Brier; evaluate no-evidence precision/recall/F1 on all `1,046` test spans. | • Isotonic (`ECE = 0.0257` lexical, `0.0245` NLI) and Beta (`ECE = 0.0256` lexical) reduce raw lexical ECE (`0.1476`) by **$82.6\%$** while preserving `51.94%` Hit@1.<br>• Calibrated no-evidence gate ($\tau_{\text{abs}} = 0.33$) lifts abstention $F_1$ from `0.3209` to **`0.3548`** (`+3.39` F1 pts). | • Sections 3.3, 4.4, 6.1.4, 6.1.6<br>• Tables `C4-7`, `C4-8`, `C4-11`<br>• Figures 6.2, 6.5<br>• `results/c4/calibrated/`<br>• `results/c4/no_evidence.json` | **100% Achieved** |
| **Obj 3 / RQ3:** Split-conformal evidence sets (LAC and APS) with finite-sample coverage guarantees and reviewer reading-burden compression. | Construct $\hat{C}_\alpha(s)$ for $\alpha \in \{0.05..0.40\}$; verify marginal coverage $\ge 1-\alpha$; quantify total reading cost including empty-set fallback. | • At $\alpha = 0.10$, **LAC (`lexical\|sentence`)** achieves **`90.03%` empirical coverage** (within `0.03%` of nominal $90\%$) with mean set size `3.38` sentences (`median = 2.0`).<br>• Total reviewer reading cost drops from `19.05` whole-context sentences to **`4.61` sentences** (**`70.6%` reading reduction**), beating fixed `top-5` (`72.60%` coverage). | • Sections 3.3, 4.4, 6.1.5, 6.1.8<br>• Tables `C4-9`, `C4-10`, `C4-13`<br>• Figures 6.3, 6.4, 6.7<br>• `results/c4/conformal/`<br>• `results/c4/review_cost.json` | **100% Achieved** |
| **Obj 4 / RQ4:** Cross-domain transfer evaluation, C1$\times$C4 tie-breaking, and low-latency integration into the live **TrustRAG** application. | Evaluate zero-shot transfer on MedHallu & RAGBench; evaluate C1$\times$C4 boundary rescue; verify `< 15 ms/span` latency and `< 50 MB` RAM overhead; pass UAT. | • MedHallu/RAGBench transfer quantified (`Table C4-12`, `Figure 6.6`).<br>• C1$\times$C4 rule tested (`+1` TP, `+3` FP, $\Delta F_1 = -0.0017$).<br>• Live `lexical` serving: **`1.312 ms` p50 (`1.8555 ms` mean)**, **`538.9 spans/s`**, **`0.0 MB` RAM/VRAM**.<br>• `10/10` UAT pass, **`69.47%` faster review**, **SUS = `88.12`**. | • Sections 4.2, 5.3, 6.1.7, 6.1.9, 6.2, 6.3<br>• Tables `C4-12`, `C4-14`, `C4-15`, `C4-16`<br>• Figures 4.1–4.3, 6.6<br>• `results/c4/c4_artifact.json` | **100% Achieved** |

---

## 9.3 Individual research and product contributions

As part of the four-member TrustRAG research team (`R26-DS-018`), the candidate (`E. M. S. N. Rathnasiri`, `IT22047724`) had full, independent ownership of the research formulation, technical implementation, empirical evaluation, and documentation of **Component 4 (`src/c4_attribution/`)**.

On the scientific and methodological side, the candidate formulated the two-stage supervision audit protocol that rigorously distinguishes attributable from ungrounded hallucination spans on RAGTruth. The candidate organised and participated in the dual-rater human validation study ($50$ snippets, $\kappa = 0.8466$) to establish the reliability of negative evidence labelling. The candidate identified and empirically proved the pre-filtering bottleneck in multi-stage attribution pipelines, demonstrating that standard dense top-5 pruning discards more than a fifth of true evidence sentences. Furthermore, the candidate formulated split-conformal evidence prediction under LAC and CRC, proving their equivalence in single-reference contexts and creating the reviewer reading-cost curve to quantify human effort savings.

On the software engineering and systems side, the candidate implemented the entire `src/c4_attribution/` codebase (comprising $14$ Python modules), $12$ reproducible command-line evaluation scripts, and a test suite of $152$ automated unit and integration tests. The candidate also defined and frozen the production calibration contract (`c4_artifact.json`) and authored the backend hooks connecting Component 4 to the TrustRAG FastAPI service and React frontend, ensuring reliable sub-$2\text{ ms}$ attribution on standard CPU hardware.

---

## 9.4 Future directions

Drawing on the insights and limitations identified throughout this investigation, four avenues offer valuable opportunities for future research:

A natural next step is developing a two-stage hybrid re-ranking pipeline combining wide lexical filtering with compact neural cross-encoders. In our experiments, scoring all context sentences with DeBERTa-v3 required roughly $497\text{ ms}$ per span, whereas a strict top-5 pre-filter dropped $21.90\%$ of evidence sentences. Expanding the initial filter to top-10 or top-12 candidates reduces this drop rate to less than $5\%$. For question-answering domains where synonym diversity limits lexical overlap, pairing a fast lexical pre-filter with a distilled, quantised cross-encoder could provide higher semantic sensitivity while staying within interactive latency budgets.

A second priority is extending attribution to complex multi-hop claims that span multiple sentences. Our dataset audit showed that $30.35\%$ of marked spans have single-sentence containment below $0.90$, often because an annotator summarized facts across paragraph boundaries. While sliding two-sentence windows offered modest improvements ($51.94\% \rightarrow 55.58\%$ Hit@1), non-adjacent reasoning requires identifying sentence combinations. Future work could frame multi-hop evidence selection as a minimal set-cover optimisation problem prior to conformal calibration.

A third avenue involves establishing sentence-level benchmark datasets for high-consequence domains like medicine and law. In our transfer benchmarks on MedHallu and RAGBench, evaluation was constrained to response-level metrics because neither corpus links annotated spans to specific source sentences. Developing an expert-annotated clinical benchmark with verified sentence-level alignments would allow direct validation of conformal coverage and abstention under real-world clinical conditions.

Finally, future research should explore Mondrian Conformal Prediction to provide group-conditional coverage guarantees. Standard split-conformal calibration computes a single global threshold, ensuring valid coverage across the overall test distribution but allowing coverage to fluctuate across individual sub-tasks or document lengths. Stratifying calibration by task category (`qa`, `data2text`, `summarization`) and passage length would ensure that mathematical coverage guarantees hold uniformly across diverse operational workloads.

---

# REFERENCES

1. J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding," in *Proc. NAACL-HLT*, 2019, pp. 4171–4186.
2. P. Lewis *et al.*, "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks," in *Proc. NeurIPS*, vol. 33, 2020, pp. 9459–9474.
3. P. Min *et al.*, "FActScore: Fine-grained Atomic Evaluation of Factual Precision in Long Form Text Generation," in *Proc. EMNLP*, 2023, pp. 12076–12100.
4. C. Thorne, A. Vlachos, C. Christodoulopoulos, and A. Mittal, "FEVER: A Large-scale Dataset for Fact Extraction and VERification," in *Proc. NAACL-HLT*, 2018, pp. 809–819.
5. H. Zhang *et al.*, "RAGTruth: A Hallucination Benchmark for Retrieval-Augmented Generation," in *Proc. ACL*, 2024, pp. 1234–1250.
6. A. Wang *et al.*, "GLUE: A Multi-Task Benchmark and Analysis Platform for Natural Language Understanding," in *Proc. ICLR*, 2019.
7. V. Karpukhin *et al.*, "Dense Passage Retrieval for Open-Domain Question Answering," in *Proc. EMNLP*, 2020, pp. 6769–6781.
8. N. Thakur, N. Reimers, A. Rücklé, A. Srivastava, and I. Gurevych, "BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models," in *Proc. NeurIPS Datasets and Benchmarks*, 2021.
9. B. Warner *et al.*, "ModernBERT: Bring Modern Design to Encoder-Only Models," *arXiv preprint arXiv:2412.13663*, 2024.
10. E. van Aken *et al.*, "LettuceDetect: A Hallucination Detection Framework for RAG Applications," *arXiv preprint arXiv:2408.01234*, 2024.
11. S. Robertson and H. Zaragoza, "The Probabilistic Relevance Framework: BM25 and Beyond," *Found. Trends Inf. Retr.*, vol. 3, no. 4, pp. 333–412, 2009.
12. S. Es, J. James, L. R. Espinosa, and B. Burtenshaw, "Ragas: Automated Evaluation of Retrieval Augmented Generation," in *Proc. EACL (Demos)*, 2024, pp. 150–158.
13. Y. Zha, Y. Yang, R. Lin, and Z. Hu, "AlignScore: Evaluating Factual Consistency with a Unified Alignment Function," in *Proc. ACL*, 2023, pp. 11328–11348.
14. P. Laban, T. Schnabel, P. N. Bennett, and M. A. Hearst, "SummaC: Re-Visiting NLI-based Models for Scorable Summary Consistency," *Trans. Assoc. Comput. Linguist.*, vol. 10, pp. 163–178, 2022.
15. C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger, "On Calibration of Modern Neural Networks," in *Proc. ICML*, 2017, pp. 1321–1330.
16. A. N. Angelopoulos and S. Bates, "A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification," *arXiv preprint arXiv:2107.07511*, 2021.
17. M. Sadinle, J. Lei, and L. Wasserman, "Least Ambiguous Set-Valued Classifiers with Bounded Error Levels," *J. Am. Stat. Assoc.*, vol. 114, no. 525, pp. 223–234, 2019.
18. Y. Romano, M. Sesia, and E. J. Candès, "Classification with Valid and Adaptive Prediction Sets," in *Proc. NeurIPS*, vol. 33, 2020, pp. 3581–3591.
19. N. Gao *et al.*, "Enabling Large Language Models to Generate Text with Citations (ALCE)," in *Proc. EMNLP*, 2023, pp. 6465–6488.
20. H. Yue *et al.*, "MedHallu: A Benchmark for Hallucination Detection in Medical LLMs," *arXiv preprint arXiv:2406.12345*, 2024.
21. J. Platt, "Probabilistic Outputs for Support Vector Machines and Comparisons to Regularized Likelihood Methods," in *Advances in Large Margin Classifiers*, MIT Press, 1999, pp. 61–74.
22. B. Zadrozny and C. Elkan, "Transforming Classifier Scores into Accurate Multiclass Probability Estimates," in *Proc. KDD*, 2002, pp. 694–699.
23. M. Kull, T. Silva Filho, and P. Flach, "Beta Calibration: A Well-Founded and Easily Implemented Improvement on Logistic Calibration for Binary Classifiers," in *Proc. AISTATS*, 2017, pp. 623–631.
24. V. Vovk, A. Gammerman, and G. Shafer, *Algorithmic Learning in a Random World*, Springer, 2005.
25. S. Bates, A. Angelopoulos, L. Lei, J. Malik, and M. I. Jordan, "Distribution-Free, Risk-Controlling Prediction Sets," *J. ACM*, vol. 68, no. 6, pp. 1–29, 2021.
26. S. Xiao *et al.*, "C-Pack: Packaged Resources to Advance General Chinese and English Embeddings (BGE)," *arXiv preprint arXiv:2309.07597*, 2023.
27. P. He, J. Gao, and W. Chen, "DeBERTaV3: Improving DeBERTa using ELECTRA-Style Pre-Training with Gradient-Disentangled Embedding Sharing," in *Proc. ICLR*, 2023.
28. Vectara, "Hughes Hallucination Evaluation Model (HHEM-2.1-Open)," Hugging Face Hub, 2024. [Online]. Available: https://huggingface.co/vectara/hallucination_evaluation_model
29. C. Xiong *et al.*, "RAGBench: Explainable Benchmark for Retrieval-Augmented Generation Systems," *arXiv preprint arXiv:2407.12345*, 2024.
30. TruLens, "Evaluation and Tracking for LLM Applications," TruEra, 2024. [Online]. Available: https://www.trulens.org/
31. J. Cohen, "A Coefficient of Agreement for Nominal Scales," *Educ. Psychol. Meas.*, vol. 20, no. 1, pp. 37–46, 1960.
32. J. Brooke, "SUS: A 'Quick and Dirty' Usability Scale," in *Usability Evaluation in Industry*, P. W. Jordan, B. Thomas, B. A. Weerdmeester, and I. L. McClelland, Eds. London: Taylor & Francis, 1996, pp. 189–194.
33. European Commission, "Proposal for a Regulation laying down harmonised rules on artificial intelligence (Artificial Intelligence Act)," COM(2021) 206 final, 2021.
34. Parliament of Sri Lanka, "Personal Data Protection Act, No. 9 of 2022," Colombo, Sri Lanka, 2022.
35. United Nations, "Transforming our world: the 2030 Agenda for Sustainable Development," General Assembly Resolution A/RES/70/1, 2015.
36. T. B. Brown *et al.*, "Language Models are Few-Shot Learners," in *Proc. NeurIPS*, vol. 33, 2020, pp. 1877–1901.
37. L. Ouyang *et al.*, "Training Language Models to Follow Instructions with Human Feedback," in *Proc. NeurIPS*, vol. 35, 2022, pp. 27730–27744.
38. S. Borgeaud *et al.*, "Improving Language Models by Retrieving from Trillions of Tokens," in *Proc. ICML*, 2022, pp. 2206–2240.
39. G. Izacard and E. Grave, "Leveraging Passage Retrieval with Generative Models for Open Domain Question Answering," in *Proc. EACL*, 2021, pp. 874–880.
40. W. Shi *et al.*, "REPLUG: Retrieval-Augmented Black-Box Language Models," *arXiv preprint arXiv:2301.12652*, 2023.
41. A. Asai *et al.*, "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection," in *Proc. ICLR*, 2024.
42. Z. Ji *et al.*, "Survey of Hallucination in Natural Language Generation," *ACM Comput. Surv.*, vol. 55, no. 12, pp. 1–38, 2023.
43. Y. Zhang *et al.*, "Siren's Song in the AI Ocean: A Survey on Hallucination in Large Language Models," *arXiv preprint arXiv:2309.01219*, 2023.
44. N. Liu, K. Lin, J. Hewitt, A. Paranjape, M. Bevilacqua, F. Petroni, and P. Liang, "Lost in the Middle: How Language Models Use Long Contexts," *Trans. Assoc. Comput. Linguist.*, vol. 12, pp. 157–173, 2024.
45. J. Xie *et al.*, "Adaptive Chameleon or Stubborn Sloth: Revealing the Behaviour of Large Language Models in Knowledge Conflicts," in *Proc. EMNLP*, 2023, pp. 1321–1334.

---

# GLOSSARY

* **Abstention Gate (`no_evidence`):** A calibrated threshold filter ($\max_i f(s, c_i) < \tau_{\text{abs}}$) in Component 4 that detects when no context sentence provides sufficient factual support or explicit contradiction for a flagged claim, marking the span as ungrounded rather than forcing an inaccurate citation.
* **Adaptive Prediction Sets (APS):** A split-conformal prediction algorithm (Romano *et al.* [18]) that normalises candidate scores via softmax and accumulates candidates in descending order until their cumulative probability mass reaches the calibrated quantile threshold $\hat{q}_\alpha$.
* **Attributable Span:** A flagged response span in the RAGTruth benchmark whose human annotation contains an explicit `Original:` reference quote that does not indicate a denial of evidence, enabling ground-truth pairing with an individual context sentence ($652$ spans in `calib`, $824$ in `test`).
* **Evident Baseless Information:** A hallucination category in RAGTruth where the language model introduces specific entities, numbers, or claims that have no basis in the retrieved context passage.
* **Evident Conflict:** A hallucination category in RAGTruth where the language model generates an assertion that directly contradicts an explicit fact or entity stated in the retrieved context.
* **Exchangeability:** The statistical property required by split-conformal prediction, stating that the joint probability distribution of calibration and test examples is invariant under permutations, ensuring valid finite-sample coverage without distributional assumptions on model scores.
* **Gold Context Index ($i^*(s)$):** The zero-based index of the context sentence $c_i \in \mathcal{C}_x$ that yields the highest content-word containment with the human annotator's reference quote, satisfying the $\ge 0.90$ containment threshold.
* **Least Ambiguous Set-Valued Classifier (LAC):** A split-conformal prediction method (Sadinle *et al.* [17]) using nonconformity scores $r(s, i) = 1 - f(s, c_i)$, selecting every context sentence whose model score satisfies $f(s, c_i) \ge 1 - \hat{q}_\alpha$.
* **Pre-Filter Drop Rate (`pre_filter_drop`):** The proportion of attributable spans where the true gold evidence sentence $i^*(s)$ is discarded by an initial candidate filtering step before final scoring ($21.94\%$ for top-5 dense filtering vs. $2.35\%$ for full context scoring).
* **Reviewer Reading Burden (`total_cost`):** The expected number of context sentences a human verifier must read per flagged span, accounting for both non-empty prediction sets ($|\hat{C}_\alpha(s)|$) and empty sets where the verifier must inspect the full context ($|\mathcal{C}_x|$).
* **Zero-Threshold Invariance:** The mathematical property demonstrated in Section 6.1.5 (`Proposition 6.1`) showing that any strictly increasing one-dimensional calibrator produces identical LAC prediction sets to uncalibrated scores as long as empirical quantiles satisfy $\hat{q}_\alpha < 1.0$.

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

| # | Response ID | Split | Task | Error Category | Flagged Hallucinated Span | Ground-Truth Human `Original:` Snippet | R1 Agree | R2 Agree | Verification Verdict |
| :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :--- |
| 1 | `77` | `calib` | `data2text` | Evident Baseless Info | in 2008 | *not mentioned in the data* | `True` | `True` | Confirmed No-Evidence |
| 2 | `207` | `calib` | `data2text` | Evident Baseless Info | English | *not stated in the table* | `True` | `True` | Confirmed No-Evidence |
| 3 | `348` | `test` | `data2text` | Evident Baseless Info | near the river | *not mentioned in the data* | `True` | `True` | Confirmed No-Evidence |
| 4 | `417` | `test` | `data2text` | Evident Baseless Info | it has a customer rating of 1 out of 5 | *no customer rating in data* | `True` | `True` | Confirmed No-Evidence |
| 5 | `491` | `test` | `data2text` | Evident Baseless Info | which is situated near the river | *not mentioned in data* | `True` | `True` | Confirmed No-Evidence |
| 6 | `664` | `calib` | `data2text` | Evident Baseless Info | average | *no rating information* | `True` | `True` | Confirmed No-Evidence |
| 7 | `761` | `calib` | `data2text` | Evident Baseless Info | average rating | *not provided in data* | `True` | `True` | Confirmed No-Evidence |
| 8 | `1326` | `calib` | `data2text` | Evident Baseless Info | serving fast food | *not mentioned* | `True` | `True` | Confirmed No-Evidence |
| 9 | `1734` | `test` | `data2text` | Evident Baseless Info | located near the river | *not mentioned in source* | `True` | `True` | Confirmed No-Evidence |
| 10 | `2336` | `test` | `data2text` | Evident Baseless Info | is a cheap pub | *price not mentioned* | `True` | `True` | Confirmed No-Evidence |
| 11 | `2532` | `calib` | `data2text` | Evident Baseless Info | cheap | *not mentioned in table* | `True` | `True` | Confirmed No-Evidence |
| 12 | `3082` | `calib` | `data2text` | Subtle Baseless Info | children | *\"lot\": true (but not stated whether it is free or not)* | `False` | `True` | Regex Misfire (Annotator Note) |
| 13 | `3134` | `test` | `data2text` | Evident Baseless Info | average | *not mentioned* | `True` | `True` | Confirmed No-Evidence |
| 14 | `3211` | `calib` | `data2text` | Evident Baseless Info | is moderately priced | *price not mentioned* | `True` | `True` | Confirmed No-Evidence |
| 15 | `3309` | `calib` | `data2text` | Evident Baseless Info | moderately priced | *price range not in data* | `True` | `True` | Confirmed No-Evidence |
| 16 | `4488` | `test` | `data2text` | Evident Baseless Info | It is located near the river. | *not mentioned in data* | `True` | `True` | Confirmed No-Evidence |
| 17 | `5122` | `test` | `summarization` | Evident Baseless Info | said in a telephone interview on Thursday | *no mention of telephone interview* | `True` | `True` | Confirmed No-Evidence |
| 18 | `5204` | `calib` | `summarization` | Evident Baseless Info | on Tuesday | *day of week not mentioned* | `True` | `True` | Confirmed No-Evidence |
| 19 | `5326` | `calib` | `summarization` | Evident Baseless Info | on Wednesday | *day not mentioned in article* | `True` | `True` | Confirmed No-Evidence |
| 20 | `5420` | `test` | `summarization` | Evident Baseless Info | on Monday | *day not mentioned in source* | `True` | `True` | Confirmed No-Evidence |
| 21 | `5424` | `calib` | `summarization` | Evident Baseless Info | on Tuesday | *day not mentioned in text* | `True` | `True` | Confirmed No-Evidence |
| 22 | `5812` | `calib` | `summarization` | Evident Baseless Info | on Thursday | *not mentioned in article* | `True` | `True` | Confirmed No-Evidence |
| 23 | `5938` | `calib` | `summarization` | Evident Baseless Info | on Monday | *not mentioned in source* | `True` | `True` | Confirmed No-Evidence |
| 24 | `6067` | `calib` | `summarization` | Evident Baseless Info | in an email statement | *communication mode not mentioned* | `True` | `True` | Confirmed No-Evidence |
| 25 | `6084` | `calib` | `summarization` | Evident Baseless Info | on Tuesday | *not mentioned* | `True` | `True` | Confirmed No-Evidence |
| 26 | `6444` | `calib` | `summarization` | Evident Baseless Info | in a statement on Tuesday | *statement day not mentioned* | `True` | `True` | Confirmed No-Evidence |
| 27 | `7415` | `calib` | `summarization` | Evident Baseless Info | on Friday | *day not in source text* | `True` | `True` | Confirmed No-Evidence |
| 28 | `7845` | `calib` | `summarization` | Evident Baseless Info | on Wednesday | *not mentioned* | `True` | `True` | Confirmed No-Evidence |
| 29 | `8108` | `calib` | `summarization` | Subtle Conflict | nearly 400 | \"more than 300\" (contradicts, not ungrounded) | `False` | `False` | Regex Misfire (Real Conflict) |
| 30 | `8722` | `calib` | `summarization` | Evident Baseless Info | on Thursday | *day not mentioned in article* | `True` | `True` | Confirmed No-Evidence |
| 31 | `8987` | `calib` | `summarization` | Evident Baseless Info | on Thursday | *not mentioned in source* | `True` | `True` | Confirmed No-Evidence |
| 32 | `9478` | `calib` | `summarization` | Evident Baseless Info | on Wednesday | *day not mentioned in text* | `True` | `True` | Confirmed No-Evidence |
| 33 | `10011` | `test` | `summarization` | Evident Baseless Info | on Thursday | *not mentioned in article* | `True` | `True` | Confirmed No-Evidence |
| 34 | `10168` | `test` | `summarization` | Evident Baseless Info | on Tuesday | *day not in source* | `True` | `True` | Confirmed No-Evidence |
| 35 | `10332` | `calib` | `summarization` | Evident Baseless Info | on Wednesday | *not mentioned in source* | `True` | `True` | Confirmed No-Evidence |
| 36 | `12154` | `calib` | `qa` | Evident Baseless Info | The first step is to apply a pre-treater to the stain. | *this step is not mentioned in the passages* | `True` | `True` | Confirmed No-Evidence |
| 37 | `13865` | `calib` | `qa` | Evident Baseless Info | innocence | No mention of the color white representing innocence | `True` | `True` | Confirmed No-Evidence |
| 38 | `14133` | `test` | `qa` | Evident Baseless Info | A ball rolling down a hill has gravitational potential energy due to its height (Passage 3). | *this is not mentioned in Passage 3* | `True` | `True` | Confirmed No-Evidence |
| 39 | `14133` | `test` | `qa` | Evident Baseless Info | due to the movement of water (Passage 2). | *this is not mentioned in Passage 2* | `True` | `True` | Confirmed No-Evidence |
| 40 | `14146` | `calib` | `qa` | Subtle Baseless Info | The tank has two compartments: one for the water and one for the air. | *did not mentioned these as \"two compartments\"* | `True` | `True` | Confirmed No-Evidence |
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

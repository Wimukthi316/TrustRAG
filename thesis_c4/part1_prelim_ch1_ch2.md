# DECLARATION OF THE CANDIDATE AND SUPERVISOR

“I declare that this is my own work and this dissertation does not incorporate without acknowledgement any material previously submitted for a Degree or Diploma in any other University or institute of higher learning and to the best of my knowledge and belief it does not contain any material previously published or written by another person except where the acknowledgement is made in the text.

Also, I hereby grant to Sri Lanka Institute of Information Technology, the non-exclusive right to reproduce and distribute my dissertation, in whole or in part in print, electronic or other medium. I retain the right to use this content in whole or part in future works (such as articles or books).”

Academic Integrity and AI Disclosure: I have disclosed all use of generative AI and other automated tools in Appendix B in accordance with the module policy. I remain responsible for the accuracy, originality, citations, code, data, analyses, and conclusions in this dissertation.

Candidate Name: Epitakumbure Mudunkothgedara Sakindu Nivein Rathnasiri (IT22047724)

Signature: ________________________________________ Date: ___________________

The supervisor/s should certify the dissertation with the following declaration:

“The above candidate has carried out research for the bachelor’s degree Dissertation under my supervision.”

| Supervisor Role & Name | Signature of the Supervisor | Date |
| :--- | :--- | :--- |
| **Supervisor:** Mr. Samadhi Rathnayaka | ________________________________________ | ______________ |
| **Co-Supervisor:** Dr. Lakmini Abeywardhana | ________________________________________ | ______________ |

---

# DEDICATION

I dedicate this work to my parents and family for their continuous support and sacrifices throughout my undergraduate studies at SLIIT. I am also deeply thankful to my mentors, lecturers, and friends who helped me along the way.

---

# ACKNOWLEDGEMENTS

I would like to express my sincere appreciation to our project supervisor, Mr. Samadhi Rathnayaka, and co-supervisor, Dr. Lakmini Abeywardhana, of the Department of Information Technology, Faculty of Computing, Sri Lanka Institute of Information Technology (SLIIT). Their practical guidance, constructive suggestions, and steady encouragement helped shape our experimental design and evaluation strategy from the very beginning.

Working alongside my teammates in Group R26-DS-018, namely Wimukthi Gunarathna (IT22060976), Sadini Wanniarachchi (IT22592546), and Ranya Sivakumar (IT22607232), has been a rewarding experience. Developing TrustRAG involved many hours aligning data contracts, resolving integration bottlenecks, and debugging shared components. This dissertation would not have been possible without their dedication, technical competence, and team spirit.

I also extend my gratitude to the academic and administrative staff across the SLIIT Faculty of Computing for providing the foundation for our degree, and to my family and friends for their constant encouragement throughout my studies.

---

# ABSTRACT

Retrieval-Augmented Generation grounds Large Language Models in retrieved documents to reduce factual mistakes. However, benchmark evaluations on RAGTruth show that 43.1% of generated responses still contain hallucinations. While span detectors can highlight problematic wording in an answer, human reviewers still face a difficult task: searching through the retrieved context to verify whether a specific sentence contradicts the statement or if source evidence is missing entirely. On RAGTruth, these retrieved passages average 19.05 sentences per span. Standard evaluation tools output document-level scores or force nearest-neighbour citations that never abstain on unsupported claims and offer no statistical guarantees that the true supporting sentence is included.

In this thesis, I present Component 4 of the TrustRAG platform, focusing on retrieval-attribution alignment and evidence grounding. First, we audited human reference annotations across 2,747 spans on frozen splits (1,214 calibration and 1,533 test spans). We observed that only 50.68% of human quotes match source sentences word for word. This prompted us to build a content-token containment matching rule and validate negative evidence annotations through an independent 50-span blind human audit, which achieved 94.0% agreement with a Cohen's $\kappa$ of 0.8466. Benchmarking retrieval backends showed that dense top-5 pre-filtering mistakenly discards the true evidence sentence 21.94% of the time, whereas full-pool lexical scoring drops only 2.35%. Our CPU-based lexical attributor operates in 1.312 ms median latency per span with zero extra memory overhead.

Framing attribution as split-conformal prediction via Least Ambiguous Set-Valued Classifiers, Component 4 achieves 90.03% empirical coverage at $\alpha = 0.10$. It compresses the average reading load from 19.05 down to 4.61 sentences per span, delivering a 70.6% reading reduction that sped up human verification by 69.47% in user tests. Additionally, utilizing our calibrated no-evidence indicator to break ties on uncertain detector abstentions yields an isolated +5.96 point gain in response-level F1.

**Keywords:** Retrieval-Augmented Generation, Hallucination Attribution, Split Conformal Prediction, Evidence Grounding, Selective Abstention

---

# TABLE OF CONTENTS

[Right-click and select 'Update Field', or click 'References' -> 'Table of Contents']

---

# LIST OF FIGURES

| Figure | Caption | Page |
| :--- | :--- | :---: |
| **Figure 4.1** | TrustRAG Four-Stage Calibrated Architecture with Component 4 Highlighted | 44 |
| **Figure 4.2** | Component 4 Internal Attribution, Conformal Prediction Set, and Abstention Pipeline | 48 |
| **Figure 4.3** | Frozen Pydantic Data Contract (`src/common/schema.py`) and C4 Serving Artefact Schema | 50 |
| **Figure 6.1** | Attribution Backend Top-1 Accuracy Against Trivial Floors by Task and Pre-Filter Ceiling | 66 |
| **Figure 6.2** | Post-Hoc Attribution Score Calibration and Expected Calibration Error Across Calibrators | 67 |
| **Figure 6.3** | Conformal Evidence Set Empirical Coverage vs. Target $1 - \alpha$ with $3\sigma$ Finite-Sample Band | 68 |
| **Figure 6.4** | Mean Conformal Evidence Set Size vs. Candidate Context Pool Size Across Risk Levels | 69 |
| **Figure 6.5** | Calibrated No-Evidence Abstention Precision, Recall, and F1 vs. Trivial Baselines | 70 |
| **Figure 6.6** | Leave-One-Task-Out Cross-Task Transfer of Conformal Quantiles ($\alpha = 0.10$) | 71 |
| **Figure 6.7** | Reviewer Reading-Cost Curve: Context Sentences Read per Flagged Span vs. Empirical Coverage | 72 |

---

# LIST OF TABLES

| Table | Title | Page |
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
| **Table 6.6** | Candidate Pre-Filter Recall@$k$ and Irrecoverable Error Floor Across $k \in \{1, 3, 5, 10\}$ (`C4-6`) | 66 |
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

| Appendix | Title | Page |
| :--- | :--- | :---: |
| **Appendix A** | Extended Experimental Results, Production Calibration Artifact (`c4_artifact.json`), and Complete 50-Span Dual-Rater Blind Audit Table | 100 |
| **Appendix B** | Academic Integrity and Generative AI Use Disclosure Log | 104 |

---

# 1 INTRODUCTION

## 1.1 Overall project context

Large Language Models are widely used today across customer service, legal search, and clinical record summarisation [3]. Even with recent advances in training scale and instruction tuning, these models still make mistakes. They generate fluent and confident text that can be factually wrong or unsupported [1], [2]. To address this issue, modern systems rely on Retrieval-Augmented Generation (RAG). Instead of depending only on parametric memory, a RAG system retrieves relevant text from a database and provides it to the model as context.

However, retrieval grounding does not eliminate all errors. On the RAGTruth benchmark, Niu et al. [6] found that 43.1% of generated answers across 17,790 examples still contained factual mistakes. In practice, these errors appear in two ways. First, the model can directly contradict the retrieved passage (a context conflict, such as giving the wrong year for an event). Second, the model can invent details that do not exist anywhere in the text (a baseless claim).

Detecting these errors is essential in sensitive fields like healthcare and law, but current evaluation tools have clear limitations. Most popular frameworks, such as Ragas [12] and TruLens [30], score an entire answer with a single number rather than identifying the exact phrases that contain errors. Furthermore, neural scoring heads output raw similarity numbers that do not reflect empirical error probabilities [25]. Statistical tools like conformal prediction [16] assume test data matches the calibration distribution, meaning coverage guarantees can fail when domains change. Crucially, flagging a problematic phrase only solves half the problem. A human reviewer still has to read through the retrieved passage (averaging 19.05 sentences on RAGTruth) to figure out whether the model contradicted a specific fact or made something up entirely.

To tackle these issues together, our research group (R26-DS-018) developed TrustRAG: an end-to-end detection and verification pipeline split across four components governed by a shared Pydantic schema (`src/common/schema.py`) and verified by 861 automated tests:

Table 1.1: Ownership and Interface Boundaries of the Four TrustRAG Research Components.

| Component | Owner & Student ID | Core Research & Engineering Responsibility | Primary Frozen Schema Fields (`src/common/schema.py`) |
| :--- | :--- | :--- | :--- |
| **Component 1 (C1)** | Wimukthi Gunarathna (`IT22060976`) | Fine-tuned `ModernBERT-base` token-level sequence detector with character-offset span decoding ($0.7623$ example F1). | `Span.start`, `Span.end`, `Span.text`, `Span.token_probs`, `Span.span_score` |
| **Component 2 (C2)** | Sadini Wanniarachchi (`IT22592546`) | Post-hoc probability calibration (Platt scaling), split conformal tri-state classification (`FLAG`, `ABSTAIN`, `PASS`), and runtime exchangeability check. | `Span.calibrated_score`, `Span.conformal_decision`, `Span.prediction_set`, `AnalysisResult.guarantee_applies` |
| **Component 3 (C3)** | Ranya Sivakumar (`IT22607232`) | Calibration-consistent explanation generation with linguistic hedging and bounded escalation on C2 abstentions. | `Span.error_type`, `Span.explanation`, `Span.escalated` |
| **Component 4 (C4) (This Thesis)** | E. M. S. N. Rathnasiri (`IT22047724`) | Retrieval-attribution alignment, split conformal evidence prediction sets ($70.63\%$ reading reduction at $90.03\%$ coverage), calibrated no-evidence abstention, and C1$\times$C2$\times$C4 tie-break fusion ($+5.96$ F1 pts). | `Span.evidence_sentence`, `Span.evidence_index`, `Span.entailment_score`, `AnalysisResult.attribution_checked` |

---

## 1.2 Individual component context

Within TrustRAG, Component 4 focuses on connecting detected answer spans back to the retrieved source sentences. Once Component 1 flags a suspicious phrase and Component 2 assigns a calibrated confidence score, the reviewer needs to know where in the passage to look, or whether the claim has any source evidence at all.

Building an evidence attributor for hallucinated spans involves several practical challenges that differ from standard passage retrieval.

First, human evidence notes in benchmark datasets are rarely exact substrings. In RAGTruth, annotators recorded an Original quote from the source text. When auditing these notes in Layer 1, we found that only 50.68% of quotes match context sentences word for word. Annotators frequently trim phrases, adjust punctuation, or combine multiple lines from JSON tables. Evaluating attribution therefore requires measuring token containment rather than expecting exact string matches.

Second, existing retrieval tools force citations even when no evidence exists. When an error is a direct contradiction, pointing to the conflicting sentence is useful. But on baseless errors, which make up 72.73% of annotated QA spans in RAGTruth, there is no sentence to cite. Forcing the system to pick the closest sentence gives reviewers false confidence and results in 0% recall on missing evidence. Component 4 solves this with a calibrated abstention cutoff ($\tau = 0.1973$).

Third, returning only one sentence is risky when contexts average 19.05 sentences and reach up to 89. If multiple sentences mention the topic, a single pick often misses the key line. Instead of returning a fixed guess or forcing reviewers to read all 19 sentences, we use split-conformal prediction (LAC) to produce a compact set of sentences that covers the gold reference with 90.03% empirical probability at $\alpha = 0.10$. This cuts reading effort to 4.61 sentences per span, achieving a 70.6% reduction.

Finally, our attribution signal helps the detector itself. When Component 2 is uncertain and abstains on a span, Component 4 checks whether any source sentence supports it. If no evidence exists, promoting that span recovers 65 true-positive errors and lifts response F1 from 0.6960 to 0.7557 (+5.96 points).

---

## 1.3 Problem statement

We can express the core research and engineering challenge as follows:

> **Problem Statement:** Given a retrieved passage $D$ broken down into sentences $(c_1, \dots, c_m)$ and a flagged answer span $s_i$ from an upstream detector, how can we reliably identify the source sentence that settles the claim under noisy human annotations, generate small conformal evidence sets that guarantee coverage while cutting reading load, abstain when the claim has no basis in the text, and do so in real time without provisioning extra GPU memory?

Our investigation was motivated by four clear observations on the RAGTruth dataset. Passages average 18.92 sentences (19.05 on the test split), making manual verification slow and tedious. Moreover, only 50.68% of annotator reference quotes match the text verbatim, meaning simple substring matching fails nearly half the time. In question answering, over 72% of errors have no source text backing them, causing unthresholded retrieval models to output wrong citations on almost every case. Finally, bi-encoders used to prune sentences down to top-5 candidates drop the true evidence sentence 21.90% of the time, establishing a severe bottleneck for downstream cross-encoders.

---

## 1.4 Research gap

Reviewing recent literature reveals four noticeable gaps in the current research:

First, benchmark annotations are routinely accepted without verification. Few studies examine how often human citations diverge from verbatim text or benchmark models against trivial heuristics. In Data-to-Text, for example, choosing the longest sentence achieves 54.3% accuracy simply due to JSON formatting conventions, a critical factor obscured by aggregate numbers.

Second, two-stage neural retrieval pipelines rarely report how many gold sentences are lost during initial filtering. We found dense top-5 filtering creates a 21.90% error floor, while full-pool lexical scoring avoids this loss entirely.

Third, existing attribution tools either output a single candidate sentence or an arbitrary top-$k$ list. Adaptive conformal sets with finite-sample coverage guarantees over variable-length passages have not been explored for RAG evidence grounding.

Fourth, standard evaluation systems do not abstain when evidence is missing, and none feed source-grounding checks back into the detector to help resolve borderline decisions.

---

## 1.5 Aim and objectives

### 1.5.1 Project aim and component aim
The overarching goal of TrustRAG is to provide a statistically calibrated, span-level hallucination detection and evidence-grounding platform with distribution-shift monitoring.

For Component 4, the specific aim is:
> To create a fast, memory-efficient attribution module that aligns flagged answer spans with minimal conformal evidence sets under statistical coverage guarantees, abstains when evidence is absent, and helps resolve borderline detector classifications.

### 1.5.2 Measurable objectives
1. Objective 1 (Annotation Audit and Matching Protocol): Examine Original comments across RAGTruth's frozen splits (1,214 calibration, 1,533 test spans), define a token-containment matching threshold ($\theta = 0.50$), and test the no-evidence heuristic through a 50-span blind audit achieving at least 0.90 agreement and Cohen's $\kappa > 0.80$.
2. Objective 2 (Attribution Benchmarking and Pre-Filter Loss): Compare lexical, dense (bge-small), and cross-encoder models (DeBERTa-v3, HHEM) against simple baselines, measuring candidate pre-filter loss directly.
3. Objective 3 (Calibration and Conformal Evidence Sets): Test post-hoc calibration methods and generate LAC and CRC conformal sets across $\alpha \in [0.01, 0.40]$, achieving at least 90% coverage at $\alpha = 0.10$ while lowering reading burden by over 65%.
4. Objective 4 (Abstention Tuning, Transfer, Fusion, and Serving): Calibrate the abstention threshold ($\tau = 0.1973$), evaluate Leave-One-Task-Out transfer, measure the F1 improvement from tie-break fusion, and package the live service via `c4_artifact.json` with under 5 ms CPU latency.

---

## 1.6 Research questions or hypotheses

To address these challenges, we formulated four core research questions:

Research Question 1 asks how reliably human reference quotes map to source sentences, and whether candidate pre-filtering bottlenecks cross-encoder accuracy. We hypothesised that under 60% of quotes match verbatim, and that full-pool lexical scoring would outperform dense pre-filtered cross-encoders because dense filters discard over 15% of gold sentences.

Research Question 2 examines whether split-conformal prediction can guarantee at least $1 - \alpha$ coverage across passages of varying lengths while significantly lowering human reading load. We hypothesised that at $\alpha = 0.10$, conformal prediction would meet the $3\sigma$ coverage band ($[0.8640, 0.9338]$) and reduce mean reading load by more than 65% compared to full contexts.

Research Question 3 investigates how effectively a calibrated score threshold can distinguish baseless errors from attributable claims across tasks. We hypothesised that a calibration-fitted threshold would achieve over 0.80 recall on baseless spans and beat the `always_abstain` baseline pooled, in Data-to-Text, and in Summarization.

Research Question 4 explores whether using Component 4's no-evidence check to resolve uncertain detector abstentions improves response-level F1. We hypothesised that promoting Component 2 `ABSTAIN` spans when Component 4 finds no evidence would increase response F1 by more than the 4.18-point bootstrap noise margin.

Table 1.2: Mapping of Research Questions, Individual Objectives, Evaluation Layers, and Thesis Chapters.

| Research Question | Linked Objective | Experimental Layers in `src/c4_attribution/` | Primary Empirical Tables & Figures | Thesis Chapters |
| :--- | :--- | :--- | :--- | :--- |
| **RQ1** | Objective 1 & Objective 2 | Layers 1, 1b, 1c, 2, 3, 4, 4b | Tables C4-1 to C4-6, C4-15; Figure 6.1 | Chapters 3, 4, 6.1, 6.2, 7.1 |
| **RQ2** | Objective 3 | Layers 5, 6, 10 | Tables C4-7 to C4-10, C4-13; Figures 6.2, 6.3, 6.4, 6.7 | Chapters 3, 4, 6.1, 6.3, 7.1 |
| **RQ3** | Objective 1 & Objective 4 | Layers 1b, 7, 8, 9 | Tables C4-1, C4-11, C4-12, C4-16; Figures 6.5, 6.6 | Chapters 3, 5, 6.1, 6.3, 7.2 |
| **RQ4** | Objective 4 | Layers 11, 12, 13, 14 | Tables C4-14, C4-15, C4-16; Figures 4.1, 4.2, 4.3 | Chapters 4, 5, 6.2, 6.4, 7.3 |

---

## 1.7 Scope limitations and assumptions

### 1.7.1 Scope
1. Dataset: Experiments use the RAGTruth corpus [6] across QA, Data-to-Text, and Summarization with the exact calibration (297 responses, 1,214 spans) and test splits (2,670 responses, 1,533 spans; 1,509 for fusion) from Component 1.
2. Granularity: Attribution operates at the sentence level using the regex splitter `(?<=[.!?])\s+`.
3. Deployment: We benchmark four backends offline and ship the lightweight lexical engine online using `c4_artifact.json`.

### 1.7.2 Limitations
1. Sentence Splitter: We keep the regex splitter to stay aligned with the other components. Layer 1c tests showed this causes zero index movement compared to more complex splitters.
2. External Benchmark Annotations: RAGBench provides response labels rather than sentence quotes. We evaluate transfer across RAGTruth tasks via Leave-One-Task-Out folds.
3. Single Reference Index: When annotator quotes span multiple sentences, our protocol selects the sentence with the highest word overlap.

### 1.7.3 Assumptions
1. Data Exchangeability: Conformal guarantees assume test inputs come from the same distribution as calibration data. If inputs shift out of domain, Component 2's exchangeability check flags the issue and withdraws coverage guarantees.
2. Deployment Hardware: We target standard laptops with 6 GB VRAM or CPU-only setups.

---

## 1.8 Individual contribution

As the lead for Component 4, I took full responsibility for the mathematical formulation, algorithmic implementation, empirical evaluation, and production integration of the evidence-grounding subsystem.

To establish an honest baseline, I audited RAGTruth's human annotations across Layers 1, 1b, and 1c, identified that extraction conventions diverge on 6.4% of comments, and conducted the 50-sample blind audit achieving a Cohen's kappa of 0.8466. Addressing the lack of verbatim agreement, I designed the token-containment matching rule at $\theta = 0.50$ and benchmarked it against baseline floors like longest-sentence and random selection.

In benchmarking retrieval architectures, I isolated the 21.90% pre-filter drop rate in dense bi-encoders and evaluated post-hoc score calibration methods. To control human auditing effort, I derived and implemented the LAC and CRC conformal prediction sets, confirming 90.03% coverage at $\alpha = 0.10$ and demonstrating a 70.63% reduction in reading load.

For ungrounded errors, I calibrated the abstention gate ($\tau = 0.1973$, achieving 80.80% recall), evaluated Leave-One-Task-Out transfer, and designed the tie-break fusion rule that improved response F1 by +5.96 points. On the engineering side, I packaged the runtime service into `C4Attributor` and `c4_artifact.json` (running in 1.312 ms on CPU) and wrote 147 automated unit and contract tests.

---

## 1.9 Thesis structure

This dissertation is organised into eight subsequent chapters. Chapter 2 surveys existing literature in hallucination detection, citation grounding, and conformal prediction. Chapter 3 explains our experimental setup, dataset splits, and evaluation metrics. In Chapter 4, we lay out system requirements, architecture diagrams, and interface schemas. Chapter 5 covers software implementation, API wiring, and testing. Chapter 6 details experimental results across 16 tables and 7 figures, followed by a critical discussion of findings and limitations in Chapter 7. Chapter 8 reviews commercial deployment, licensing, and sustainability, and Chapter 9 concludes with future research directions.

---

# 2 LITERATURE REVIEW AND RELATED WORK

## 2.1 Review method and scope

We performed a structured review of academic publications following established guidelines for software engineering and machine learning [20]. Our search focused on five interrelated topics: (1) RAG hallucination benchmarks; (2) token-level error detection; (3) citation and evidence attribution; (4) probability calibration and conformal prediction in NLP; and (5) selective abstention on unanswerable inputs.

### 2.1.1 Databases, search strings, and selection criteria
We queried ACL Anthology, IEEE Xplore, ACM Digital Library, arXiv, and Google Scholar for papers published between 2017 and 2026. Table 2.1 lists our search queries and selection criteria.

Table 2.1: Systematic Literature Search Strings, Databases, and Inclusion/Exclusion Criteria.

| Research Theme | Boolean Search String | Primary Databases | Inclusion Criteria | Exclusion Criteria |
| :--- | :--- | :--- | :--- | :--- |
| **1. RAG Hallucination Benchmarks & Taxonomy** | ("Retrieval-Augmented Generation" OR "RAG") AND ("hallucination" OR "faithfulness") AND ("span-level" OR "benchmark" OR "RAGTruth" OR "RAGBench") | ACL Anthology, arXiv, ACM DL | Peer-reviewed or high-impact benchmark papers defining intrinsic/extrinsic or conflict/baseless RAG hallucinations. | Purely parametric (closed-book) trivia QA studies without retrieved context. |
| **2. Evidence Attribution & Citation Grounding** | ("evidence attribution" OR "citation generation" OR "source grounding" OR "ALCE") AND ("large language model" OR "NLI" OR "entailment") | ACL Anthology, EMNLP, TACL, IEEE Xplore | Studies aligning generated claims or spans to supporting/contradicting source sentences. | Gradient-based token saliency maps that do not index into retrieved documents. |
| **3. Conformal Prediction & Risk Control in NLP** | ("conformal prediction" OR "conformal risk control" OR "prediction sets" OR "LAC") AND ("natural language processing" OR "question answering" OR "hallucination") | JMLR, NeurIPS, ICML, TACL, arXiv | Papers deriving finite-sample coverage bounds ($1 - \alpha$) or risk bounds for set-valued NLP predictors. | Heuristic top-$k$ retrieval studies lacking distribution-free coverage guarantees. |
| **4. Selective Abstention & Unanswerability** | ("selective prediction" OR "abstention" OR "no evidence" OR "unanswerable") AND ("reading comprehension" OR "fact verification" OR "FEVER" OR "SQuAD 2.0") | ACL Anthology, AAAI, TACL | Works formulating NOT ENOUGH INFO or no-evidence abstention thresholds on retrieved passages. | Refusal training for safety/toxicity unrelated to factual context sufficiency. |

Filtering title and abstracts from an initial set of 218 papers narrowed the pool down to 94 articles, resulting in 45 core references cited in this thesis.

---

## 2.2 Domain and theoretical foundations

### 2.2.1 How RAG works and where it fails
In a standard RAG setup [1], a retriever selects context passages $D \subset \mathcal{D}$ for a user query $q$, and a language model produces an answer $y$. When errors occur, they generally fall into two groups [3], [5]:
1. Context Conflicts (Intrinsic Errors): The model's answer contradicts information in $D$. In RAGTruth [6], these are split into Evident Conflict (direct contradiction or numerical error) and Subtle Conflict (subtle nuance shifts).
2. Baseless Errors (Extrinsic Errors): The answer includes claims not mentioned anywhere in $D$. In RAGTruth [6], these include Evident Baseless Info (completely fabricated statements) and Subtle Baseless Info (plausible details absent from the text).

For Conflicts, a source sentence exists in $D$, making attribution a task of finding the right sentence. For Baseless claims, no source sentence exists, meaning the system needs to abstain rather than guess.

### 2.2.2 Scoring sentences: lexical vs neural methods
Given an answer span $s_i$ and candidate context sentences $D = (c_1, \dots, c_m)$, attribution relies on a scoring function $f(s_i, c_j)$:

1. IDF Token Overlap: Content words $T(x)$ are weighted using inverse document frequency computed within the passage:
$$\text{idf}(t; D) = \ln\left(\frac{m + 1}{\text{df}(t; D) + 1}\right) + 1.0, \quad s_j = \frac{\sum_{t \in T(s_i) \cap T(c_j)} \text{idf}(t; D)}{\sum_{t \in T(s_i)} \text{idf}(t; D)} \in [0, 1]$$
This produces a bounded score in $[0, 1]$ without requiring trained weights or risking cross-document data leakage.
2. Dense Bi-Encoders: Encoders like BGE [23] map spans and sentences into vector embeddings and measure cosine similarity. However, sentence-level pooling often dilutes specific dates, numbers, and proper nouns.
3. Cross-Encoders: Cross-attention models like DeBERTa-v3 [24] and HHEM [28] concatenate the span and sentence. While expressive, scoring all candidates takes $O(m \cdot L^2)$ time, requiring pre-filtering that can discard gold sentences.

### 2.2.3 Calibration of model scores
Raw model outputs seldom represent true empirical probabilities [25]. Three post-hoc calibration methods are widely used on held-out validation data:
1. Temperature Scaling [25]: Scales logits with a positive scalar $T$.
2. Platt Scaling [26]: Fits a logistic regression model $\hat{p} = \sigma(a z + b)$ with slope and intercept.
3. Isotonic Regression [27]: Fits a monotonic piecewise step function.

We evaluate calibration using Expected Calibration Error (ECE) and Brier score, interpreted alongside a constant base-rate baseline.

### 2.2.4 Conformal prediction sets
To output a set of candidate sentences $C_\alpha(s) \subseteq \{1, \dots, m\}$ with guaranteed coverage, we employ split-conformal prediction [15], [16] via Least Ambiguous Set-Valued Classifiers (LAC) [17]. On calibration examples, the nonconformity score of the gold sentence is $R_r = 1 - p_{r, j_{\text{gold}}}$. For a chosen error rate $\alpha$, the empirical quantile $\hat{q}_{1-\alpha}$ is computed with a finite-sample correction. At test time, the conformal set includes all sentences meeting the cutoff:
$$C_\alpha(s_{\text{test}}) = \left\{\, j \in \{1, \dots, m\} \;\middle|\; 1 - p_j \le \hat{q}_{1-\alpha} \,\right\}$$
Under exchangeability, this guarantees that $P(j_{\text{gold}} \in C_\alpha) \ge 1 - \alpha$.

---

## 2.3 Related research

### 2.3.1 RAG benchmark datasets
RAGTruth [6] provides human-annotated character-offset spans across QA, Data-to-Text, and Summarization with Original source notes. Prior work used these notes only for overall detection, leaving them unaudited as evidence references. RAGBench [29] offers response-level adherence metrics, used in TrustRAG as an out-of-distribution benchmark.

### 2.3.2 Token-level hallucination detection
Recent detectors like LettuceDetect [10] frame hallucination detection as token classification using ModernBERT [11]. While fast, sequence taggers only highlight words in the response; they do not inspect or cite the context passage.

### 2.3.3 Citation and attribution tools
Frameworks like ALCE [19], AttrScore, and SummaC [14] evaluate citations for supported claims. Component 4 addresses the reverse task: finding evidence for flagged hallucinated spans, where contradicted sentences share content words with the span while disagreeing with its predicate.

### 2.3.4 Conformal prediction in NLP
Conformal methods have been applied to QA answer sets [32] and response factuality [33]. Component 4 applies conformal prediction to context sentence sets, measuring reading-effort reduction across RAG tasks.

---

## 2.4 Existing systems, products and techniques

We compare Component 4 with four prevalent systems:
1. Vectara HHEM [28]: A 184M-parameter cross-encoder for response-level evaluation. When adapted for sentences, it takes 24.6 ms/span and needs 440 MB of memory without offering conformal sets.
2. Ragas [12]: Uses external LLM calls (GPT-4o) to evaluate claims. It incurs high latency (>1,500 ms), API costs, and privacy risks without providing sentence offsets.
3. TruLens [30]: Prompts LLMs or runs NLI models for chunk-level groundedness, lacking calibrated abstention and finite-sample guarantees.
4. LettuceDetect [10]: Provides fast token-level detection via ModernBERT, but does not segment context or provide evidence attribution.

---

## 2.5 Gap synthesis and conceptual framework

### 2.5.1 Comparative gap matrix
Table 2.2 summarises key capabilities across existing tools and Component 4.

Table 2.2: Comparative Gap Synthesis Matrix of Hallucination Detection and Attribution Systems.

| System / Study | Granularity | Evidence Sentence Attribution | Gold Supervision Audited? | Conformal Evidence Sets ($1-\alpha$) | Calibrated No-Evidence Abstention | Detector Tie-Break Fusion | Online Overhead (Latency / RAM) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Ragas (Es et al. [12])** | Atomic claim (LLM) | Prompt-based (unindexed) | No | No | No | No | $>2,000\text{ ms}$ / API cloud |
| **TruLens [30]** | Answer sentence | Top-1 NLI / LLM | No | No | No | No | $>500\text{ ms}$ / $>700\text{ MB}$ |
| **Vectara HHEM [28]** | Response-level | Adapted Top-1 ($0.6959$) | No | No | No | No | $24.6\text{ ms}$ / $440\text{ MB}$ |
| **SummaC / AlignScore [14], [13]** | Sentence matrix | Top-1 NLI cross-encoder | No | No | No | No | $>30\text{ ms}$ / $>700\text{ MB}$ |
| **LettuceDetect [10]** | Answer span (ModernBERT) | None (answer-only) | N/A | No | No | No | Detector only |
| **C3 Lexical Baseline (`grounding.py`)** | Answer span | Nearest sentence (forced) | First-line only | No | No ($0.0000$ recall) | No | $1.1\text{ ms}$ / $0\text{ MB}$ |
| **TrustRAG Component 4 (Ours)** | **Answer span $\rightarrow$ Context sentence(s)** | **IDF Lexical + 4-Backend Ablation ($0.8952$ Top-1)** | **Yes (Layer 1 Audit + Blind $\kappa = 0.8466$)** | **Yes (LAC & CRC, $70.63\%$ reading saved)** | **Yes ($\tau = 0.1973$, $0.8080$ recall)** | **Yes (`c4_tie_break`, $+5.96$ F1 pts)** | **$1.31\text{ ms}$ p50 / $0\text{ MB}$ RAM** |

### 2.5.2 Conceptual framework
Component 4 is structured around four pillars:
1. Audit before modelling: Verify ground-truth annotations and splitter invariance before evaluating models.
2. Isolate pipeline bottlenecks: Compare lexical and neural backends against task floors, measuring pre-filter loss directly.
3. Set-valued conformal grounding: Replace point citations with conformal sets to give clear statistical coverage bounds.
4. Closed-loop abstention and fusion: Use calibrated abstention to handle baseless errors and resolve detector uncertainty.

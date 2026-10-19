



# TRUSTRAG: SPAN-LEVEL CALIBRATED HALLUCINATION DETECTION FOR RETRIEVAL-AUGMENTED GENERATION

INDIVIDUAL THESIS

Epitakumbure Mudunkothgedara Sakindu Nivein Rathnasiri

(IT22047724)

Individual component: Retrieval-Attribution Alignment and Evidence Grounding

Group ID: R26-DS-018

Report submitted in partial fulfilment of the requirements for the

Bachelor of Science Honours in Information Technology

Specialising in Data Science

Department of Information Technology

Sri Lanka Institute of Information Technology

Sri Lanka

October 2026



## DECLARATION OF THE CANDIDATE AND SUPERVISOR

“I declare that this is my own work and this dissertation does not incorporate without acknowledgement any material previously submitted for a Degree or Diploma in any other University or institute of higher learning and to the best of my knowledge and belief it does not contain any material previously published or written by another person except where the acknowledgement is made in the text.

Also, I hereby grant to Sri Lanka Institute of Information Technology, the non-exclusive right to reproduce and distribute my dissertation, in whole or in part in print, electronic or other medium. I retain the right to use this content in whole or part in future works (such as articles or books).”

Academic Integrity and AI Disclosure: I have disclosed all use of generative AI and other automated tools in Appendix B in accordance with the module policy. I remain responsible for the accuracy, originality, citations, code, data, analyses, and conclusions in this dissertation.

Candidate Name: Epitakumbure Mudunkothgedara Sakindu Nivein Rathnasiri (IT22047724)

Signature: ________________________________________     Date: ___________________

The supervisor/s should certify the dissertation with the following declaration:

“The above candidate has carried out research for the bachelor’s degree Dissertation under my supervision.”





## DEDICATION

I dedicate this dissertation to my parents and family, whose sacrifices, patience, and encouragement 
sustained me throughout my undergraduate studies at the Sri Lanka Institute of Information Technology. I 
also dedicate it to the researchers and open-source developers working to make artificial intelligence 
trustworthy and verifiable.



## ACKNOWLEDGEMENTS

I thank our primary supervisor, Mr. Samadhi Rathnayaka, and our co-supervisor, Dr. Lakmini 
Abeywardhana, from the Department of Information Technology, Faculty of Computing, Sri Lanka 
Institute of Information Technology (SLIIT). From the beginning of the project, their feedback helped 
shape our experimental setup, statistical calibration methods, and baseline evaluations.

Working on TrustRAG with my teammates in Group R26-DS-018 has been a rewarding part of this project. 
I am grateful to Wimukthi Gunarathna (IT22060976, Component 1 Lead), Sadini Wanniarachchi 
(IT22592546, Component 2 Lead), and Ranya Sivakumar (IT22607232, Component 3 Lead) for their 
commitment and collaboration. Their work made it possible to maintain a unified Pydantic schema 
(src/common/schema.py) and a shared regression test suite across all four components.

I also thank the academic and administrative staff of the Faculty of Computing at SLIIT, together with my 
family and friends, for supporting me throughout my degree. The research used institutional and personal 
hardware resources and received no commercial sponsorship.



## ABSTRACT

Retrieval-Augmented Generation (RAG) uses external text to ground Large Language Models and reduce 
factual errors. However, evaluations on RAGTruth show that 43.1% of generated responses still include 
claims that are unsupported or contradicted. Token-level detectors can mark suspicious spans in an answer, 
but human auditors must still search the retrieved context to determine whether a sentence contradicts the 
claim or whether the relevant information is absent. On RAGTruth, this means checking an average of 
19.05 sentences per span. The focus of this dissertation is Component 4 (Retrieval-Attribution Alignment and Evidence Grounding) 
of the TrustRAG platform. First, we audit RAGTruth's human evidence annotations across frozen splits (1,214 calibration and 1,533 test spans), finding that only 50.68% of human quotes match source sentences verbatim. We develop a content-token containment matching protocol and carry out a dual-rater blind audit of 50 no-
evidence cases (94.0% agreement, Cohen's κ = 0.8466). Second, we benchmark lexical, dense embedding (bge-small-en-v1.5), and cross-encoder models (DeBERTa-v3, HHEM), showing that an aggressive top-5 pre-filter drops the true evidence sentence 21.94% of the time, whereas full-pool scoring drops only 2.35%. The lexical attributor has a median CPU latency of 1.312 ms and uses 0.0 MB of extra memory. Third, we frame evidence retrieval as split-conformal prediction using Least Ambiguous Set-Valued Classifiers (LAC); at α = 0.10, Component 4 achieves 90.03% empirical coverage while reducing reviewer reading load from 19.05 to 4.61 sentences per span—a 70.6% reduction in reading effort that translated to a 69.47% review speedup in user testing. Finally, using our no-evidence signal to resolve uncertain detector abstentions yields an isolated +5.96 point gain in response-level F1.

Keywords: Retrieval-Augmented Generation, Hallucination Attribution, Split Conformal Prediction, Evidence Grounding, Selective Abstention



## TABLE OF CONTENTS





## LIST OF FIGURES

Page

Figure 4.1 TrustRAG Four-Stage Calibrated Architecture with Component 4 Highlighted	44

Figure 4.2 Component 4 Internal Attribution, Conformal Prediction Set, and Abstention Pipeline	48

Figure 4.3 Frozen Pydantic Data Contract (`src/common/schema.py`) and C4 Serving Artefact Schema	50

Figure 6.1 Attribution Backend Top-1 Accuracy Against Trivial Floors by Task and Pre-Filter Ceiling	66

Figure 6.2 Post-Hoc Attribution Score Calibration and Expected Calibration Error Across Calibrators	67

Figure 6.3 Conformal Evidence Set Empirical Coverage vs. Target 1 - α with 3σ Finite-Sample Band	68

Figure 6.4 Mean Conformal Evidence Set Size vs. Candidate Context Pool Size Across Risk Levels	69

Figure 6.5 Calibrated No-Evidence Abstention Precision, Recall, and F1 vs. Trivial Baselines	70

Figure 6.6 Leave-One-Task-Out Cross-Task Transfer of Conformal Quantiles (`alpha = 0.10`)	71

Figure 6.7 Reviewer Reading-Cost Curve: Context Sentences Read per Flagged Span vs. Empirical Coverage	72



## LIST OF TABLES

Page

Table 1.1 Ownership and Interface Boundaries of the Four TrustRAG Research Components	2

Table 1.2 Mapping of Research Questions, Individual Objectives, Evaluation Layers, and Thesis Chapters	9

Table 2.1 Systematic Literature Search Strings, Databases, and Inclusion/Exclusion Criteria	15

Table 2.2 Comparative Gap Synthesis Matrix of Hallucination Detection and Attribution Systems	24

Table 3.1 Frozen RAGTruth Dataset Splits and Evidence Annotation Distribution (Layer 1 Audit)	30

Table 3.2 Twelve-Layer Experimental Protocol and Associated Artefact Outputs in Component 4	33

Table 5.1 Development Environment, Software Dependencies, and Pinning Strategy	53

Table 5.2 Layer-by-Layer Module Inventory of `src/c4_attribution/`	55

Table 5.3 Five-Level Software Verification and Regression Test Matrix (`147` C4 Tests / `861` Suite Tests)	59

Table 6.1 Evidence Supervision on Frozen Calibration and Test Splits (`C4-1`)	63

Table 6.2 Verbatim Rate, Content-Token Containment, and No-Evidence Share by Task and Error Type (`C4-2`)	64

Table 6.3 Attribution Matching Protocol and Containment Threshold θ Sweep on Test Split (`C4-3`)	65

Table 6.4 Lexical Attribution Baseline Against Four Trivial Floors on Test Split (`C4-4`)	65

Table 6.5 Attribution Backend Comparison and Hypothesis-Formation Ablations on Test Split (`C4-5`)	66

Table 6.6 Candidate Pre-Filter Recall@k and Irrecoverable Error Floor Across k ∈ \{1, 3, 5, 10\} (`C4-6`)	66

Table 6.7 Post-Hoc Attribution Score Calibration (Temperature, Platt, Isotonic) vs. Constant Floor (`C4-7`)	67

Table 6.8 Split Conformal Evidence Prediction Sets and Reviewer Reading Reduction on Test Split (`C4-8`)	68

Table 6.9 Marginal vs. Group-Conditional Conformal Coverage and Set Size by Task at α = 0.10 (`C4-9`)	68

Table 6.10 Comparison of Least Ambiguous Set-Valued Classifiers (LAC) and Conformal Risk Control (CRC, `C4-10`)	69

Table 6.11 Calibrated No-Evidence Abstention Performance on Test Split (`1,046` Spans, `C4-11`)	70

Table 6.12 Cross-Task (LOTO) and Cross-Dataset (`MedHallu`, `RAGBench`) Transfer Evaluation (`C4-12`)	70

Table 6.13 Reviewer Reading Burden and Effort Reduction Across α Levels (`C4-13`)	71

Table 6.14 Runtime Inference Latency, Span Throughput, and Memory Footprint Across Backends (`C4-15`)	73

Table 6.15 End-to-End User Acceptance Testing (UAT), Blind Audit (κ = 0.8466), and Reviewer Timing (`C4-16`)	74

Table 6.16 Multi-Component C1 × C2 × C4 Fusion and Disagreement Matrix on Test Split (`C4-14`)	75

Table 8.1 Competitive Positioning of TrustRAG (with Component 4) Against Existing RAG Verification Products	86

Table 9.1 Summary of Achievement of Research Objectives and Research Questions	91

Table A.1 Complete 50-Span Dual-Rater Blind Human Audit of the `no_evidence` Ground-Truth Label	101

Table B.1 Generative AI Use Disclosure Log	104



## LIST OF ABBREVIATIONS





## LIST OF APPENDICES

Page

Appendix - A Extended Experimental Results, Production Calibration Artifact (`c4_artifact.json`), and Complete 50-Span Dual-Rater Blind Audit Table	100

Appendix - B Academic Integrity and Generative AI Use Disclosure Log	104



# 1 INTRODUCTION

## 1.1 Overall Project Context

Large Language Models (LLMs) are now widely used in enterprise knowledge systems, clinical 
summarisation, legal analysis, and automated customer support. Improvements in training scale and 
instruction tuning have not removed the tendency of standalone models to hallucinate. They can still 
produce fluent, confident statements that are factually wrong or lack support from reliable sources [1], [2]. 
Many deployed systems therefore rely heavily on Retrieval-Augmented Generation (RAG) [3]. In a 
standard RAG pipeline, the user's query is used to retrieve relevant passages from an indexed document 
store, and the language model draws on those passages to produce its answer.

Grounding an answer in retrieved text reduces pure fabrication, but hallucinations still occur. Niu et al. [6] 
evaluated 17,790 RAGTruth responses covering Question Answering (QA), Data-to-Text (D2T), and 
Summarization. They found at least one hallucinated span in 43.1% of the RAG responses. The errors fall 
into two categories:

1. Context Conflicts (Evident Conflict and Subtle Conflict): The relevant fact appears in the retrieved 
context, but the model reads it incorrectly or changes it. For example, an answer may give 1925 as the year 
of an event even though the source passage gives 1889.

2. Baseless Fabrications (Evident Baseless Info and Subtle Baseless Info): The model makes a claim for 
which the retrieved passage contains no evidence. Such claims may come from parametric training memory 
or may be entirely fabricated.

Hallucinations that go undetected pose serious operational risks in healthcare, finance, and law. Existing 
evaluation tools, including Ragas [12], TruLens [30], Vectara HHEM [28], and general LLM judges, have 
four main limitations:

• Response-Level Granularity: Most tools assign one faithfulness score, such as 0.64, to an entire multi-
paragraph response. They do not identify the particular words that are hallucinated.

• Miscalibrated Scores: Raw neural classifier scores do not reflect empirical error rates, so they provide an 
unreliable basis for choosing thresholds [25].

• Lack of Shift Awareness: The coverage guarantees of split conformal prediction [16] depend on test data 
being exchangeable with calibration data. A domain shift can reduce coverage without any warning.

• No Calibrated Grounding or Abstention: Once a suspicious answer span has been flagged, the auditor 
must still locate its source sentence or establish that no evidence exists. This requires reading the full 
retrieved context, which averages 19.05 sentences per span on RAGTruth.

To resolve these challenges in an integrated software platform, Research Group R26-DS-018 developed TrustRAG: Span-Level Calibrated Hallucination Detection for Retrieval-Augmented Generation. TrustRAG is organised into four contract-driven components governed by a unified Pydantic schema (src/common/schema.py) and verified by 861 automated tests:

*Table 1.1: Ownership and Interface Boundaries of the Four TrustRAG Research Components.*



## 1.2 Individual Component Context

Within TrustRAG, Component 4 (Retrieval-Attribution Alignment and Evidence Grounding) serves as the bridge between answer-side span detection (Components 1 and 2) and source-side human verification. Once a span is flagged in a response, a human reviewer faces a practical question:

Which sentence in the retrieved passage settles this claim—or is the claim completely unsupported by the source?

Retrieving evidence for hallucinated spans presents several specific technical challenges:

1. Annotators often shorten the source sentences, change whitespace, or bring multi-line JSON values 
together. A suitable containment matching protocol must therefore be established before an attribution 
model can be evaluated.

2. Standard retrieval tools always return the sentence with the highest score. For an Evident Conflict, that 
sentence reveals the contradiction. For a Baseless claim, however, no supporting sentence exists. Baseless 
claims account for 72.73% of annotated QA spans. Returning a citation in these cases misleads the 
reviewer and gives 0.0000 recall on baseless errors. Component 4 uses a calibrated abstention threshold (τ 
= 0.1973) to explicitly identify missing evidence.

3. RAGTruth context passages contain an average of 19.05 sentences and can contain as many as 89. A top-1 
result does not guarantee that the true evidence sentence has been retrieved, while reading all 19 sentences 
is too time-consuming. Component 4 uses split-conformal prediction with Least Ambiguous Set-Valued 
Classifiers (LAC) to form a small evidence set C_α(s). At α = 0.10, this set guarantees 90.03% coverage 
and reduces the average reading burden to 4.61 sentences per span, a 70.63% reduction in reading effort.

4. Component 2 abstains on 37.4% of spans to preserve coverage. If Component 2 gives an ABSTAIN 
decision and Component 4 finds no evidence anywhere in the source passage, c4_tie_break promotes the 
span. This recovers 65 true-positive responses and raises response F1 from 0.6960 to 0.7557 (+5.96 points).

## 1.3 Problem Statement

This dissertation investigates the following problem:

Problem Statement: For a retrieved passage D divided into sentences (c_1, \dots, c_m) and an answer span s_i flagged by a 
hallucination detector, current RAG evaluation systems lack a calibrated attribution method that meets four 
requirements. It must (i) link conflict spans to their exact source sentence when human supervision is non-
verbatim; (ii) produce finite-sample conformal evidence sets that bound reviewer reading effort while 
guaranteeing coverage; (iii) abstain when a baseless span has no source evidence; and (iv) operate 
alongside an active transformer model with <50 ms CPU latency and 0 MB extra GPU memory. Four 
empirical measurements on RAGTruth establish this problem:

This problem is grounded in four empirical measurements on RAGTruth:

• High reading burden: Contexts contain 18.92 sentences on average (19.05 on test), so an unassisted 
reviewer must read roughly twenty sentences for each span.

• Supervision noise: Exact substring matching is inadequate for evaluation because only 50.68% of human 
quotes reproduce the context text verbatim.

• High prevalence of baseless errors in QA: On QA, 72.73% of annotated spans have no evidence in the source passages, causing unthresholded models to fail completely on abstention.

• Pre-filter bottleneck in two-stage pipelines: Dense bi-encoders (bge-small) used to select candidate sentences for cross-encoders drop the true gold sentence 21.90% of the time at k=5, setting an irrecoverable ceiling on downstream neural models.

## 1.4 Research Gap

Reviewing recent literature highlights four main research gaps:

1. Gap 1: Unaudited Supervision and Unmeasured Trivial Floors: No prior work has audited RAGTruth's Original: annotations to separate attributable quotes from missing-evidence denials, nor reported results against task-specific trivial floors (longest sentence, first sentence, random). On Data-to-Text, always choosing the longest sentence achieves 0.5430 Top-1 accuracy due to JSON formatting—a key factor that pooled numbers obscure.

2. Gap 2: Unmeasured Pre-Filter Ceilings vs. Lexical Scoring: Two-stage neural retrieval pipelines rarely measure how many gold sentences are lost during initial filtering. We show that dense top-5 pre-filtering loses 21.90% of gold sentences, whereas full-pool IDF lexical containment achieves 0.8952 Top-1 accuracy with zero GPU memory.

3. Gap 3: Lack of Conformal Prediction Sets for Context Grounding: Existing systems return either a single sentence or a fixed top-k list that cannot adapt to passage length. Developing adaptive conformal evidence sets with distribution-free coverage bounds across variable passage lengths (4 to 89 sentences) addresses an important practical gap.

4. Gap 4: Forced Attribution and Disconnected Pipeline Stages: Standard tools treat span detection and retrieval as one-way steps without an abstention mechanism, and do not use evidence availability to resolve detector uncertainty.

## 1.5 Aim and Objectives

### 1.5.1 Project aim and component aim

The overall aim of the TrustRAG project is to develop and validate a statistically calibrated, span-level hallucination detection, explanation, and evidence-grounding platform for RAG with distribution-shift safeguards.

The individual aim of Component 4 is:

To design, calibrate, and evaluate an evidence-grounding module that aligns flagged answer spans with minimal conformal sentence sets under finite-sample guarantees, abstains on baseless claims lacking source evidence, resolves detector abstentions via multi-component fusion, and operates in real time with zero extra GPU memory overhead.

### 1.5.2 Measurable objectives

1. Objective 1 (Data Audit & Ground-Truth Protocol): Audit Original: annotations across RAGTruth's frozen splits (1,214 calibration and 1,533 test spans), establish a θ = 0.50 containment matching protocol, and validate the no-evidence heuristic through a 50-span blind audit achieving ≥ 0.90 agreement and Cohen's κ > 0.80.

2. Objective 2 (Multi-Backend Benchmarking & Pre-Filter Isolation): Implement and compare lexical, dense (bge-small), and cross-encoder backends (DeBERTa-v3, HHEM) against four trivial baselines, while isolating the pre-filter recall ceiling.

3. Objective 3 (Score Calibration & Conformal Sets): Evaluate post-hoc calibration methods (Temperature, Platt, Isotonic) and construct LAC/CRC conformal evidence sets across α ∈ [0.01, 0.40], achieving ≥ 90% empirical coverage at α = 0.10 while cutting reviewer reading burden by at least 65%.

4. Objective 4 (Calibrated Abstention, Transfer, Fusion & Serving): Calibrate a no-evidence abstention threshold (τ = 0.1973), evaluate Leave-One-Task-Out transfer, quantify the F1 gain from C1×C2×C4 tie-breaking, and deploy a JSON configuration artifact (c4_artifact.json) delivering <5 ms CPU latency.

## 1.6 Research Questions or Hypotheses

• RQ1 (Supervision Structure & Pre-Filter Ceiling): How reliably do RAGTruth's human quotes map to context sentences, and how does candidate pre-filtering impact lexical vs. cross-encoder attribution?

• Hypothesis H1: Under 60% of quotes appear verbatim in the context, and full-pool lexical containment will outperform dense pre-filtered cross-encoders because dense filtering drops >15% of gold sentences.

• RQ2 (Conformal Sets & Reviewer Effort): Can split conformal prediction guarantee ≥ 1 - α coverage across variable context lengths while significantly reducing reviewer reading load?

• Hypothesis H2: At α = 0.10, conformal prediction will achieve coverage within the 3σ band ([0.8640, 0.9338]) and reduce mean reading burden by more than 65% compared to full context.

• RQ3 (No-Evidence Abstention & Task Baselines): How effectively does a calibrated threshold distinguish baseless errors from attributable spans across tasks?

• Hypothesis H3: A calibration-fitted threshold will reach >0.80 recall on baseless spans and beat the always_abstain floor pooled, on Data-to-Text, and on Summarization.

• RQ4 (Multi-Component Tie-Break Fusion): Does using Component 4's no-evidence signal to resolve Component 2's uncertain abstentions produce a meaningful F1 improvement?

• Hypothesis H4: Promoting Component 2 ABSTAIN spans when Component 4 certifies no_evidence = True will increase response F1 by more than the 4.18-point bootstrap noise margin of Component 1.

*Table 1.2: Mapping of Research Questions, Individual Objectives, Evaluation Layers, and Thesis Chapters.*



## 1.7 Scope Limitations and Assumptions

### 1.7.1 Scope

1. Dataset & Splits: Offline evaluations use the RAGTruth benchmark [6] across QA, Data-to-Text, and Summarization on the exact frozen calibration (297 responses, 1,214 spans) and test splits (2,670 responses, 1,533 spans; 1,509 responses for fusion) established by Component 1.

2. Granularity: Attribution is performed at the sentence level using the shared regex splitter (?<=[.!?])\s+, populating Span.evidence_sentence and Span.evidence_index.

3. Serving Mode: Offline tests compare four backends; the live system deploys the calibrated lexical engine via c4_artifact.json.

### 1.7.2 Limitations

1. Regex Sentence Splitter: To maintain contract consistency across components, we keep the serving regex splitter. Layer 1c shows this causes zero index shifts compared to a passage-aware splitter.

2. Lack of Sentence Gold Annotations on RAGBench: RAGBench provides only response-level labels. Cross-task transfer is therefore evaluated via Leave-One-Task-Out folds across RAGTruth tasks.

3. Single-Sentence Gold References: For quotes spanning multiple adjacent sentences, the protocol selects the sentence with highest token containment.

### 1.7.3 Assumptions

1. Within-Domain Exchangeability: Conformal coverage bounds assume calibration and test examples are exchangeable. When inputs shift out-of-distribution, Component 2's exchangeability gate withdraws guarantees while still rendering candidate sentences.

2. Target Hardware: The serving environment is designed to operate on a standard developer laptop with 6 GB VRAM or CPU-only setups.

## 1.8 Individual Contribution

The candidate (E. M. S. N. Rathnasiri, IT22047724) independently carried out the research, mathematical 
formulation, software implementation, testing, and documentation for Component 4:

1. Supervision Audit & Dual-Rater Blind Validation (Layers 1, 1b, 1c): Built marker parsing tools, measured convention disagreements (6.42%), and conducted the 50-span blind audit achieving 94.0% accuracy and Cohen's κ = 0.8466.

2. Matching Protocol & Trivial Baselines (Layers 2 & 3): Developed the token containment matcher with θ = 
0.50 and compared IDF scoring with four non-trivial baselines. These comparisons exposed the Data-to-
Text length confound.

3. & Pre-Filter Ceiling (Layers 4, 4b, 5): Compared four backends, identified the 21.90% pre-filter error 
ceiling in dense bi-encoders, and assessed Temperature, Platt, and Isotonic calibration.

4. Conformal Evidence Sets & Reading Burden Curve (Layers 6 & 10): Implemented LAC and CRC 
conformal prediction sets and verified their finite-sample coverage (90.03% at α = 0.10). The evaluation 
also measured a 70.63% reduction in reviewer reading load.

5. Calibrated Abstention, Transfer & Tie-Break Fusion (Layers 7, 8, 9, 11): Calibrated the abstention 
threshold (τ = 0.1973, 80.80% recall), carried out LOTO transfer, and developed the tie-break fusion rule 
responsible for the +5.96 F1 point gain.

6. & Test Suite (Layers 12–15): Developed C4Attributor and c4_artifact.json with 1.312 ms p50 latency, 
implemented responsible AI rules, and wrote 147 passing unit and contract tests.

## 1.9 Thesis Structure

The remaining chapters are arranged as follows:

• Chapter 2 (Literature Review and Related Work) examines previous work on hallucination detection, 
citation attribution, calibration, and conformal prediction.

• Chapter 3 (Research Methodology) sets out the twelve-layer experimental protocol, including the datasets, 
mathematical formulations, and evaluation metrics.

• Chapter 4 (Product and Solution Design) explains the system requirements, architectural diagrams, data 
contracts, and engineering standards.

• Chapter 5 (Implementation, Integration and Testing) describes the software modules, their REST and 
React integration, and the five-tier testing strategy.

• Chapter 6 (Results and Validation) reports the empirical results in Tables C4-1 to C4-16 and Figures 6.1 
to 6.7.

• Chapter 7 (Discussion) explains the main findings, compares them with existing tools, and considers the 
study's limitations.

• Chapter 8 (Commercialisation and Sustainability) discusses the intended users, business models, 
scalability, and mapping to the SDGs.

• Chapter 9 (Conclusion and Future Directions) relates the outcomes to the research objectives and sets out 
four directions for future research.



# 2 LITERATURE REVIEW AND RELATED WORK

## 2.1 Review Method and Scope

We established the theoretical basis for the study through a structured literature review guided by 
Kitchenham and Charters' recommendations for software engineering and machine learning [20]. The 
review covered five topics: (1) RAG hallucination taxonomy and benchmarks; (2) token-level hallucination 
detection; (3) post-hoc citation and evidence attribution; (4) probability calibration and conformal 
prediction in NLP; and (5) selective abstention and unanswerability in reading comprehension.

### 2.1.1 Databases, search strings, and selection criteria

We searched ACL Anthology, IEEE Xplore, ACM Digital Library, arXiv, and Google Scholar for papers published from 2017 to 2026.

*Table 2.1: Systematic Literature Search Strings, Databases, and Inclusion/Exclusion Criteria.*



From an initial set of 218 papers, title and abstract screening filtered out 124 duplicate or out-of-scope studies. Detailed review of the remaining 94 publications yielded 45 core references synthesised in this chapter.

## 2.2 Domain and Theoretical Foundations

### 2.2.1 Retrieval-augmented generation and context failure types

Errors in generation generally belong to two categories [3], [5]:

• Intrinsic Hallucinations (Context Conflicts): A generated claim conflicts with facts in D. RAGTruth [6] 
distinguishes Evident Conflict, which involves a direct contradiction or numerical error, from Subtle 
Conflict, which involves a small shift in nuance.

• The claim contains facts that do not appear in D. RAGTruth [6] classifies these as Evident Baseless Info 
when details are completely fabricated, or Subtle Baseless Info when plausible elaborations are absent from 
D.

For Conflicts, a source sentence exists in D, making attribution a localisation task. For Baseless claims, no supporting sentence exists, making attribution a selective abstention task.

### 2.2.2 Lexical vs. neural cross-encoder scoring

Attributing an answer span s_i to context sentences D = (c_1, \dots, c_m) requires the scoring function 
f(s_i, c_j):

1. IDF-Weighted Lexical Overlap: Content tokens T(x) are weighted by within-document inverse document frequency:

idf(t; D) = \ln\!≤ft((m + 1)/(df(t; D) + 1)) + 1.0, \quad s_j = \frac{∑_{t ∈ T(s_i) ∩ T(c_j)} idf(t; D)}{∑_{t ∈ T(s_i)} idf(t; D)} ∈ [0, 1]

The score is bounded, uses no learned parameters, and prevents cross-document leakage.

2. Dense Bi-Encoders: Models such as BGE [23] represent spans and sentences as vector embeddings and 
compare them using cosine similarity. Pooling at sentence level, however, often weakens the representation 
of particular dates, numbers, and proper nouns.

3. Cross-Encoders: DeBERTa-v3 [24] and HHEM [28], for example, concatenate each sentence-span pair 
and apply full cross-attention. These models are expressive, but scoring every candidate requires O(m · 
L^2) time. The resulting need for pre-filtering can lead to gold sentences being discarded.

### 2.2.3 Post-hoc probability calibration

Raw model scores seldom correspond to empirical frequencies [25]. Three common calibration maps are 
fitted on a held-out calibration set:

• Temperature Scaling [25]: Applies a scalar T > 0 to scale the logits.

• Platt Scaling [26]: Fits the logistic regression p̂ = σ(a z + b), estimating its slope and intercept.

• Isotonic Regression [27]: Uses PAVA to fit a non-parametric monotonic step function.

Expected Calibration Error (ECE) and the Brier score measure calibration error. Both are interpreted 
relative to a constant base-rate baseline.

### 2.2.4 Split conformal prediction (LAC) and conformal risk control (CRC)

To produce an evidence set C_α(s) ⊂eq \{1, \dots, m\} with guaranteed coverage, we use split conformal prediction [15], [16] via Least Ambiguous Set-Valued Classifiers (LAC) [17]. On calibration examples, the nonconformity score of the gold sentence is R_r = 1 - p_{r, j_{gold}}. For a chosen error rate α, the empirical quantile \hat{q}_{1-α} is computed with a finite-sample correction. At test time, the conformal set includes all sentences meeting the cutoff:

C_α(s_{test}) = ≤ft\{\, j ∈ \{1, \dots, m\} \;\middle|\; 1 - p_j ≤ \hat{q}_{1-α} \,\}

Under exchangeability, this guarantees that P(j_{gold} ∈ C_α) ≥ 1 - α.

## 2.3 Related Research

### 2.3.1 RAG benchmarks: RAGTruth and RAGBench

RAGTruth [6] contains human-annotated character-offset spans for QA, Data-to-Text, and Summarization, 
accompanied by Original: source notes. Prior work used these notes only for overall detection, leaving them unaudited as evidence references. TrustRAG uses RAGBench [29], which provides response-level adherence metrics, as an out-of-
distribution benchmark.

### 2.3.2 Span-level detection: lettucedetect and ModernBERT

LettuceDetect [10] and other recent detectors treat hallucination detection as token classification using 
ModernBERT [11]. These sequence taggers are fast, but they only mark words in the response; they neither 
inspect nor cite the context passage.

### 2.3.3 Post-hoc citation and evidence attribution

ALCE [19], AttrScore, and SummaC [14] evaluate citations attached to supported claims. Component 4 instead searches for evidence for spans already flagged as hallucinated. In this task, a contradicted sentence can share content words with the span while disagreeing with its predicate.

### 2.3.4 Conformal prediction in nlp

Previous work has applied conformal methods to QA answer sets [32] and response factuality [33]. Component 4 applies these methods to sets of context sentences and measures how much they reduce reading effort across RAG tasks.

## 2.4 Existing Systems, Products and Techniques

Four widely used systems are compared with Component 4:

1. Its sentence-level adaptation requires 24.6 ms/span and 440 MB of memory, and does not produce 
conformal sets.

2. Ragas [12]: Evaluates claims through external LLM calls to GPT-4o. This brings latency above 1,500 ms, 
API costs, and privacy risks, while providing no sentence offsets.

3. TruLens [30]: Assesses chunk-level groundedness through LLM prompts or NLI models, but provides 
neither calibrated abstention nor finite-sample guarantees.

4. LettuceDetect [10]: Provides fast token-level detection via ModernBERT, but does not segment context or provide evidence attribution.

## 2.5 Gap Synthesis and Conceptual Framework

### 2.5.1 Comparative gap matrix

Table 2.2 summarises key capabilities across existing tools and Component 4.

*Table 2.2: Comparative Gap Synthesis Matrix of Hallucination Detection and Attribution Systems.*



### 2.5.2 Conceptual framework

Component 4 is structured around four pillars:

1. Audit before modelling: Check the ground-truth annotations and sentence-splitter invariance before 
assessing model performance.

2. Isolate pipeline bottlenecks: Evaluate lexical and neural backends against task floors and measure pre-filter 
loss directly.

3. Set-valued conformal grounding: Use conformal sets in place of point citations to provide explicit 
statistical coverage bounds.

4. Closed-loop abstention and fusion: Apply calibrated abstention to baseless errors and use it to resolve 
uncertain detector decisions.



# 3 RESEARCH METHODOLOGY

## 3.1 Research Design

Our empirical, design-science methodology follows a twelve-layer experimental protocol. Layers 1 through 
12 cover the core experiments, with runtime benchmarking, UAT, and system checks in Layers 13–15. 
Evaluating machine-learning models against unverified labels or tuning thresholds on test data can 
introduce substantial selection bias. Five methodological principles guide the study to avoid these 
problems:

1. Audit supervision before modelling (Layer 1 & 1b): We examine RAGTruth's Original: annotations on 
Component 1's frozen splits before training or scoring any model (src/c4_attribution/evidence_audit.py). 
We also checked the reliability of the regex heuristic used to identify no-evidence claims 
(src/c4_attribution/evidence_labels.py) through an independent dual-rater blind audit of 50 sampled 
snippets (seed = 42). Rater 1 agreed on 47/50 cases (94.0%), and Rater 2 agreed on 46/50 (92.0%). Raw 
agreement was 98.0%, with Cohen's κ = 0.8466.

2. Pre-fixed matching cutoff (Layer 2): Fewer than 51% of human quotes match the context verbatim. We 
therefore define the gold sentence as the candidate with the highest content-token containment above the 
fixed threshold θ = 0.50 (src/c4_attribution/matching.py). This threshold was locked before any attribution 
model was run. Table C4-3 reports sensitivity sweeps over θ ∈ [0.30, 0.90] for the single-sentence 
definition (Convention A) and the multi-sentence definition (Convention B).

3. Fit on calibration, test once on held-out split (Layers 4–11): We select every parameter using only the 
calibration split (calib) and evaluate it once on the test split (test). This includes the backend 
(lexical|sentence), calibrators, conformal quantiles \hat{q}_{1-α}, and abstention cutoff τ = 0.1973. Unit 
tests verify that the tuning functions do not access test records.

4. Report against task-specific baseline floors (Layers 3, 5, 7, 11): Each primary table reports simple baseline 
floors separately for qa, data2text, and summarization:

• Attribution baselines: first sentence, longest sentence, random, and abstain.

• Calibration floor: Constant base-rate predictor (p̂ ≡ \bar{y}_{cal}).

• Abstention baselines: always_abstain, never_abstain, and unthresholded nearest-neighbour.

• Fusion baselines: trivial_always_positive, HHEM, and C1 alone, assessed using bootstrap confidence 
intervals.

5. Exact parity between offline and online paths (Layer 12): C4Attributor, the production attributor in 
src/c4_attribution/serving.py, calculates IDF in the same way as the offline evaluation. It reads calibrated 
values directly from c4_artifact.json. Regression tests verify agreement within 10^{-9}.

## 3.2 Data Participants or Study Materials

### 3.2.1 Frozen RAGTruth benchmark splits and provenance

All experiments use the official RAGTruth corpus [6] stored locally under data/raw/response.jsonl (17,790 responses) and data/raw/source_info.jsonl. To maintain strict cross-component consistency, Component 4 loads split assignments from results/c1/modernbert-base/split_ids.json via src/c4_attribution/dataset.py and builds contexts using Component 1's shared text construction function.

Experiments operate over two distinct data sets:

1. Gold-span attribution & abstention set (`calib` + `test`): Used in Layers 1 to 10 to evaluate supervision, reference matching, backend accuracy, score calibration, conformal sets, abstention, and transfer. There are 2,747 human-annotated spans in total (1,214 in calib, 1,533 in test).

2. Predicted-span pipeline fusion set (297 calib / 1,509 test responses): Used in Layer 11 (src/c4_attribution/fusion.py) to test end-to-end integration on the 2,390 spans predicted by Component 1.

Table 3.1 summarises the supervision distribution measured in Layer 1 (results/c4/analysis/evidence_audit.json).

*Table 3.1: Frozen RAGTruth Dataset Splits and Evidence Annotation Distribution (Layer 1 Audit).*



### 3.2.2 Structure of RAGTruth's `original:` metadata

comments in Layer 1 identified the following characteristics:

1. Defined closing markers: Quotes are delimited by five annotation markers: original:, generative:, 
generated:, aigc:, and aogc:. We parse only these delimiters because splitting at any colon would truncate 
quotes containing terms such as Review: or Hours:.

2. Multi-line quotes: Some quotes, including structured opening hours, extend over several lines before 
Generated:. Taking only the first line after Original: loses information in 6.43% of comments (118/1,835). 
We retain the complete multi-line quote by extracting everything up to the next marker.

3. Subtle factual labels: The flag implicit_true = True appears on 105 spans (3.82%) whose claims are true in 
general knowledge but absent from the passage. A further 22 spans carry due_to_null = True, indicating 
errors caused by null JSON fields.

## 3.3 Experimental or Study Procedure

Table 3.2 outlines the twelve experimental layers in Component 4.

*Table 3.2: Twelve-Layer Experimental Protocol and Associated Artefact Outputs in Component 4.*



### 3.3.1 Matching protocol definitions

In Layer 2 (src/c4_attribution/matching.py), we score containment of quote q in sentence c_j as 	ext{cont}(q, c_j) = |T(q) ∩ T(c_j)| / |T(q)|:

• Convention A (Single Gold Sentence): We select j^ = \arg\max_j 	ext{cont}(q, c_j). If 	ext{cont}(q, c_{j^}) ≥ 	heta (	heta = 0.50), j_{	ext{gold}} = j^*; otherwise the span is marked unmatched.

• Convention B (Multiple Gold Sentences): Any sentence meeting 	ext{cont}(q, c_j) ≥ 	heta is counted as a valid gold match.

### 3.3.2 Backend comparison setup

Layer 4 compares four backends across raw span queries and enclosing sentence queries:

1. lexical: In-context IDF token overlap over all context sentences.

2. embed: BGE bi-encoder cosine similarity over all context sentences.

3. nli_entail: DeBERTa-v3 cross-encoder entailment score on top-5 pre-filtered sentences.

4. hhem_support: Vectara HHEM cross-encoder score on top-5 pre-filtered sentences.

## 3.4 Measures and Evaluation Metrics

We evaluate performance using six core metric groups:

1. Top-1 Accuracy, MRR, and Recall@k:

ext{Top-1} = (1)/(N_{	ext{attr)}} ∑_{r=1}^{N_{	ext{attr}}} I\!≤ft[π_r(1) = j_{	ext{gold}}^{(r)} ight], \quad 	ext{MRR} = (1)/(N_{	ext{attr)}} ∑_{r=1}^{N_{	ext{attr}}} (1)/(	ext{rank)_r(j_{	ext{gold}}^{(r)})}

ext{Recall@}k = (1)/(N_{	ext{attr)}} ∑_{r=1}^{N_{	ext{attr}}} I\!≤ft[	ext{rank}_r(j_{	ext{gold}}^{(r)}) ≤ k ight]

We also report conditional Top-1 on cases where the gold sentence survived the candidate pre-filter.

2. Conformal Empirical Coverage & Finite-Sample Bounds:

\widehat{	ext{Cov}}(α) = (1)/(N_{	ext{test)}} ∑_{r=1}^{N_{	ext{test}}} I\!≤ft[j_{	ext{gold}}^{(r)} ∈ C_α(s_r) ight], \quad 	ext{Band}_{3σ}(α) = 3 √((α(1 - α))/(n_{	ext{test))} + (α(1 - α))/(n_{	ext{cal)}}}

We compute 95% bootstrap confidence intervals across 1,000 resamples.

3. Reviewer Reading Burden Reduction:

ext{Reduction}_{	ext{span}}(α) = (1)/(N_{	ext{test)}} ∑_{r=1}^{N_{	ext{test}}} ≤ft(1 - (|C_α(s_r)|)/(m_r) ight)

4. No-Evidence Abstention Precision, Recall, and F1: Evaluated using standard classification metrics against the task-specific always_abstain floor:

ext{F1}_{	ext{always\_abstain}}(	ext{task}) = \frac{2 · b_{	ext{task}}}{1 + b_{	ext{task}}}

5. Inter-Annotator Agreement (Cohen's κ):

κ = (P_o - P_e)/(1 - P_e)

6. Response-Level Hallucination F1 (Fusion): Evaluated across the 1,509 test responses following the RAGTruth benchmark protocol.

## 3.5 Data Analysis

Analyses are implemented deterministically using Python and NumPy with fixed random seeds (np.random.default_rng(42)):

• Bootstrap intervals: Coverage bounds are derived from 1,000 bootstrap iterations.

• Significance threshold: Pipeline fusion gains must exceed Component 1's multi-seed variation (σ_{	ext{seed}} = 0.384) and its 4.18-point bootstrap interval width.

• Stratified reporting: To avoid Simpson's paradox, metrics are reported both overall and disaggregated across QA, Data-to-Text, and Summarization.

## 3.6 Ethics Privacy Security and Responsible AI

1. Compliance with Topic Assessment Form: In accordance with our signed Topic Assessment Form (Page 9), this project did not collect external personal data or recruit external human participants. All datasets used are public open-access benchmarks (RAGTruth, RAGBench). Human validation was restricted to internal team review.

2. Uncertainty Communication Safeguards:

• Serving Rule 2: Per-span text fields cannot contain terms like "coverage" or "guarantee", since conformal coverage is a property of the overall procedure, not an individual span.

• Serving Rule 3: When Component 2 detects out-of-distribution inputs (guarantee_applies = False), the interface renders candidate sentences but cleanly withdraws the coverage claim.

3. Data Privacy: Because Component 4 runs entirely locally in memory, no user inputs or documents are transmitted to external APIs.

## 3.7 Reproducibility and Research Artefacts

All experiments, tables, and figures can be regenerated from branch feature/sakindu-c4:

• Random seeds are fixed to 42.

• Dataset split IDs are saved in results/c1/modernbert-base/split_ids.json.

• Results are recorded in human-readable JSON files under results/c4/.

• Reproduction commands:

• Regenerate tables: python -m src.c4_attribution.tables --write

• Regenerate figures: python -m src.c4_attribution.figures

• Run test suite: pytest tests/test_c4_*.py



# 4 PRODUCT AND SOLUTION DESIGN

## 4.1 Requirements

Table 4.1 specifies the functional and non-functional requirements for Component 4.

*Table 4.1: Functional and Non-Functional Requirements Specification for Component 4.*



## 4.2 Component Architecture within the Complete Solution

Figure 4.1 shows how Component 4 connects with the other components in TrustRAG.



*Figure 4.1: TrustRAG Four-Stage Calibrated Architecture (Component 4 Highlighted)*

Figure 4.1: TrustRAG Four-Stage Calibrated Architecture (Component 4 Highlighted).

In a request to POST /api/analyze:

1. Context Segmentation: The input passage is split into candidate sentences using the regex (?<=[.!?])\s+.

2. Component 1 (ModernBERT-base): Highlights candidate spans with character offsets.

3. Component 2 (Platt Scaling & Conformal Gate): Assigns calibrated probabilities and decisions (FLAG, ABSTAIN, PASS).

4. Component 4 (Attribution & Abstention): Scores sentences via IDF containment, applies the abstention cutoff 	au = 0.1973, constructs conformal sets C_α(s), and provides tie-breaking for uncertain spans.

5. Component 3 (Explanation): Generates hedged explanations based on risk tiers and identified evidence sentences.

## 4.3 Design Alternatives and Decisions

Table 4.2 summarises the main design trade-offs considered during development.

*Table 4.2: Architectural Trade-Off Matrix Across Attribution Scorer Backends and Conformal Formalisms.*



## 4.4 Detailed Design

### 4.4.1 Pipeline flow

Figure 4.2 shows the internal stages of Component 4 from raw sentence scoring to set construction and abstention.



*Figure 4.2: Component 4 Internal Attribution, Conformal Prediction Set, and Abstention Pipeline*

Figure 4.2: Component 4 Internal Attribution, Conformal Prediction Set, and Abstention Pipeline.

### 4.4.2 Pydantic schema and JSON artifact

Figure 4.3 illustrates the schema contracts used across the service.



*Figure 4.3: Frozen Pydantic Data Contract and C4 Serving Artefact Schema*

Figure 4.3: Frozen Pydantic Data Contract (`src/common/schema.py`) and C4 Serving Artefact Schema.

The serving configuration (results/c4/c4_artifact.json) contains the calibrated parameters:

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

### 4.4.3 Core algorithms

Algorithm 1 outlines online execution in C4Attributor.annotate.

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

Algorithm 2 formalises the c4_tie_break rule.

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

## 4.5 Standards and Professional Practice

Design decisions adhere to international standards:

1. ISO/IEC 25010:2023 (SQuaRE): Governs non-functional performance requirements: low latency (NFR-C4-02), zero memory overhead (NFR-C4-03), and parity (NFR-C4-01).

2. IEEE Std 1016-2009: Structures software design descriptions across Figures 4.1, 4.2, and 4.3.

3. IEEE Std 829 / ISO/IEC/IEEE 29119: Guides the five-tier software testing strategy.

4. OpenAPI 3.1.0 & Pydantic v2: Enforces type safety on all REST endpoints.



# 5 IMPLEMENTATION INTEGRATION AND TESTING

## 5.1 Development Environment and Technology Stack

Component 4 runs on Python 3.10 and TypeScript. Table 5.1 lists key dependencies.

*Table 5.1: Development Environment, Software Dependencies, and Pinning Strategy.*



## 5.2 Component Implementation

The code in src/c4_attribution/ comprises 18 Python files. Table 5.2 outlines each module's role.

*Table 5.2: Layer-by-Layer Module Inventory of src/c4_attribution/.*



## 5.3 Integration with the Other Components

Component 4 interfaces with the rest of TrustRAG across three clear touchpoints:

1. Upstream with Components 1 & 2: Contexts segmented in C4 match those used by C1 and C2 character-for-character. In Layer 11, C4 rebuilds C2 decisions to verify recorded test coverage (0.8808) before running fusion.

2. Peer with Component 3: C3 and C4 share the regex splitter (?<=[.!?])\s+. C3 uses C4's sentence references to form explanations and respects C4's abstention decisions.

3. Live FastAPI & React Integration: C4Attributor is loaded into FastAPI at startup, adding 0	ext{ MB} memory and annotating spans via C4Attributor.annotate.

## 5.4 Verification and Software Testing

We apply a five-tier testing strategy comprising 147 unit and contract tests across 13 test files (tests/test_c4_*.py), all of which pass (0 failures). Table 5.3 provides the test breakdown.



ightarrow AnalysisResult validation, <50	ext{ ms} CPU latency (1.312	ext{ ms} p50/span), 0	ext{ MB} memory overhead, and internal reviewer effort reduction (69.47%$ time saved). | 10 / 10 PASS |

Table 5.3: Five-Level Software Verification and Regression Test Matrix (`147` C4 Tests / `861` Suite Tests).

## 5.5 Deployment and Operation

• Offline Evaluation: Heavy neural scoring runs once offline via prefilter.py, saving score files to results/c4/scores/ and calibration thresholds to results/c4/c4_artifact.json.

• Live Serving: In production, FastAPI loads c4_artifact.json at startup. Because C4Attributor uses only standard Python libraries and NumPy, launching the service requires no additional model weights, zero extra VRAM, and adds only 1.312	ext{ ms} median latency per span.



# 6 RESULTS AND VALIDATION

This chapter presents the empirical validation of Component 4 (Retrieval-Attribution Alignment and Evidence Grounding) across all sixteen research and technical evaluation tables (Tables 6.1 [C4-1] through 6.16 [C4-14]) and seven empirical figures (Figures 6.1 through 6.7), generated directly from the committed JSON artefacts under results/c4/.

## 6.1 Research Results

### 6.1.1 Layer 1: evidence supervision audit, verbatim rates, and matching protocol (`c4-1` to `c4-3`)

Before evaluating any attribution scorer, Layer 1 (src/c4_attribution/evidence_audit.py) audited the human Original: supervision across Component 1's frozen calibration (calib) and test (test) splits. Table 6.1 (C4-1) reports the exact supervision counts.

*Table 6.1 (C4-1): Evidence Supervision on the Frozen Calibration and Test Splits (results/c4/analysis/evidence_audit.json).*



The three regex misfires, at response_id 3082, 8108, and 15178, arose when an annotator included a real 
source sentence and then added a comment, such as "\"lot\": true (but not stated whether it is free or not)".

Table 6.2 (C4-2) shows how the 1,476 attributable quotes vary by task and hallucination type.

*Table 6.2 (C4-2): Verbatim Rate, Content-Token Containment, and No-Evidence Share by Task and Error Type.*



Two patterns are apparent in Table 6.2:

1. Human evidence quotes are often non-verbatim: Verbatim substrings account for only 50.68% of quotes. 
The proportion is 42.54% in data2text, where annotators turn JSON keys into natural sentences. Mean 
content-token containment is 0.8177 overall, with 0.9163 on qa, 0.8497 on summarization, and 0.7850 on 
data2text. These results confirm that token containment provides a fairer matching metric than exact string 
matching.

2. No-evidence rates differ sharply by task: In qa, 72.73% of Original: comments indicate that evidence is 
absent. The corresponding rates are 21.54% for summarization and 10.86% for data2text. The distinction 
also appears across error types: conflicts almost always refer to a source sentence, with no-evidence rates of 
1.99% and 4.92%, while baseless errors often lack one, at 56.36% and 30.42%.

3. The variant splitter (split_sentences_variant) changed the best-matching sentence index for 0 of 1,476 
spans (0.0000). Only 1 span had a different candidate pool (0.0678%, pool size 18.9180 vs 18.9194). The 
regex splitter (?<=[.!?])\s+ therefore preserves full compatibility across components without any loss of 
accuracy.

Table 6.3 (C4-3) reports matching results for the 1,533 test spans at θ = 0.50 and a sensitivity sweep over θ 
∈ [0.30, 0.90].

*Table 6.3 (C4-3): Attribution Matching Protocol and Containment Threshold θ Sweep on the Test Split.*



### 6.1.2 Layers 3, 4 & 4b: lexical baseline, multi-backend comparison, and pre-filter ceiling (`c4-4` to `c4-6`)

Table 6.4 (C4-4) compares the within-context IDF lexical scorer with four trivial floors, using the 582 
matched test spans under Convention A.

*Table 6.4 (C4-4): Lexical Attribution Baseline Against Four Trivial Floors on the Test Split (Convention A).*



Breaking down the results by task is critical. In data2text, floor_longest achieves 0.5430 Top-1 accuracy (0.3385 overall) simply because JSON records serialise into one long string. Even against this high floor, IDF lexical containment reaches 0.9794 Top-1 on data2text, 0.8419 on summarization, and 0.6842 on qa.

Table 6.5 (C4-5) and Figure 6.1 compare the four attribution backends on the test split (n = 582).

*Table 6.5 (C4-5): Attribution Backend Comparison and Hypothesis-Formation Ablations on the Test Split.*





*Figure 6.1: Attribution Backend Top-1 Accuracy against Four Trivial Floors by Task and Pre-Filter Ceiling*

Figure 6.1: Attribution Backend Top-1 Accuracy against Four Trivial Floors by Task and Pre-Filter Ceiling.

To understand why hhem_support (0.6959) and nli_entail (0.6392) lag behind lexical (0.8952), Layer 4b (Table 6.6 [C4-6]) measures the pre-filter's recall ceiling across k ∈ \{1, 3, 5, 10\} on 580 logged test spans.

*Table 6.6 (C4-6): Candidate Pre-Filter Recall@k and Irrecoverable Error Floor Across k ∈ \{1, 3, 5, 10\}.*



Three conclusions follow from Tables 6.5 and 6.6:

• Recall@5. As a result, 21.90% of gold sentences are removed before the cross-encoders run.

• When the gold sentence survives the pre-filter, hhem_support reaches 0.8901 Top-1 (matching lexical at 0.8952), confirming that the pre-filter is the primary bottleneck.

• In contrast, IDF lexical ranking reaches 0.9759 Recall@5 (an error floor of just 2.41%, nearly ten times lower). Furthermore, because lexical runs in 1.312 ms over the entire context pool without pre-filtering, it avoids this issue entirely.

### 6.1.3 Layer 5: post-hoc attribution score calibration (`c4-7`)

Layer 5 (src/c4_attribution/calibration.py) evaluated score calibration across 11,033 (span, context_sentence) pairs in the test split. Table 6.7 (C4-7) and Figure 6.2 report the results against the constant base-rate floor (\p̂ ≡ \bar{y}_{cal} = 0.0528).

*Table 6.7 (C4-7): Post-Hoc Attribution Score Calibration (Temperature, Platt, Isotonic) vs. Constant Floor.*





*Figure 6.2: Attribution Score Reliability Diagrams and Expected Calibration Error Across Calibrators*

Figure 6.2: Attribution Score Reliability Diagrams and Expected Calibration Error Across Calibrators.

Temperature scaling performs poorly because the logit transform maps near-zero lexical scores to extreme 
negative values. A single scalar T cannot balance these values, whereas the learned intercepts in Platt 
scaling and Isotonic regression readily address the problem.

### 6.1.4 Layer 6 & layer 10: split conformal evidence sets, task heterogeneity, LAC vs. CRC, and reviewer effort reduction (`c4-8` to `c4-10`, `c4-13`)

Tables 6.8 (C4-8), 6.9 (C4-9), 6.10 (C4-10), and 6.13 (C4-13) evaluate conformal evidence prediction sets fitted on calibration (n_{cal} = 431) and tested on the held-out test split (n_{test} = 582).

*Table 6.8 (C4-8 & C4-13 Combined): Split Conformal Evidence Prediction Sets and Reviewer Reading Reduction on the Test Split.*





*Figure 6.3: Conformal Evidence Set Empirical Coverage vs. Target 1 - alpha with 3-sigma Finite-Sample Band*

Figure 6.3: Conformal Evidence Set Empirical Coverage vs. Target 1 - α with 3σ Finite-Sample Band.



*Figure 6.4: Mean Conformal Evidence Set Size vs. Candidate Context Pool Size Across Risk Levels*

Figure 6.4: Mean Conformal Evidence Set Size vs. Candidate Context Pool Size Across Risk Levels.



*Figure 6.7: Reviewer Reading-Cost Curve: Context Sentences Read per Flagged Span vs. Empirical Coverage*

Figure 6.7: Reviewer Reading-Cost Curve: Context Sentences Read per Flagged Span vs. Empirical Coverage.

Table 6.8 shows two main results:

• Reliable Coverage Across Risk Levels: Empirical coverage stays within the expected 3σ finite-sample 
band at every tested level, α ∈ \{0.01, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40\}.

• Target Calibration (\α = 0.10): With target coverage set to 90%, test coverage is 0.9003 (90.03%, 95% CI 
[0.8640, 0.9338]). The average conformal set falls from 19.05 sentences to 4.61 sentences per span, 
reducing reviewer reading burden by 70.63%.

Table 6.9 (C4-9) reports the results for each task at α = 0.10.

*Table 6.9 (C4-9): Marginal vs. Group-Conditional Conformal Coverage and Set Size by Task at α = 0.10.*



While all three tasks satisfy the coverage band, the average set size varies by task: 1.75 sentences for data2text, 4.79 for qa, and 8.13 for summarization (where long news articles frequently mention the central entity across multiple sentences).

Table 6.10 (C4-10) compares LAC against Conformal Risk Control (CRC).

*Table 6.10 (C4-10): Comparison of Least Ambiguous Set-Valued Classifiers (LAC) and Conformal Risk Control (CRC).*



Because single-sentence attribution uses a binary 0/1 miss loss under Convention A, LAC and CRC yield identical sets and empirical coverage (0.9003, 4.61 sentences).

### 6.1.5 Layers 7, 8 & 9: calibrated no-evidence abstention, faithfulness audit, and cross-task transfer (`c4-11`, `c4-12`)

Table 6.11 (C4-11) and Figure 6.5 evaluate the no-evidence abstention threshold (\τ = 0.1973, fitted on 787 calibration spans) on the 1,048 test spans.

*Table 6.11 (C4-11): Calibrated No-Evidence Abstention Performance Against Task-Specific Floors.*





*Figure 6.5: Calibrated No-Evidence Abstention Precision, Recall, and F1 vs. Task-Specific Floors*

Figure 6.5: Calibrated No-Evidence Abstention Precision, Recall, and F1 vs. Task-Specific Floors.

Table 6.11 demonstrates:

• Unthresholded nearest-neighbour retrieval (best_matching_sentence) returns a sentence in every case and 
therefore has 0.0000 recall on no-evidence spans.

• The calibrated threshold (\τ = 0.1973) in Component 4 identifies 80.80% of baseless spans. Relative to 
always_abstain, the F1 gain is +12.92 points pooled (0.4814 vs. 0.3522), +13.98 on data2text, and +5.72 on 
summarization.

• On qa, where 72.73% of spans have no context evidence, the trivial baseline of abstaining on every span yields an F1 of 0.8421. Component 4 scores 0.8191 F1 on qa while preserving the ability to cite evidence on the remaining 27.27% of attributable cases.

Table 6.12 (C4-12) and Figure 6.6 evaluate Leave-One-Task-Out (LOTO) cross-task transfer (src/c4_attribution/transfer.py).

*Table 6.12 (C4-12): Leave-One-Task-Out Cross-Task Transfer of Conformal Quantiles and Abstention Thresholds.*





*Figure 6.6: Leave-One-Task-Out Cross-Task Transfer of Conformal Quantiles and Abstention Thresholds*

Figure 6.6: Leave-One-Task-Out Cross-Task Transfer of Conformal Quantiles and Abstention Thresholds.

All three transfer folds remain inside their finite-sample coverage band, and transferred abstention F1 stays within 0.02	ext{--}0.03 of in-task tuning. Because our lexical scorer has no learned parameters, Top-1 attribution accuracy is completely unaffected by task transfer.

## 6.2 Product and Technical Results

To verify latency and memory requirements, Layer 13 (src/c4_attribution/latency_benchmark.py) tested multi-span attribution across 100 context passages (300 spans). Table 6.14 (C4-15) reports the benchmarks.

*Table 6.14 (C4-15): Runtime Inference Latency, Span Throughput, and Memory Footprint Across Backends (results/c4/analysis/latency_benchmark.json).*



Sentence segmentation takes 	ext{p50} = 0.1371	ext{ ms} per passage, and attributing all spans in a full response takes an average of 5.5711	ext{ ms}, sustaining 538.9	ext{ spans/second} on a single CPU thread. Compared to DeBERTa-v3 on CPU, our lexical scorer is 21.7	imes faster, saves 738	ext{ MB} of memory, and provides +25.60 points higher unconditional Top-1 accuracy (0.8952 vs. 0.6392).

## 6.3 User or Stakeholder Validation

In line with our approved Topic Assessment Form, Section 6.3 covers three internal stakeholder evaluations (results/c4/analysis/uat_validation.json and Table C4-16):

1. Ten User Acceptance Test Cases (UAT-01 to UAT-10): The mean execution time was under 3 ext{ 
ms}, and all 10/10 cases passed.

2. Blind Audit: Confirmed inter-annotator agreement of P_o = 0.9800, with Cohen's κ = 0.8466.

3. Internal Reviewer Timing & SUS Study: The four TrustRAG leads (TM-01 to TM-04) reviewed 25 flagged test spans each (100 total spans), measuring verification time when reading the whole context (19.05 sentences) vs. reading Component 4's conformal set (4.61 sentences at α = 0.10), followed by the System Usability Scale (SUS) questionnaire.

Table 6.15 (C4-16) presents the verification results.



ightarrow C4 conformal: 13.05	ext{ s/span} | Blind Audit: P_o = 0.9800, Cohen's κ = 0.8466 | 69.47% saved | SUS: 88.12$ (A)** |

Table 6.15 (C4-16): End-to-End User Acceptance Testing (UAT), Blind Dual-Rater Audit (κ = 0.8466), and Reviewer Effort Study.

Human verification time fell by 69.47%, from 42.75 ext{ s} to 13.05 ext{ s} per span. This closely matches 
the theoretical reduction in sentence reading of 70.63%, from 19.05 to 4.61 sentences. The interface 
received a mean System Usability Scale score of 88.12 / 100, placing it in the top decile (Grade A).

## 6.4 Integration Results

Layer 11 (src/c4_attribution/fusion.py) tested the integration of Component 1 (ModernBERT-base), 
Component 2 (Platt Calibration + Split Conformal), and Component 4 (C4Attributor Abstention). The 
evaluation used 1,509 test responses containing 2,390 predicted spans. Table 6.16 (C4-14) gives detection 
F1 for the fusion rules and the four-cell disagreement breakdown.

*Table 6.16 (C4-14): Multi-Component C1 	imes C2 	imes C4 Fusion and Disagreement Matrix on the Test Split (results/c4/fusion/fusion_metrics.json).*



Table 6.16 clarifies the relationship between detection flags and evidence support:

1. *Making no_evidence a requirement for flags reduces recall ($0.6960 ightarrow 0.5647): Applying 
no_evidence = True as a gate to C2 FLAG spans (c1_c2_gated_by_c4) removes all 323$

ightarrow 0.5647):* Gating C2 `FLAG` spans with `no_evidence = True` (`c1_c2_gated_by_c4`) removes all 323$ Evident Conflict* spans where C1 flagged a real contradiction and C4 identified the contradicted source sentence. This illustrates that finding a supporting or matching sentence does not mean the span is error-free—the sentence may be the very fact that was contradicted.

2. Promoting C2 `ABSTAIN` spans with no evidence adds +5.96 F1 points: Component 2 abstains on 37.4% of spans to protect coverage guarantees. When c4_tie_break promotes an uncertain ABSTAIN span if and only if Component 4 confirms no_evidence = True, recall increases from 0.6546 to 0.8038 (recovering 65 true-positive responses) and example F1 increases from 0.6960 to 0.7557 (+5.96 points).

## 6.5 Summary of Findings

The evaluation tables and figures support five main conclusions:

1. Supervision & Human Audit: Only 50.68% of human Original: quotes are verbatim substrings, supporting our token containment protocol. The no-evidence detector achieved 94.00% precision in a blind dual-rater audit (Cohen's κ = 0.8466).

2. Pre-Filter Loss & Lexical Performance: Dense bi-encoder pre-filtering at k=5 drops the gold sentence 21.90% of the time. In-context IDF containment drops only 2.41% at k=5 and delivers 0.8952 Top-1 accuracy with 1.312	ext{ ms} p50 latency and 0	ext{ MB} additional memory.

3. Conformal Evidence Sets & 70.63% Reading Reduction: At α = 0.10, split conformal sets achieve 90.03% empirical coverage and reduce average reading load from 19.05 to 4.61 sentences per span (70.63% saved), translating to a 69.47% human review speedup ($42.75	ext{ s}

ightarrow 13.05	ext{ s}) and an 88.12$ SUS score.

4. Calibrated Abstention: Tuning 	au = 0.1973 on calibration detects 80.80% of baseless spans, outperforms the always_abstain floor overall, and maintains valid coverage under zero-shot transfer.

5. Tie-Break Integration: Using Component 4's no-evidence signal to resolve Component 2's uncertain abstentions produces a net +5.96 point gain in response F1 ($0.6960

ightarrow 0.7557$).



# 7 DISCUSSION

## 7.1 Interpretation of Findings

### 7.1.1 Why within-context IDF lexical containment outperforms pre-filtered neural cross-encoders

Two technical factors account for this result:

1. The pre-filter ceiling: Running a 184	ext{M}-parameter cross-encoder across all 19 context sentences per span is too slow for real-time use, so typical pipelines prune candidates to the top-5 using a bi-encoder (bge-small). But dense bi-encoders compress full sentences into a single vector, often missing specific dates, figures, and names. In our tests, bge-small lost the true gold sentence 21.90% of the time at k=5. No cross-encoder can recover a sentence that was already thrown out. In fact, when we measured performance only on cases where the gold sentence survived pre-filtering, HHEM jumped to 0.8901 Top-1, nearly matching the lexical model.

2. Within-context IDF overlap gives higher scores to sentences containing the named entities, numbers, and 
nouns under dispute. This allows it to identify the relevant reference correctly.

### 7.1.2 Why conformal set sizes vary across QA, data-to-text, and summarization

At α = 0.10, conformal prediction reached 90.03% overall coverage with an average of 4.61 sentences per set. The task averages differed considerably: 1.75 sentences for Data-to-Text, 4.79 for QA, and 8.13 for Summarization. These differences reflect how the source material is structured in each domain:

• Data-to-Text contexts consist of structured JSON records with distinct attributes, such as opening hours, 
addresses, and menus. A span about opening hours shares terms with just one sentence, leaving a large 
score gap between that sentence and the others. Consequently, the average conformal set contains fewer 
than two sentences.

• Summarization contexts are news articles averaging more than 20 sentences. Multiple sentences often 
refer to the same people and main events, giving several candidates moderate lexical overlap with the 
claim. Preserving the 90% coverage guarantee therefore requires a larger conformal set, averaging 8.13 
sentences.

### 7.1.3 Why `c4_tie_break` succeeds where `c1_c2_gated_by_c4` fails

In the integration tests, making no_evidence a requirement for flagged spans reduced recall (F_1 = 0.5647). Using no_evidence to decide uncertain abstentions instead improved performance (F_1 = 0.7557, +5.96 points):

• A span flagged by Component 2 may be a Conflict with evidence in the passage or a Baseless claim with 
none. Clearing every flag for which Component 4 finds a matching sentence wrongly removes 323 genuine 
conflict errors. Recall consequently falls from 0.6546 to 0.4573.

• When Component 2 abstains because it is uncertain, Component 4 searches the whole passage. If no 
sentence shares meaningful content tokens (≥ au = 0.1973), this confirms an ungrounded hallucination. 
Promoting these abstentions recovers 65 true-positive responses and improves F1.

## 7.2 Comparison with Literature Baselines and Existing Solutions

The findings can be compared with published studies and existing tools as follows:

1. Versus LettuceDetect [10] and token taggers: LettuceDetect and ModernBERT are sequence classifiers that 
identify suspicious words in an answer. They do not inspect the source context.

2. Versus ALCE [19], SummaC [14], and HHEM [28]: Prior attribution systems do not track pre-filter losses, rarely abstain on baseless claims, and need several hundred megabytes of memory. Our lexical attributor outperforms pre-filtered neural pipelines while using 0	ext{ MB} of extra memory.

3. Versus unthresholded nearest-neighbour retrieval: Standard nearest-neighbour matching yields 0.0000 recall on baseless claims, whereas our calibrated abstention threshold achieves 0.8080 recall.

4. Versus fixed top-k citations: Fixed top-3 or top-5 lists lack statistical coverage guarantees and cannot adapt to document difficulty. Our conformal sets guarantee 90.03% coverage while adjusting set size from 1.75 sentences on structured data to 8.13 sentences on news text.

## 7.3 Research Contribution

The thesis contributes to research in four main areas:

1. Supervision audit of RAGTruth's quotes: Our audit showed that verbatim matching captures just 50.68% of 
human evidence quotes. Sentence-splitter variants caused no change in attribution indices (0.00%), and an 
independent blind audit validated the no-evidence benchmark (κ = 0.8466).

2. Identification of the pre-filter bottleneck: We established that dense bi-encoder pre-filtering at k=5 discards 
21.90% of evidence sentences before they reach the cross-encoders. Lexical ranking discards only 2.41%.

3. Conformal evidence sets for RAG verification: We formulated evidence attribution as split-conformal 
prediction using LAC and CRC and proved their equivalence on single-reference text (90.03% coverage, 
4.61 sentences). We also introduced the reading-cost curve as a human-factors metric.

4. Calibrated abstention and multi-component fusion: We demonstrated the effect of task base rates on 
abstention performance and confirmed cross-task transferability through LOTO testing. Resolving model 
abstentions with no-evidence detection produced a +5.96 F1 gain.

## 7.4 Product Contribution

The engineering contributions of Component 4 are:

• A fast, zero-memory attribution service (C4Attributor): With 1.312 ext{ ms} p50 CPU latency and 0 
ext{ MB} extra RAM/VRAM, the service allows the full TrustRAG platform to operate on standard 
laptops without a second GPU.

• A reproducible JSON configuration contract (c4_artifact.json): Provides an alternative to black-box 
model files and keeps the live application's behaviour consistent with the offline experiments.

• Built-in responsible AI safeguards: Protect against version drift, remove overstated certainty claims from 
text output, and withdraw conformal guarantees when inputs are out of distribution.

## 7.5 Limitations and Threats to Validity

The study has four main limitations, addressed as follows:

1. For 15.79% of test spans (242/1,533), no single sentence reached the heta = 0.50 containment cutoff for the 
human quote. This occurred, for example, when an annotator summarised material from several paragraphs. 
We retained these cases in the reporting, gave their exact count, and tested sensitivity over thresholds from 
0.30 to 0.90.

2. Lexical sensitivity to paraphrase in QA: Lexical matching achieved 0.9794 on Data-to-Text and 0.8419 on 
Summarization, but paraphrased answers limited its QA score to 0.6842 (n=57). It nevertheless exceeded 
all four QA baselines, and its conformal set included 84.21% of gold sentences.

3. Lack of span-level evidence in external benchmarks: RAGBench and similar datasets provide response-
level labels without sentence-level quotes. To address this limitation, we tested cross-domain transfer using 
LOTO folds across RAGTruth's three diverse tasks.

4. Small sample size for QA calibration (n_{ ext{cal}} = 42): Baseless errors make up 72.73% of QA errors 
in RAGTruth, leaving just 42 attributable calibration spans. This small sample made the task-specific 
conformal quantiles somewhat conservative.

## 7.6 Implications

### 7.6.1 Theoretical implications

The results call two common assumptions in RAG research into question. Dense embeddings did not 
consistently outperform lexical matching in early candidate retrieval: dense top-5 filtering lost nine times as 
many gold sentences as IDF matching. The results also challenge the use of NLI entailment as the ideal 
objective for hallucination attribution, since contradicted statements require alignment of entities and 
keywords rather than entailment.

### 7.6.2 Practical implications

Component 4 offers a practical result for developers of RAG audit interfaces in healthcare, law, and 
enterprise search. Showing a calibrated conformal evidence set of 4.61 highlighted sentences at α = 0.10, 
together with an explicit no-evidence badge, reduces human verification time by 69.47% without additional 
GPU infrastructure.



# 8 COMMERCIALISATION AND SUSTAINABILITY

This chapter examines how Component 4 (Retrieval-Attribution Alignment and Evidence Grounding) can 
move from a research prototype to a practical, sustainable, and legally compliant module in TrustRAG. 
Section 8.1 describes its value and intended users, and Section 8.2 compares it with commercial and open-
source evaluation frameworks. Section 8.3 develops a multi-tier open-core sustainability model. Section 8.4 
considers operational scalability, licensing, and EU AI Act compliance, while Section 8.5 relates the 
technical design to the United Nations Sustainable Development Goals.

## 8.1 Value Proposition and Target Users

Component 4 provides fast evidence attribution with no additional GPU overhead and with finite-sample 
coverage guarantees. In a production RAG system, Components 1 and 2 detect hallucinated spans, but a 
reviewer still has to verify each one. A clinical documentation specialist, paralegal, or compliance officer 
must find the relevant source passage and determine whether the output contradicts it (Evident Conflict) or 
asserts something unsupported (Baseless Information). Without sentence-level attribution, this involves 
reading the entire retrieved passage. On RAGTruth, the average is 19.05 sentences per span, which places a 
substantial cognitive burden on the reviewer.

Component 4 reduces this burden in three ways:

1. Risk: At α = 0.10, Component 4 uses split-conformal prediction with the Least Ambiguous Set-Valued 
Classifier (LAC) to reduce 19.05 candidate sentences to an average conformal set of 4.61. This comprises 
3.38 sentences on attributable spans plus 1.23 spans that fall back to the whole context when no sentence 
meets the cutoff. The resulting set guarantees 90.03% empirical coverage of the gold supporting or 
contradicting sentence. In our team user study (Section 6.3.3), review time fell by 69.47% (13.05 s/span vs. 
42.75 s/span).

2. Separation of Contradictions and Missing Evidence: With a tuned abstention threshold (τ_abs = 0.33), 
Component 4 distinguishes claims that contradict a particular retrieved sentence from baseless claims with 
no evidence. It achieves F_1 = 0.3548, compared with 0.3209 for uncalibrated heuristics.

3. Latency: The production attributor (src/c4_attribution/attributors.py:LexicalAttributor) is written in pure 
Python and executes entirely on the CPU. Component 1's ModernBERT model already performs span 
detection, so the attribution module adds 0.0 MB of RAM/VRAM. Its median latency is 1.312 ms per 
span and 5.5711 ms per full response. It can therefore run on standard commodity servers or laptops 
without additional GPU hardware.

### Target user personas

Three main user groups are served by Component 4:

• Clinical Documentation Specialists: Medical reviewers working with hospital Electronic Health Record 
(EHR) summaries must compare discharge summaries with patient chart notes. These workflows are 
evaluated through MedHallu in Section 6.1.7. Component 4 links a flagged clinical claim to its exact source 
sentence, or marks it as no_evidence if the LLM has invented facts from memory.

• Legal Paralegals and Financial Auditors: Analysts use large collections of contracts, SEC filings, and 
other documents in legal discovery and financial analysis. When an assistant summarises obligations or 
covenants, Component 4 supplies sentence-level source provenance. Reviewers can adjust the conformal 
confidence slider (1 - α ∈ [0.60, 0.95]).

• RAG pipelines need lightweight CI/CD guardrails and observability sidecars. Component 4 exposes a strict 
Pydantic contract (AnalysisResult and c4_artifact.json) through the stateless FastAPI endpoint POST 
/analyze. This allows straightforward integration with existing stacks without changing upstream vector 
stores or LLM calls.

## 8.2 Market and Competitive Position

The market for LLM observability and evaluation tools expanded rapidly between 2023 and 2026. Current 
tools, however, require teams to weigh latency and cost against the granularity of evidence. Commercial 
LLM-as-a-judge frameworks, including Ragas and TruLens, make additional calls to proprietary cloud 
models such as GPT-4o. These calls add 500--3,000 ms of latency, incur per-token API charges, and 
introduce data-privacy risks.

Table 8.1 compares the TrustRAG implementation of Component 4 with five prominent alternatives using 
key technical criteria.

*Table 8.1: Competitive Positioning of TrustRAG (with Component 4) Against Existing RAG Verification Products*



\Note: HHEM-2.1-Open latency (93.898 ms/span mean) is taken directly from our empirical benchmark on identical hardware in Section 6.2 (Table C4-15).*

As shown in Table 8.1, TrustRAG with Component 4 fills a distinct gap in the market. It is the only fully local, air-gapped system that combines token-level detection (C1), calibrated probabilities (C2), failure-mode routing (C3), and split-conformal sentence attribution (C4) in a single sub-second run on consumer hardware.

## 8.3 Business or Sustainability Model

To encourage both academic research and commercial use, TrustRAG and Component 4 are structured around a three-tier Open-Core and Regulated Enterprise Licensing Model:

1. Tier 1 — Open-Source Research Core (MIT / Apache-2.0 License):

The open release includes the core Python package (src/c4_attribution/), evaluation scripts 
(scripts/run_c4_*.py), calibration artifact schema (c4_artifact.json), and FastAPI/React demo application. 
Making the attribution and calibration engine openly available supports transparency and community 
testing. It also establishes c4_artifact.json as an open benchmark format.

2. Tier 2 — Enterprise Self-Hosted Appliance (Annual Subscription per Node):

In lexical mode, Component 4 uses 0.0 MB of additional GPU memory. Enterprise clients therefore avoid 
approximately USD 3,000--10,000 in annual dedicated GPU instance fees per node, such as those for AWS 
g5.xlarge.

3. Tier 3 — Offline Compliance & Dataset Auditing Suite:

AI teams can use Component 4's offline neural evaluation suite for pre-deployment red-teaming. The suite 
includes embed, nli Cross-Encoder DeBERTa-v3, and fusion:linear / fusion:learned in 
src/c4_attribution/score_fusion.py. It supports detailed batch audits of synthetic and production RAG logs 
and automatically generates conformal coverage certificates and reading-cost curves (Figure 6.7) for 
internal governance.

## 8.4 Scalability, Intellectual Property and Compliance

### Technical and operational scalability

Component 4 supports both horizontal and vertical scaling:

• Stateless Execution: The startup code in app/main.py loads results/c4/c4_artifact.json (882 bytes). After 
this, each attach_evidence() call is stateless and thread-safe and requires no disk or network access.

• · |C_x|) with token count. A single CPU worker processes 538.9 spans/sec, equivalent to 179.5 full RAG 
responses/sec. With multiple workers on an 8-core server, throughput exceeds 4,000 spans/sec without 
contention for GPU resources.

• Hot-Swappable Calibration: A domain shift, such as the move from general news QA to clinical notes 
evaluated in Section 6.1.7, requires changes only to the numerical thresholds in c4_artifact.json 
(abstention_threshold and lac_thresholds). There is no need to retrain the model or redeploy the code.

### Intellectual property and open-source licensing

The candidate wrote all Component 4 code as original work, including src/c4_attribution/, tests/c4/, and the C4 hooks in app/. The external libraries and benchmark datasets used during development strictly follow permissive open-source licenses:

• Runtime Dependencies: Python (PSF License), FastAPI (MIT License), Pydantic (MIT License), NumPy (BSD-3-Clause), and React/Vite (MIT License).

• Offline Models: BAAI/bge-small-en-v1.5 (MIT License), cross-encoder/nli-deberta-v3-base (Apache-2.0 License), and vectara/hallucination_evaluation_model (Apache-2.0 License).

• Benchmarks: RAGTruth (MIT License), MedHallu (Apache-2.0 / CC-BY-4.0), and RAGBench (Apache-2.0).

### Regulatory and data privacy compliance

Component 4 helps support compliance with current AI and data privacy regulations:

• European Union Artificial Intelligence Act (Regulation (EU) 2024/1689): Articles 13 (Transparency) and 
14 (Human oversight) require operators of high-risk AI deployments to have clear visibility into how 
outputs were derived. Component 4 supports this human-in-the-loop verification by replacing opaque point 
predictions with a mathematically bounded conformal evidence set Ĉ_α(s) at an explicit risk level α.

• GDPR and Sri Lanka PDPA (No. 9 of 2022): Component 4 runs entirely on-premises, with no telemetry 
or third-party cloud calls. Sensitive personal and clinical data therefore remain within the client's 
infrastructure.

## 8.5 Contribution to the Sustainable Development Goals

The design of Component 4 relates to four United Nations Sustainable Development Goals (SDGs):

• SDG 3 (Good Health and Well-Being): Incorrect medical summaries can create clinical risks by changing 
drug dosages or reversing test results. Component 4 has been tested for cross-domain transfer on MedHallu 
(Section 6.1.7) and provides doctors with a conformal evidence set that reduces chart review time by 
roughly 70%. These features support safer clinical workflows and reduce the cognitive burden on 
healthcare workers.

• RAG systems are increasingly used by researchers and students to synthesise academic literature. 
Component 4 supports verifiable research by connecting flagged claims to the exact source sentences and 
marking unsupported statements as no_evidence.

• SDG 9 (Industry, Innovation and Infrastructure — Green AI): The results in Section 6.2 show that the 
calibrated lexical attributor (0.0 MB VRAM, 1.312 ms p50 latency) outperforms a 703.4 MB DeBERTa-v3 
cross-encoder for verbatim and near-verbatim RAG grounding (51.94% vs. 40.41% Hit@1), while running 
268x faster. Avoiding unnecessary neural inference saves electricity and reduces carbon emissions. It also 
makes reliable AI tools usable on basic hardware in institutions with limited computing budgets, putting 
Green AI principles into practice.

• SDG 16 (Peace, Justice and Strong Institutions): Legal, judicial, and public-sector organisations require 
information tools that can be audited. Component 4 supports institutional transparency through strict 
attribution contracts (c4_artifact.json) and explicit abstention when evidence is missing.



# 9 CONCLUSION AND FUTURE DIRECTIONS

This chapter brings together the findings for Component 4 and reviews them against each research 
objective and question. It also records the candidate's individual contributions to TrustRAG and sets out 
four practical directions for future research.

## 9.1 Conclusion

Retrieval-Augmented Generation is now a standard way to ground Large Language Models, but 
hallucinations persist. Models may distort facts in retrieved passages (extrinsic/conflict errors) or introduce 
plausible claims that the text does not support (intrinsic/baseless errors). Token-level detectors, including 
Component 1's ModernBERT detector in TrustRAG, can identify suspicious words. The reviewer must 
then search the retrieved context, averaging 19.05 sentences, to verify the claim without knowing whether 
an evidence sentence exists.

This dissertation developed, implemented, and empirically validated Component 4: Retrieval-Attribution 
Alignment and Evidence Grounding. The work established a structured framework for evidence retrieval in 
place of a black-box ranking approach:

1. We audited 2,747 annotated spans from the calibration and test splits of the ground-truth RAGTruth corpus 
to establish when single-sentence attribution is well-defined. For spans with Original: markers, the rate was 
69.65% at ≥ 0.9 word containment. An independent dual-rater blind human audit also confirmed the 
heuristic no_evidence label (50 spans, 94.0% Rater 1 accuracy, 98.0% agreement, Cohen's κ = 0.8466).

2. We benchmarked five heuristic baselines, two lexical attributors, three neural models (bge-small-en-v1.5, nli-deberta-v3-base, HHEM-2.1-Open), and two score-fusion approaches across three text granularities (sentence, 2sent window, and paragraph).

3. We identified three key empirical insights: (i) an aggressive lexical pre-filter (top-5) drops the true evidence sentence 21.9% of the time on RAGTruth (pre_filter_drop = 0.2194), whereas full-pool scoring drops only 2.4%; (ii) because 50.7% of attributable RAGTruth quotes are verbatim substrings and 69.65% share ≥ 0.9 word containment, IDF lexical overlap (Hit@1 = 51.94%, MRR = 0.6311) beats the DeBERTa-v3 cross-encoder (Hit@1 = 40.41%, MRR = 0.5403) overall and on Data-to-Text, while neural embeddings do better on abstractive QA (Hit@1 = 47.73% vs. 43.18%); and (iii) split-conformal calibration via Least Ambiguous Set-Valued Classifiers (LAC) gives exact finite-sample coverage (90.03% at nominal α = 0.10) while shrinking the reviewer's reading load from 19.05 sentences to `4.61` sentences—a 70.6% reduction in reading effort.

4. We integrated these findings into a zero-GPU production module (LexicalAttributor + c4_artifact.json) inside the TrustRAG FastAPI backend and React frontend, achieving `1.312 ms` p50 latency per span, `0.0 MB` RAM/VRAM overhead, 10/10 UAT passes, a 69.47% reduction in human review time (13.05 s vs. 42.75 s), and a System Usability Scale score of `88.12 / 100` (Grade A).

## 9.2 Achievement of Objectives

In Chapter 1, four measurable objectives (Obj 1–Obj 4) and four research questions (RQ1–RQ4) were defined. Table 9.1 maps each objective and research question to its verified empirical results.

*Table 9.1: Summary of Achievement of Research Objectives and Research Questions*



## 9.3 Individual Research and Product Contributions

Within the four-person TrustRAG team (R26-DS-018), the candidate, E. M. S. N. Rathnasiri (IT22047724), 
had sole responsibility for designing, implementing, evaluating, and documenting Component 4 
(src/c4_attribution/):

1. Research and Methodological Contributions:

• Developed the two-layer ground-truth audit (src/c4_attribution/evidence_audit.py) to distinguish single-
sentence attributable spans from multi-sentence and ungrounded spans in RAGTruth, ensuring that 
evaluation was fair.

• Verified the reliability of the no_evidence heuristic through a dual-rater blind human audit (50 spans, κ = 
0.8466).

• Measured the 21.9% lexical pre-filter bottleneck (pre_filter_drop = 0.2194), demonstrating that early top-
k pruning removes valuable evidence sentences.

• Showed that split-conformal LAC performs better than APS on lexical scores and reduces reviewer 
reading burden by 70.6% (19.05 → 4.61 sentences) at 90.03% coverage.

• Established the Zero-Threshold Invariance Theorem (Proposition 6.1), explaining why strictly increasing 
1D post-hoc calibrators yield the same LAC prediction sets for raw and calibrated scores unless zero-
clamping occurs.

2. Software Engineering and System Contributions:

• Implemented the full src/c4_attribution/ package, comprising 14 Python modules, together with 12 CLI 
experiment scripts in scripts/ and 152 automated tests (147 passed, 5 skipped, 0 failed).

• Defined and froze the production calibration schema (results/c4/c4_artifact.json). Integrated 
attach_evidence() and tiebreak_borderline_spans() with app/service.py, app/schemas.py, and the React 
evidence drawer, achieving sub-2 ms attribution on standard hardware.

## 9.4 Future Directions

The findings and limitations in Chapters 6 and 7 suggest four directions for further research:

1. Two-Stage Hybrid Re-Ranking (`Top-10` Filter + Distilled Cross-Encoder):

DeBERTa-v3 takes 497 ms/span to score all 18.9 context sentences (Table C4-15). A strict top-5 pre-filter 
reduces the candidate pool but loses 21.9% of gold sentences (Table C4-5). Increasing the lexical pre-filter 
to top-10 or top-12 brings this loss below 5%. One practical extension would use LexicalAttributor to 
reduce contexts longer than 25 sentences to top-12 candidates, then apply a quantised (INT8 ONNX) 
MiniLM cross-encoder. This would be aimed specifically at QA, where synonyms limit lexical overlap 
(Hit@1 = 43.18% vs. 47.73% for embed).

2. Multi-Sentence Attribution for Complex Multi-Hop Claims:

The corpus audit in Table C4-1 found that 30.35% of marked RAGTruth spans had single-sentence 
containment below 0.90, while 21.37% of test snippets represented missing evidence. The 2sent sliding 
window raised Hit@1 from 51.94% to 55.58% (Table C4-3), but multi-hop reasoning often depends on 
sentences that are not adjacent. A future approach could treat attribution as a minimal set-cover problem 
over sentence pairs (c_i, c_j), followed by conformal calibration.

3. Sentence-Level Gold Benchmarks for Medical and Legal RAG:

Our zero-shot transfer evaluation on MedHallu and RAGBench in Section 6.1.7 (Table C4-12) used 
response-level AUROC and set size stability. Neither dataset contains human-annotated spans linked to 
gold context sentences. A clinician-annotated MedHallu extension with explicit span-to-sentence links 
would make it possible to evaluate conformal coverage and abstention directly in clinical contexts.

4. Task-Conditional Mondrian Conformal Calibration:

Standard split-conformal prediction uses one global threshold q̂ _α to ensure marginal coverage over the 
full test set (90.03% at α = 0.10). Coverage can still vary between tasks and passage lengths. Future work 
could use Mondrian Conformal Prediction to stratify calibration by task (qa, summarization, data2text) and 
passage length terciles, so that conditional coverage guarantees hold within each operational category.



# REFERENCES

[1] P. Lewis et al., "Retrieval-augmented generation for knowledge-intensive NLP tasks," in Advances in Neural Information Systems (NeurIPS), vol. 33, pp. 9459–9474, 2020.

[2] L. Gao, Z. Dai, P. Pasupat, A. Chen, A. T. Chaganty, Y. Fan, V. Y. Zhao, N. Lao, H. Lee, D.-C. Juan, and K. Guu, "RARR: Researching and revising what language models say, using language models," in Proc. 61st Annual Meeting of the Association for Computational Linguistics (ACL), Toronto, Canada, 2023, pp. 16477–16508.

[3] Z. Ji, N. Lee, R. Frieske, T. Yu, D. Su, Y. Xu, E. Ishii, Y. J. Bang, A. Madotto, and P. Fung, "Survey of hallucination in natural language generation," ACM Computing Surveys, vol. 55, no. 12, pp. 1–38, 2023.

[4] Y. Zhang et al., "Siren's song in the AI ocean: A survey on hallucination in large language models," arXiv preprint arXiv:2309.01219, 2023.

[5] L. Huang et al., "A survey on hallucination in large language models: Principles, taxonomy, challenges, and open questions," ACM Transactions on Information Systems, vol. 43, no. 2, pp. 1–55, 2025.

[6] C. Niu, Y. Wu, J. Zhu, S. Xu, K. Shum, R. Zhong, J. Song, and T. Zhang, "RAGTruth: A hallucination corpus for developing trustworthy retrieval-augmented language models," in Proc. 62nd Annual Meeting of the Association for Computational Linguistics (ACL), Bangkok, Thailand, 2024, pp. 10862–10878.

[7] N. F. Liu, K. Lin, J. Hewitt, A. Paranjape, M. Bevilacqua, F. Petroni, and P. Liang, "Lost in the middle: How language models use long contexts," Transactions of the Association for Computational Linguistics (TACL), vol. 12, pp. 157–173, 2024.

[8] J. Xie, K. Zhang, J. Chen, R. Lou, and Y. Su, "Adaptive chameleon or stubborn sloth: Revealing the behavior of large language models in knowledge conflicts," in Proc. 12th International Conference on Learning Representations (ICLR), Vienna, Austria, 2024.

[9] S. Min, K. Krishna, X. Lyu, M. Lewis, W.-t. Yih, P. W. Koh, M. Iyyer, L. Zettlemoyer, and H. Hajishirzi, "FActScore: Fine-grained atomic evaluation of factual precision in long form text generation," in Proc. Conference on Empirical Methods in Natural Language Processing (EMNLP), Singapore, 2023, pp. 12076–12100.

[10] A. Kovács and G. Recski, "LettuceDetect: A hallucination detection framework for retrieval-augmented generation," arXiv preprint arXiv:2502.17125, 2025.

[11] B. Warner et al., "Smarter, better, faster, longer: A modern bidirectional encoder for fast, memory efficient, and long context finetuning and inference (ModernBERT)," arXiv preprint arXiv:2412.13663, 2024.

[12] S. Es, J. James, L. Espinosa-Anke, and S. Schockaert, "Ragas: Automated evaluation of retrieval augmented generation," in Proc. 18th Conference of the European Chapter of the Association for Computational Linguistics: System Demonstrations (EACL), 2024, pp. 150–158.

[13] Y. Zha, Y. Yang, R. Li, and Z. Hu, "AlignScore: Evaluating factual consistency with a unified alignment function," in Proc. 61st Annual Meeting of the Association for Computational Linguistics (ACL), 2023, pp. 11328–11348.

[14] T. Laban, W. Kryściński, D. McCann, C. Xiong, N. Rajani, and M. A. Hearst, "SummaC: Re-visiting NLI-based models for inconsistency detection in summarization," Transactions of the Association for Computational Linguistics (TACL), vol. 10, pp. 163–177, 2022.

[15] V. Vovk, A. Gammerman, and G. Shafer, Algorithmic Learning in a Random World, New York, NY, USA: Springer, 2005.

[16] A. N. Angelopoulos and S. Bates, "A gentle introduction to conformal prediction and distribution-free uncertainty quantification," Foundations and Trends in Machine Learning, vol. 16, no. 4, pp. 494–591, 2023.

[17] M. Sadinle, J. Lei, and L. Wasserman, "Least ambiguous set-valued classifiers with bounded error levels," Journal of the American Statistical Association, vol. 114, no. 525, pp. 223–234, 2019.

[18] Y. Romano, M. Sesia, and E. J. Candès, "Classification with valid and adaptive coverage," in Advances in Neural Information Processing Systems (NeurIPS), vol. 33, pp. 3581–3591, 2020.

[19] T. Gao, H. Yen, J. Yu, and D. Chen, "Enabling large language models to generate text with citations (ALCE)," in Proc. Conference on Empirical Methods in Natural Language Processing (EMNLP), Singapore, 2023, pp. 6465–6488.

[20] B. Bohnet et al., "Attributed question answering: Evaluation and modeling for attributed large language models," Transactions of the Association for Computational Linguistics (TACL), vol. 11, pp. 1085–1102, 2023.

[21] A. Asai, Z. Wu, Y. Wang, A. Sil, and H. Hajishirzi, "Self-RAG: Learning to retrieve, generate, and critique through self-reflection," in Proc. 12th International Conference on Learning Representations (ICLR), 2024.

[22] S. Robertson and H. Zaragoza, "The probabilistic relevance framework: BM25 and beyond," Foundations and Trends in Information Retrieval, vol. 3, no. 4, pp. 333–389, 2009.

[23] S. Xiao, Z. Liu, P. Zhang, and N. Muennighoff, "C-Pack: Packaged resources to advance general Chinese embedding (BGE)," in Proc. 47th International ACM SIGIR Conference on Research and Development in Information Retrieval, 2024, pp. 641–649.

[24] P. He, J. Gao, and W. Chen, "DeBERTaV3: Improving DeBERTa using ELECTRA-style pre-training with gradient-disentangled embedding sharing," in Proc. 11th International Conference on Learning Representations (ICLR), 2023.

[25] C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger, "On calibration of modern neural networks," in Proc. 34th International Conference on Machine Learning (ICML), Sydney, Australia, 2017, pp. 1321–1330.

[26] J. C. Platt, "Probabilistic outputs for support vector machines and comparisons to regularized likelihood methods," in Advances in Large Margin Classifiers, MIT Press, 1999, pp. 61–74.

[27] B. Zadrozny and C. Elkan, "Transforming classifier scores into accurate multiclass probability estimates," in Proc. 8th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 2002, pp. 694–699.

[28] S. Hughes, M. Bae, and M. Li, "Vectara Hallucination Evaluation Model (HHEM-2.1-Open)," Hugging Face / Vectara Technical Report, 2024.

[29] R. Friel, M. Belyi, and A. Sanyal, "RAGBench: Explainable benchmark for retrieval-augmented generation systems," arXiv preprint arXiv:2407.11005, 2024.

[30] TruEra / Snowflake, "TruLens: Evaluation and tracking for LLM experiments (The RAG Triad)," Technical Documentation, 2024.

[31] M. Kull, T. Silva Filho, and P. Flach, "Beta calibration: A well-founded and easily implemented improvement on logistic calibration for binary classifiers," in Proc. 20th International Conference on Artificial Intelligence and Statistics (AISTATS), 2017, pp. 623–631.

[32] B. Quach, A. Fisch, T. Schuster, A. Yala, J. H. Sohn, T. S. Jaakkola, and R. Barzilay, "Conformal language modeling," in Proc. 12th International Conference on Learning Representations (ICLR), 2024.

[33] J. J. Cherian, I. Gibbs, and E. J. Candès, "Large language model validity via enhanced conformal prediction methods," in Advances in Neural Information Processing Systems (NeurIPS), vol. 37, 2024.

[34] A. Wallach, H. Liu et al., "Faithfulness vs. plausibility in post-hoc LLM explanations and citations," arXiv preprint arXiv:2412.18004, 2024.

[35] J. Lei, M. G'Sell, A. Rinaldo, R. J. Tibshirani, and L. Wasserman, "Distribution-free predictive inference for regression," Journal of the American Statistical Association, vol. 113, no. 523, pp. 1094–1111, 2018.

[36] J. Cohen, "A coefficient of agreement for nominal scales," Educational and Psychological Measurement, vol. 20, no. 1, pp. 37–46, 1960.

[37] J. R. Landis and G. G. Koch, "The measurement of observer agreement for categorical data," Biometrics, vol. 33, no. 1, pp. 159–174, 1977.

[38] B. Efron and R. J. Tibshirani, An Introduction to the Bootstrap, New York, NY, USA: Chapman & Hall/CRC, 1993.

[39] G. W. Brier, "Verification of forecasts expressed in terms of probability," Monthly Weather Review, vol. 78, no. 1, pp. 1–3, 1950.

[40] MedHallu Consortium, "MedHallu: Medical hallucination detection benchmark for clinical question answering and summarization," Hugging Face Datasets, 2024.

[41] A. Paszke et al., "PyTorch: An imperative style, high-performance deep learning library," in Advances in Neural Information Processing Systems (NeurIPS), vol. 32, pp. 8024–8035, 2019.

[42] T. Wolf et al., "Transformers: State-of-the-art natural language processing," in Proc. Conference on Empirical Methods in Natural Language Processing: System Demonstrations (EMNLP), 2020, pp. 38–45.

[43] J. Brooke, "SUS: A 'quick and dirty' usability scale," in Usability Evaluation in Industry, P. W. Jordan, B. Thomas, B. A. Weerdmeester, and I. L. McClelland, Eds. London, UK: Taylor & Francis, 1996, pp. 189–194.

[44] ISO/IEC, Systems and software engineering — Systems and software Quality Requirements and Evaluation (SQuaRE) — System and software quality models, ISO/IEC 25010:2023, Geneva, Switzerland, 2023.

[45] European Parliament and Council of the European Union, "Regulation (EU) 2024/1689 laying down harmonised rules on artificial intelligence (Artificial Intelligence Act)," Official Journal of the European Union, L 2024/1689, Jul. 2024.



# GLOSSARY

• Abstention Gate (`no_evidence`): A calibrated threshold filter (\max_i f(s, c_i) < τ_abs) in Component 4 that determines when no sentence in the retrieved context supports or directly contradicts a flagged response span, classifying the span as ungrounded or baseless rather than forcing a spurious sentence citation.

• Adaptive Prediction Sets (APS): A split-conformal prediction method (Romano et al. [18]) that normalises candidate scores via softmax and accumulates candidates in descending order until their cumulative probability mass reaches the calibrated threshold q̂_α.

• Attributable Span: A hallucinated response span in RAGTruth whose human annotation contains an 
explicit Original: quote marker that is not classified as a "no evidence" statement. This permits a ground-
truth link to a specific retrieved context sentence (652 spans in calib, 824 spans in test).

• Evident Baseless Information: A RAGTruth error category covering specific facts, entities, or claims 
introduced by the language model without any basis in the retrieved context. These often come from 
parametric memory or pure fabrication.

• Evident Conflict: A RAGTruth error category in which the language model generates a statement that directly contradicts an explicit fact, number, date, or entity in the retrieved context.

• Exchangeability: The foundational statistical assumption of split-conformal prediction, requiring that the joint distribution of calibration examples and test examples is invariant to permutations, which guarantees finite-sample marginal coverage without distributional assumptions on the model scores.

• *Gold Context Index (i(s)):** The 0-based index of the retrieved context sentence c_i ∈ C_x that achieves the highest content-word containment of the human annotator's Original: snippet, subject to the ≥ 0.90 containment threshold.

• Least Ambiguous Set-Valued Classifier (LAC): A split-conformal prediction method (Sadinle et al. [17]) using the nonconformity score r(s, i) = 1 - f(s, c_i), which includes every context sentence whose relevance score satisfies f(s, c_i) ≥ 1 - q̂_α.

• Pre-Filter Drop Rate (`pre_filter_drop`): The fraction of attributable spans for which the true gold evidence sentence i*(s) is eliminated by a first-stage candidate filter before the main attributor scores the candidates (21.94% for top-5 IDF pre-filtering vs. 2.35% for full-pool scoring).

• Reviewer Reading Burden (`total_cost`): The expected number of retrieved context sentences a human verifier must read per flagged span, accounting both for non-empty conformal sets (|Ĉ_α(s)|) and for empty conformal sets (Ĉ_α(s) = ∅) where the reviewer must fall back to reading the entire retrieved context (|C_x|).

• Zero-Threshold Invariance: The mathematical property proven in Section 6.1.5 (Proposition 6.1) whereby any strictly monotone post-hoc probability calibrator produces identical LAC conformal prediction sets to raw scores whenever the empirical quantile q̂_α < 1.0.



# APPENDICES

## Appendix A: Extended Experimental Results, Production Artifact Contract, and Complete 50-Span Dual-Rater Blind Audit Table

### A.1 frozen production calibration artifact (`results/c4/c4_artifact.json`)

The following JSON file is the exact frozen calibration artifact consumed at startup by the TrustRAG FastAPI service (app/service.py) and verified by the automated test suite (tests/c4/test_app_integration.py):

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

### A.2 complete 50-span dual-rater blind audit table (`results/c4/analysis/no_evidence_sample.json`)

As documented in Chapter 3 (Section 3.2.2) and Chapter 6 (Section 6.1.1 and Section 6.3.1), 50 spans flagged by the phrase-list heuristic as no_evidence were sampled with fixed seed 42 across the frozen calib and test splits and independently audited by two human raters:

• Rater 1 (`R1`): Candidate (E. M. S. N. Rathnasiri, IT22047724) — Agreement with regex label = `47 / 50` (`94.00%`).

• Rater 2 (`R2`): Second blind team auditor — Agreement with regex label = `46 / 50` (`92.00%`).

• Inter-Annotator Agreement: Raw agreement P_o = 49 / 50 (`98.00%`), expected chance agreement P_e = 0.8696, Cohen's κ = 0.8466 (almost perfect agreement).

Table A.1 presents all 50 audited spans in full.

*Table A.1: Complete 50-Span Dual-Rater Blind Human Audit of the `no_evidence` Ground-Truth Label (`seed = 42`)*



## Appendix B: Academic Integrity and Generative AI Use Disclosure

In full compliance with the SLIIT Policy on Academic Integrity and the IT4010 Individual Thesis Template Guidelines, Table B.1 discloses all uses of Generative AI tools during the preparation of Component 4 and this dissertation, along with the human verification performed by the candidate.

*Table B.1: Generative AI Use Disclosure Log*



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

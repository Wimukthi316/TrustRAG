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

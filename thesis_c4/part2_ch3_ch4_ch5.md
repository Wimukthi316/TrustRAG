# 3 RESEARCH METHODOLOGY

## 3.1 Research design

Our experimental methodology follows a design-science approach structured across twelve analytical layers (Layers 1 to 12, followed by latency benchmarking and acceptance testing in Layers 13 to 15). To prevent data leakage and avoid training on noisy ground truth, we established five practical safeguards from day one.

Before running any model, we audited the human supervision in RAGTruth across Layers 1 and 1b. Inspecting calibration snippets and conducting an independent blind dual-rater audit on 50 sampled annotations verified our negative-evidence extraction at 94.0% agreement (Cohen's $\kappa = 0.8466$).

We then locked the containment matching threshold at $\theta = 0.50$ in Layer 2 prior to testing retrieval backends, while recording sensitivity curves across $\theta \in [0.30, 0.90]$ under both single-sentence (Convention A) and multi-sentence (Convention B) rules.

Critically, every fitted parameter (including the chosen scorer `lexical|sentence`, calibrators, conformal quantiles, and the abstention cutoff $\tau = 0.1973$) was derived strictly from the calibration split (`calib`) and evaluated exactly once on the held-out test split (`test`).

To keep comparisons honest, we evaluated models against simple task-specific baselines (Layers 3, 5, 7, 11), including random selection, first and longest sentence heuristics, and constant base-rate predictors.

Finally, regression tests enforce strict numerical parity ($\le 10^{-9}$) between offline evaluation scripts and the live `C4Attributor` service loaded by FastAPI.
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
| *Task: `qa`* | $696$ | $407$ | $0.5848$ | $111$ | $296$ | **$0.7273$** | $11.57$ / $10.0$ / $37.0$ |
| *Task: `data2text`* | $1,148$ | $948$ | $0.8258$ | $844$ | $103$ | $0.1086$ | $18.67$ / $17.0$ / $36.0$ |
| *Task: `summarization`* | $903$ | $480$ | $0.5316$ | $521$ | $143$ | $0.2154$ | $20.89$ / $19.0$ / $89.0$ |
| *Error: `Evident Conflict`* | $981$ | $956$ | $0.9745$ | $937$ | $19$ | $0.0199$ | $18.95$ / $17.0$ / $89.0$ |
| *Error: `Subtle Conflict`* | $214$ | $183$ | $0.8551$ | $174$ | $9$ | $0.0492$ | $19.42$ / $17.0$ / $62.0$ |
| *Error: `Evident Baseless Info`* | $1,089$ | $456$ | $0.4187$ | $198$ | $257$ | $0.5636$ | $18.71$ / $16.0$ / $81.0$ |
| *Error: `Subtle Baseless Info`* | $463$ | $240$ | $0.5184$ | $167$ | $73$ | $0.3042$ | $18.68$ / $16.0$ / $74.0$ |

*Table 3.1: Frozen RAGTruth Dataset Splits and Evidence Annotation Distribution (Layer 1 Audit).*

### 3.2.2 Structure of RAGTruth's `original:` metadata
During the Layer 1 audit, close examination of RAGTruth's `Original:` comments brought several structural quirks to light.

First, annotators closed their quotation blocks using five specific tags (`original:`, `generative:`, `generated:`, `aigc:`, and `aogc:`). Splitting on general colon characters breaks down whenever context sentences contain words like `Review:` or `Hours:`, so our parser restricts tag detection strictly to this closed set.

Second, structured data quotes routinely wrap across multiple lines before the next marker. Reading only the first line after `Original:` inadvertently discards evidence in 6.4% of comments (118 cases); our extractor captures lines continuously until the next tag is encountered.

Finally, 105 spans (3.8%) were annotated as `implicit_true` (facts that hold in the real world but are absent from the context), while 22 spans were flagged as `due_to_null` due to missing JSON attributes.
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
| **Layers 13 to 15** | `latency_benchmark.py`, `uat_validation.py`, `arch_diagrams.py`, `tables.py`, `figures.py` | Benchmark CPU/GPU latency and memory (`C4-15`), execute 10-case UAT and internal team verification timing study (`C4-16`), and render 10 figures + 16 tables. | `results/c4/analysis/latency_benchmark.json`, `uat_validation.json`, `results/c4/tables.md`, `paper/figures/c4_*.png` |

*Table 3.2: Twelve-Layer Experimental Protocol and Associated Artefact Outputs in Component 4.*

### 3.3.1 Matching protocol definitions
In Layer 2 (`src/c4_attribution/matching.py`), we score containment of quote $q$ in sentence $c_j$ as $\text{cont}(q, c_j) = |T(q) \cap T(c_j)| / |T(q)|$. We evaluate under two conventions. Under Convention A (Single Gold Sentence), we select $j^* = \arg\max_j \text{cont}(q, c_j)$, marking the span as matched if $\text{cont}(q, c_{j^*}) \ge \theta$ (with pre-fixed threshold $\theta = 0.50$); otherwise the span is marked unmatched. Under Convention B (Multiple Gold Sentences), any sentence meeting $\text{cont}(q, c_j) \ge \theta$ is counted as a valid gold match.

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

Analyses are implemented deterministically using Python and NumPy with fixed random seeds (`np.random.default_rng(42)`). Empirical coverage bounds and confidence intervals are derived from 1,000 bootstrap iterations. For multi-component pipeline fusion, gains must exceed Component 1's multi-seed variation ($\sigma_{\text{seed}} = 0.384$) and its 4.18-point bootstrap interval width to be considered statistically meaningful. To guard against Simpson's paradox, metrics are reported both overall and disaggregated across QA, Data-to-Text, and Summarization.

---

## 3.6 Ethics privacy security and responsible AI

In line with our approved Topic Assessment Form (Page 9), this investigation did not collect external personal data or recruit external human participants; all experiments operate on public open-access benchmarks (`RAGTruth`, `RAGBench`), with human validation restricted to internal team review.

To communicate uncertainty responsibly, Component 4 enforces two strict runtime rules:
Per-span text fields cannot contain terms like `'coverage'` or `'guarantee'`, since conformal coverage is a property of the overall procedure, not an individual span (Serving Rule 2).
When Component 2 detects out-of-distribution inputs (`guarantee_applies = False`), the interface continues rendering candidate sentences but cleanly withdraws the coverage claim (Serving Rule 3).

Additionally, because Component 4 executes completely in memory on local hardware, no user text or retrieved documents are ever transmitted to external APIs.
---

## 3.7 Reproducibility and research artefacts

All experiments, tables, and figures can be regenerated deterministically from git branch `feature/sakindu-c4`. Random seeds are pinned to 42, dataset split assignments are saved in `results/c1/modernbert-base/split_ids.json`, and all empirical outputs are stored in human-readable JSON files under `results/c4/`. The full experimental suite can be reproduced directly using the command line: `python -m src.c4_attribution.tables --write` to regenerate evaluation tables, `python -m src.c4_attribution.figures` to regenerate plots, and `pytest tests/test_c4_*.py` to execute the complete automated test suite.

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

When a request arrives at `POST /api/analyze`, the input passage is segmented into candidate sentences via regex `(?<=[.!?])\s+`. Component 1's `ModernBERT-base` detector flags suspicious character spans, while Component 2 applies Platt calibration and a conformal gate to produce a tri-state classification (`FLAG`, `ABSTAIN`, `PASS`).

Component 4 then scores each context sentence via within-context IDF containment. If the maximum similarity falls below $\tau = 0.1973$, it certifies the claim as ungrounded (`no_evidence`); otherwise, it builds a conformal prediction set $\mathcal{C}_\alpha(s)$ and supplies tie-breaking logic for uncertain spans. Finally, Component 3 generates hedged explanations based on the assigned risk tiers and identified evidence sentences.
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

Our architectural and engineering decisions follow international software engineering standards. ISO/IEC 25010:2023 (SQuaRE) governed our non-functional requirements, specifically sub-50ms CPU latency (`NFR-C4-02`), zero GPU memory overhead (`NFR-C4-03`), and offline/online numerical parity (`NFR-C4-01`). IEEE Std 1016-2009 guided our architectural representations across Figures 4.1, 4.2, and 4.3, while IEEE Std 829 / ISO/IEC/IEEE 29119 structured our five-tier test suite. Type contracts across all REST endpoints are validated through OpenAPI 3.1.0 and Pydantic v2.
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

In production deployment, heavy neural benchmarking runs once offline via `prefilter.py`, saving score files to `results/c4/scores/` and freezing calibration thresholds in `results/c4/c4_artifact.json`. At application startup, FastAPI loads this lightweight JSON contract. Because `C4Attributor` relies purely on standard Python libraries and NumPy, launching the service requires no external neural weights, consumes zero additional VRAM, and introduces only $1.312\text{ ms}$ median latency per span.

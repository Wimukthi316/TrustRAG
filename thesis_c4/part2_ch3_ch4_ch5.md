# 3 RESEARCH METHODOLOGY

## 3.1 Research design

This study adopts a **quantitative, empirical, and design-science research methodology** structured as a twelve-layer experimental protocol (`Layers 1` through `12`, extended with runtime latency, UAT, and architectural verification in `Layers 13–15`). In machine learning systems engineering, a frequent methodological failure is *post-hoc selection bias*—tuning thresholds on the test split, reporting pooled numbers that hide task-level regressions, or evaluating against noisy heuristic labels without auditing them first. To eliminate these threats to validity, the research design of Component 4 enforces five strict methodological invariants:

1. **Audit-Before-Modelling Rule (Layer 1 & Layer 1b):** Before any attribution model is evaluated, the human supervision source (`Original:` fields in RAGTruth's annotator comments) is audited on Component 1's frozen splits (`src/c4_attribution/evidence_audit.py`). The regular-expression heuristic used to separate *no-evidence denials* from *attributable quotes* (`src/c4_attribution/evidence_labels.py`) is gated on a seeded ($n = 50$, $\text{seed} = 42$) dual-rater blind human audit (`results/c4/analysis/no_evidence_sample.json`). Only because the human audit verified $94.00\%$ primary agreement ($47/50$, exceeding the pre-registered $0.90$ threshold) and $98.00\%$ inter-annotator agreement (Cohen's $\kappa = 0.8466$) is the no-evidence label permitted to be quoted as a ground-truth label in Layer 7.
2. **Pre-Fixed Reference Matching Threshold (Layer 2):** Because fewer than $51\%$ of human `Original:` quotes appear verbatim in the segmented context, the reference gold sentence $j_{\text{gold}}$ is defined as the context sentence maximising content-token containment above a fixed threshold $\theta = 0.50$ (`src/c4_attribution/matching.py`). The value $\theta = 0.50$ was locked prior to running any attribution scorer, and a full sensitivity sweep across $\theta \in \{0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90\}$ is reported alongside both single-sentence (`Convention A`) and multi-sentence (`Convention B`) matching conventions (`Table C4-3`).
3. **Strict Calibration-Split Fitting, Single Test-Split Evaluation (Layers 4–11):** Every fitted quantity in Component 4—the selected backend and hypothesis convention (`lexical|sentence`), post-hoc score calibrators (`temperature`, `platt`, `isotonic`), split conformal LAC quantiles $\hat{q}_{1-\alpha}$, CRC thresholds $\hat{\lambda}_{\text{CRC}}$, and the no-evidence abstention threshold $\tau = 0.1973$—is fitted **exclusively on the frozen calibration split (`calib`)** and applied **once** to the held-out test split (`test`). Automated unit tests (`tests/test_c4_abstention.py`, `tests/test_c4_conformal.py`, `tests/test_c4_fusion.py`) assert that calibration fitting functions never receive test-split records.
4. **Mandatory Task-Stratified Floor Reporting (Layers 3, 5, 7, 11):** Every empirical table prints non-trivial baseline floors alongside the proposed method, broken down by task (`qa`, `data2text`, `summarization`):
   * *Attribution floors (`Layer 3`):* `first` sentence, `longest` sentence, `uniform random` ($\text{seed} = 42$), and `always abstain`.
   * *Calibration floor (`Layer 5`):* Constant base-rate predictor ($\hat{p} \equiv \bar{y}_{\text{cal}}$), which achieves near-zero ECE without reading its input.
   * *Abstention floors (`Layer 7`):* `always_abstain` (whose F1 is a deterministic function of each task's no-evidence base rate), `never_abstain`, and `nearest_sentence_no_threshold`.
   * *Fusion floors (`Layer 11`):* `trivial_always_positive` ($0.5177$), `HHEM adapted` ($0.7012$), `C1 alone` ($0.7623$), and `LettuceDetect-large` ($0.7918$), evaluated against Component 1's multi-seed standard deviation ($0.384$ points) and $1,000$-iteration bootstrap $95\%$ confidence interval width ($4.18$ points).
5. **Offline-to-Online Zero-Drift Contract Parity (Layer 12):** The shipped online attributor (`C4Attributor` in `src/c4_attribution/serving.py`) executes the exact same scoring function (`lexical.idf_overlap_scores`) and reads its thresholds from a human-readable JSON artefact (`results/c4/c4_artifact.json`), verified by `tests/test_c4_serving.py` to match offline evaluation within $10^{-9}$.

---

## 3.2 Data participants or study materials

### 3.2.1 Frozen RAGTruth benchmark splits and provenance
All offline experiments use the official **RAGTruth** corpus (Niu et al., ACL 2024 [4]), stored locally under `data/raw/response.jsonl` ($17,790$ responses) and `data/raw/source_info.jsonl`. To guarantee zero data leakage and $100\%$ cross-component compatibility with Component 1 (`ModernBERT-base`), Component 2 (Platt calibration and split conformal prediction), and Component 3 (explanation generation), Component 4 loads the exact frozen split identifiers from `results/c1/modernbert-base/split_ids.json` via `src/c4_attribution/dataset.py` and constructs contexts using Component 1's exact `build_context_and_question` rule (`src/c1_detector/build_examples.py`).

Two distinct evaluation universes exist on these frozen splits and are never conflated:
1. **Gold-Span Attribution & Abstention Universe (`calib` + `test` across all responses):** Used in Layers 1 through 10 to evaluate evidence supervision, reference matching, backend attribution accuracy, pre-filter ceilings, score calibration, split conformal evidence sets, no-evidence abstention, and cross-task transfer. Across `calib` and `test`, there are **$2,747$ human-annotated gold hallucination spans** ($1,214$ in `calib`, $1,533$ in `test`).
2. **Predicted-Span Multi-Component Fusion Universe (`calib` = $297$ responses, `test` = $1,509$ responses):** Used in Layer 11 (`src/c4_attribution/fusion.py`) to evaluate end-to-end `C1` $\times$ `C2` $\times$ `C4` pipeline fusion on the predicted spans emitted by Component 1 (`results/c1/calib/probabilities.jsonl` and `results/c1/test/probabilities.jsonl`, containing $2,390$ predicted test spans across $1,509$ test responses).

Table 3.1 summarises the exact supervision distribution across `calib`, `test`, tasks, and error types measured by Layer 1 (`results/c4/analysis/evidence_audit.json`).

| Slice / Stratum | Annotated Gold Spans ($n$) | Spans with `Original:` Marker | Share with `Original:` | Attributable Spans (Quote Exists) | No-Evidence Spans (Denial) | No-Evidence Share of `Original:` | Candidate Context Pool Mean / Median / Max (Sentences) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
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

### 3.2.2 Anatomy of RAGTruth's `Original:` annotator metadata and confounds
Inspecting the raw annotator metadata (`meta`) across the $2,747$ gold spans in Layer 1 (`src/c4_attribution/evidence_labels.py`) uncovered four structural properties that directly informed our preprocessing design:
1. **Closed Marker Vocabulary (`MARKERS`):** Annotators delimit source quotes and model generations using five case-insensitive markers: `original:`, `generative:`, `generated:`, `aigc:`, and `aogc:` (occurring $1,092$, $554$, $92$, $0$, and $97$ times as closing markers across the $1,835$ comments). Using a generic regex such as `\n\s*[A-Za-z]+\s*:` to detect the end of an `Original:` block catastrophically truncates real quotes in Data-to-Text and Summarization because source documents frequently contain embedded labels like `Review:`, `Monday:`, `Friday:`, `Sunday:`, and `Weekends:`. Component 4 therefore restricts marker splitting strictly to the closed five-token tuple `("original", "generative", "generated", "aigc", "aogc")`.
2. **Until-Next-Marker vs. First-Line-Only Extraction:** Component 3's `labels_audit.original_snippet` extracts only the first line after `Original:`, whereas multi-line quotes (such as restaurant opening hours across days of the week) wrap across two to four lines before `Generated:`. In Layer 1 (`convention_disagreement`), we measured both rules across all $1,835$ comments: they agree on $1,717$ comments and **disagree on $118$ comments ($6.43\%$)**. Component 4 uses *until-next-marker* extraction so multi-line quotes are preserved intact, and reports the $6.43\%$ divergence transparently (`results/c4/analysis/evidence_audit.json`).
3. **Confounds (`implicit_true` and `due_to_null`):** Exactly $105$ spans ($3.82\%$ of all gold spans, and $22.68\%$ of `Subtle Baseless Info` spans) are marked `implicit_true = True`—statements that are factually true in the real world (e.g., *"symptoms of Raynaud's disease appear due to cold temperature"*) but absent from the retrieved passages. Another $22$ spans ($0.80\%$) are marked `due_to_null = True` (errors caused by `null` fields in JSON tables). Tracking these flags explains why NLI cross-encoders trained on general world knowledge can assign high entailment to `Subtle Baseless Info` spans that are actually ungrounded in the passage.

---

## 3.3 Experimental or study procedure

The experimental procedure of Component 4 is organised into a deterministic twelve-layer pipeline (plus three technical validation layers for Chapter 4 and Chapter 6), summarised in Table 3.2.

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

### 3.3.1 Attribution matching protocol (`Convention A` and `Convention B`)
In Layer 2 (`src/c4_attribution/matching.py`), given an annotated span with `Original:` quote $q$ and segmented context sentences $D = (c_1, \dots, c_m)$, we compute the content-token containment of $q$ in each sentence $c_j$:
$$\text{cont}(q, c_j) = \frac{|T(q) \cap T(c_j)|}{|T(q)|}$$
Two formal ground-truth conventions are evaluated side by side:
* **Convention A (Single Canonical Gold Sentence):** Let $j^* = \arg\max_{j \in \{1,\dots,m\}} \text{cont}(q, c_j)$ (breaking ties on the lowest sentence index $j$). If $\text{cont}(q, c_{j^*}) \ge \theta$ (with $\theta = 0.50$), then $j_{\text{gold}} = j^*$; otherwise the span is marked `unmatched` and excluded from top-1 accuracy denominators (with the exact unmatched count printed in `Table C4-3`).
* **Convention B (Multi-Sentence Gold Set):** All context sentences satisfying $G_\theta(q) = \{j \mid \text{cont}(q, c_j) \ge \theta\}$ form the gold set, and a prediction $\hat{j}$ is credited as correct if $\hat{j} \in G_\theta(q)$.

### 3.3.2 Multi-backend attribution and hypothesis-formation ablations
In Layer 4 (`src/c4_attribution/prefilter.py`), we evaluate a $4 \times 2$ factorial grid across four scoring backends and two hypothesis-formation rules:
* **Four Scoring Backends:**
  1. `lexical`: Parameter-free IDF content-token containment (`lexical.idf_overlap_scores`), scored exhaustively over all $m$ context sentences.
  2. `embed`: Dense bi-encoder cosine similarity using `BAAI/bge-small-en-v1.5` ($33.4\text{M}$ parameters), scored over all $m$ context sentences.
  3. `nli_entail`: Cross-encoder entailment probability $P(\text{entailment})$ from `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` ($184.4\text{M}$ parameters), evaluated on the top $k=5$ pre-filtered sentences per span.
  4. `hhem_support`: Support score from `vectara/hallucination_evaluation_model` ($184.0\text{M}$ parameters), evaluated on the top $k=5$ pre-filtered sentences per span.
* **Two Hypothesis-Formation Rules:**
  1. `span` hypothesis: The raw flagged substring $s_i$ (which may be a 2-word fragment such as `"August 8th"` or `"its third week"`) is passed directly as the query/hypothesis.
  2. `sentence` hypothesis: The full generated answer sentence enclosing $s_i$ is passed as the query/hypothesis, restoring grammatical subject-verb-object context around short fragments.

---

## 3.4 Measures and evaluation metrics

To evaluate every dimension of Component 4 rigorously, we define seven formal families of metrics:

1. **Top-1 Attribution Accuracy, Mean Reciprocal Rank (MRR), and Recall@k:**
   Over $N_{\text{attr}}$ matched attributable test spans with gold sentence index $j_{\text{gold}}^{(r)}$ and ranked candidate indices $(\pi_r(1), \dots, \pi_r(m_r))$ sorted in descending order of score (ties broken by smaller sentence index):
   $$\text{Top-1} = \frac{1}{N_{\text{attr}}} \sum_{r=1}^{N_{\text{attr}}} \mathbb{I}\!\left[\pi_r(1) = j_{\text{gold}}^{(r)}\right], \quad \text{MRR} = \frac{1}{N_{\text{attr}}} \sum_{r=1}^{N_{\text{attr}}} \frac{1}{\text{rank}_r(j_{\text{gold}}^{(r)})}$$
   $$\text{Recall@}k = \frac{1}{N_{\text{attr}}} \sum_{r=1}^{N_{\text{attr}}} \mathbb{I}\!\left[\text{rank}_r(j_{\text{gold}}^{(r)}) \le k\right], \quad \text{ErrorFloor@}k = 1 - \text{Recall@}k$$
   In addition to unconditional Top-1 across all $N_{\text{attr}} = 582$ test spans, Layer 4 reports **Conditional Top-1 Given Pre-Filter Survival**:
   $$\text{Top-1}_{\mid \text{survived}} = \frac{\sum_{r=1}^{N_{\text{attr}}} \mathbb{I}[\pi_r(1) = j_{\text{gold}}^{(r)} \land j_{\text{gold}}^{(r)} \in \mathcal{P}_k(s_r)]}{\sum_{r=1}^{N_{\text{attr}}} \mathbb{I}[j_{\text{gold}}^{(r)} \in \mathcal{P}_k(s_r)]}$$
   which isolates the cross-encoder's intrinsic ranking ability from the pre-filter's recall ceiling.

2. **Conformal Empirical Coverage, Finite-Sample Band, and Clopper–Pearson / Bootstrap CIs:**
   At miscoverage level $\alpha$, empirical coverage on the test split is:
   $$\widehat{\text{Cov}}(\alpha) = \frac{1}{N_{\text{test}}} \sum_{r=1}^{N_{\text{test}}} \mathbb{I}\!\left[j_{\text{gold}}^{(r)} \in \mathcal{C}_\alpha(s_r)\right]$$
   Because finite calibration and test splits exhibit binomial sampling variability even under exact exchangeability, Layer 6 (`evidence_conformal.py`) computes the theoretical $3\sigma$ finite-sample standard-error band accounting for both $n_{\text{cal}}$ and $n_{\text{test}}$:
   $$\text{Band}_{3\sigma}(\alpha) = 3 \sqrt{\frac{\alpha(1 - \alpha)}{n_{\text{test}}} + \frac{\alpha(1 - \alpha)}{n_{\text{cal}}}}$$
   Coverage is judged `inside_band = True` if $\widehat{\text{Cov}}(\alpha) \ge (1 - \alpha) - \text{Band}_{3\sigma}(\alpha)$. We simultaneously report $95\%$ bootstrap confidence intervals $[\text{CI}_{\text{low}}, \text{CI}_{\text{high}}]$ (`1,000` resamples, $\text{seed} = 42$), mean prediction set size $\overline{|\mathcal{C}_\alpha|} = \frac{1}{N_{\text{test}}}\sum_r |\mathcal{C}_\alpha(s_r)|$, singleton rate $\frac{1}{N_{\text{test}}}\sum_r \mathbb{I}[|\mathcal{C}_\alpha(s_r)| = 1]$, and empty-set rate $\frac{1}{N_{\text{test}}}\sum_r \mathbb{I}[|\mathcal{C}_\alpha(s_r)| = 0]$.

3. **Reviewer Reading Burden Reduction and Response-Level Compression:**
   In Layer 10 (`src/c4_attribution/review_cost.py`), for each span $r$ with context pool size $m_r$ and conformal set size $|\mathcal{C}_\alpha(s_r)|$, the **per-span reading reduction** is:
   $$\text{Reduction}_{\text{span}}(\alpha) = \frac{1}{N_{\text{test}}} \sum_{r=1}^{N_{\text{test}}} \left(1 - \frac{|\mathcal{C}_\alpha(s_r)|}{m_r}\right)$$
   At the response level, because multiple spans in the same response may cite overlapping context sentences, the unique sentences a reviewer must read for response $d$ is $U_\alpha(d) = \left|\bigcup_{s_i \in d} \mathcal{C}_\alpha(s_i)\right|$, yielding the **response-level compression ratio** $\text{Compression}_{\text{resp}}(\alpha) = \frac{1}{|D_{\text{test}}|}\sum_{d} \frac{U_\alpha(d)}{m_d}$.

4. **No-Evidence Abstention Precision, Recall, and F1:**
   Let $y_r^{\text{no\_ev}} \in \{0, 1\}$ denote whether span $r$ is a ground-truth no-evidence span (`label.no_evidence == True`), and let $\hat{y}_r^{\text{no\_ev}} = \mathbb{I}[\max_j s_{r,j} < \tau]$ denote Component 4's abstention prediction. Standard binary Precision, Recall, and F1 are computed on the test split and compared against the task-specific `always_abstain` floor:
   $$\text{F1}_{\text{always\_abstain}}(\text{task}) = \frac{2 \cdot b_{\text{task}}}{1 + b_{\text{task}}}, \quad \text{where } b_{\text{task}} = \mathbb{P}(y^{\text{no\_ev}} = 1 \mid \text{task})$$

5. **Inter-Annotator Agreement (Cohen's Kappa $\kappa$):**
   For the dual-rater blind audit of $N = 50$ sampled no-evidence snippets (`src/c4_attribution/evidence_audit.py`), let $P_o = (a + d)/N$ be the observed agreement rate and $P_e = \left(\frac{a+b}{N}\cdot\frac{a+c}{N}\right) + \left(\frac{c+d}{N}\cdot\frac{b+d}{N}\right)$ be the expected chance agreement. Cohen's Kappa [38] is:
   $$\kappa = \frac{P_o - P_e}{1 - P_e}$$

6. **Response-Level Hallucination Detection Example F1 (Layer 11 Fusion):**
   Following RAGTruth [4] and Component 1 (`src/c1_detector/evaluate_c1.py`), a response $d$ is gold-positive ($Y_d = 1$) if it contains at least one gold hallucinated span, and predicted-positive ($\hat{Y}_d = 1$) under fusion rule $\mathcal{R}$ if at least one predicted span in $d$ satisfies rule $\mathcal{R}$.

---

## 3.5 Data analysis

All statistical analyses in Component 4 are implemented in pure Python and NumPy with deterministic pseudo-random number generators (`np.random.default_rng(42)`):
* **Non-Parametric Bootstrap Confidence Intervals:** Coverage confidence intervals in `Table C4-8` are computed via $B = 1,000$ paired bootstrap resamples of the test split (`seed = 42`), reporting the $2.5\text{th}$ and $97.5\text{th}$ percentiles.
* **Noise-Floor Gating for Pipeline Fusion:** In Layer 11 (`src/c4_attribution/fusion.py`), any response-level F1 difference is evaluated against Component 1's multi-seed standard deviation ($\sigma_{\text{seed}} = 0.384$ F1 points across seeds $42, 7, 13$) and $95\%$ bootstrap interval width ($4.18$ F1 points). A fusion rule is only claimed to provide a statistically distinguishable improvement if its isolated gain exceeds the $1.0$-point noise floor (and preferably clears the $4.18$-point bootstrap width, as `c4_tie_break` does with $+5.96$ points over `c1_c2_flag_only`).
* **Stratified Subgroup Analysis:** Because Simpsons's paradox can mask task-specific failures in pooled metrics, every primary analysis (`Tables C4-2, C4-4, C4-9, C4-11, C4-12`) disaggregates results across `qa`, `data2text`, and `summarization`.

---

## 3.6 Ethics privacy security and responsible AI

1. **Compliance with Signed Topic Assessment Form (TAF Page 9):**
   In strict accordance with Section 6 (Page 9) of our signed SLIIT Topic Assessment Form, **this research does not recruit external human participants, collect personal identifiable information (PII), or conduct external crowd-sourced surveys**. All benchmark datasets (`RAGTruth` and `RAGBench`) are publicly released, de-identified academic corpora governed by open research licences (MIT / Apache-2.0). The human validation reported in Layer 1b and Section 6.3 consists strictly of the internal quality audit specified in the TAF, in which team members independently inspected 50 sampled model/annotator outputs and timed span verification effort on the local interface.
2. **Honest Uncertainty Communication (Serving Rules 2 and 3):**
   A major responsible-AI risk in decision-support software is *overclaiming statistical guarantees*. If an interface prints `"90% coverage guarantee"` next to an individual span, or continues claiming a conformal guarantee when the input comes from an out-of-distribution clinical or legal domain, operators may place unwarranted trust in an uncalibrated output. Component 4 enforces two hard code-level guardrails (`src/c4_attribution/serving.py`):
   * **Serving Rule 2:** Per-span evidence strings are checked by `serving.evidence_is_safe(text)` and may never contain `"coverage"`, `"guarantee"`, or `"%"`—because split conformal coverage is a marginal property of the procedure, not a probability attached to a single span.
   * **Serving Rule 3:** When Component 2's runtime exchangeability gate detects distribution shift (`guarantee_applies = False`), Component 4 still renders the candidate evidence sentence to aid the human reviewer, but the UI and API explicitly withdraw the coverage guarantee.
3. **Zero Third-Party Data Exfiltration:**
   Unlike Ragas [5] or cloud LLM-as-a-Judge tools that transmit sensitive retrieved documents to external commercial APIs (`OpenAI` / `Anthropic`), Component 4's served attribution pipeline runs $100\%$ locally in memory with zero external network calls.

---

## 3.7 Reproducibility and research artefacts

Every number, table, and figure in this thesis is deterministically reproducible from the committed source code and JSON artefacts on branch `feature/sakindu-c4`:
* **Fixed Random Seeds:** All sampling, random baseline floors, and bootstrap confidence intervals use `SEED = 42` (`np.random.default_rng(42)`).
* **Frozen Split Provenance:** Calibration and test split IDs are pinned in `results/c1/modernbert-base/split_ids.json`.
* **Human-Readable JSON Artefacts:** No binary Python pickles (`.pkl`) are used anywhere in Component 4. All twelve layers write structured, schema-versioned JSON files under `results/c4/` (`results/c4/c4_artifact.json`, `results/c4/analysis/*.json`, `results/c4/lexical/*.json`, `results/c4/scores/*.json`, `results/c4/calibration/*.json`, `results/c4/conformal/*.json`, `results/c4/transfer/*.json`, `results/c4/fusion/*.json`).
* **Single-Command Regeneration:**
  * Regenerate all 16 markdown tables (`C4-1` to `C4-16`): `python -m src.c4_attribution.tables --write`
  * Regenerate all 7 empirical data plots: `python -m src.c4_attribution.figures`
  * Regenerate all 3 architecture diagrams: `python -m src.c4_attribution.arch_diagrams`
  * Run the 147 automated C4 regression tests: `pytest tests/test_c4_*.py`

---

# 4 PRODUCT AND SOLUTION DESIGN

## 4.1 Requirements

The engineering design of Component 4 is governed by eight Functional Requirements (**FR-C4-01** to **FR-C4-08**) and four Non-Functional Requirements (**NFR-C4-01** to **NFR-C4-04**), specified in Table 4.1.

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

Figure 4.1 illustrates the overall four-stage **TrustRAG** system architecture and highlights how Component 4 integrates with Component 1 (Detection), Component 2 (Calibration, Conformal Prediction, and Runtime Exchangeability Gate), Component 3 (Explanation), and the shared FastAPI/React presentation layer.

![Figure 4.1: TrustRAG Four-Stage Calibrated Architecture (Component 4 Highlighted)](file:///S:/Research%20New%20Project/TrustRAG/paper/figures/c4_arch_system.png)
*Figure 4.1: TrustRAG Four-Stage Calibrated Architecture (Component 4 Highlighted).*

When a client submits a `POST /api/analyze` request containing `(question, context, answer, alpha)`:
1. **Context Segmentation (`sentences.split_sentences`):** The backend splits `context` using the frozen regex `(?<=[.!?])\s+` into `AnalysisResult.context_sentences` ($c_1, \dots, c_m$).
2. **Stage 1 — Component 1 (`ModernBERT-base` Detector):** Scores answer tokens and extracts contiguous candidate spans with character offsets `(start, end)` and raw `span_score`.
3. **Stage 2 — Component 2 (Platt Calibrator + Split Conformal + Exchangeability Gate):** Transforms `span_score` into `calibrated_score`, assigns `ConformalDecision` (`FLAG`, `ABSTAIN`, or `PASS`) at level `alpha`, and runs `DistributionCheck` over input features to set `AnalysisResult.guarantee_applies`.
4. **Stage 4 — Component 4 (`C4Attributor` Evidence Grounding & Abstention):** For each non-PASS span, Component 4 computes IDF content-token containment scores across `context_sentences`, applies the calibrated abstention threshold ($\tau = 0.1973$), constructs the LAC conformal evidence set $\mathcal{C}_\alpha(s_i)$, populates `Span.evidence_sentence`, `Span.evidence_index`, and `Span.entailment_score`, sets `AnalysisResult.attribution_checked = True`, and supplies the `no_evidence` signal to resolve C2 `ABSTAIN` spans (`c4_tie_break`).
5. **Stage 3 — Component 3 (Calibration-Consistent Explanation):** Consumes C2's calibrated risk tier and C4's `evidence_sentence` (or abstention status) to render a natural-language explanation with risk-monotonic hedging.

---

## 4.3 Design alternatives and decisions

During the design of Component 4, four major architectural trade-offs were evaluated empirically rather than decided by assumption. Table 4.2 summarises the alternatives considered, the empirical evidence gathered on the calibration split, and the rationale for the final design decision.

| Design Dimension | Alternatives Evaluated | Empirical Findings on Calibration / Test Splits | Final Architectural Decision & Rationale |
| :--- | :--- | :--- | :--- |
| **1. Online vs. Offline Attribution Scorer** | (A) `lexical` (IDF content-token containment)<br>(B) `embed` (`bge-small-en-v1.5`)<br>(C) `hhem_support` (`Vectara HHEM`, $k=5$)<br>(D) `nli_entail` (`DeBERTa-v3-base`, $k=5$) | Unconditional Top-1 (`Table C4-5`):<br>• `lexical`: **$0.8952$** ($1.31\text{ ms}$, $0\text{ MB}$)<br>• `hhem_support`: $0.6959$ ($24.6\text{ ms}$, $440\text{ MB}$)<br>• `nli_entail`: $0.6392$ ($28.5\text{ ms}$, $738\text{ MB}$)<br>• `embed`: $0.3007$ ($8.42\text{ ms}$, $133\text{ MB}$) | **Offline 4-backend benchmark + Online `lexical` serving (`c4_artifact.json`).** `lexical` achieves the highest accuracy on both `calib` ($0.9095$) and `test` ($0.8952$) while adding $0\text{ MB}$ RAM/VRAM on a 6 GB laptop where `ModernBERT` is already loaded. |
| **2. Candidate Pre-Filter Strategy (Layer 4b)** | (A) Dense embedding top-$k$ pre-filter (`bge-small`)<br>(B) IDF lexical top-$k$ pre-filter<br>(C) Exhaustive full-pool lexical scoring | At $k=5$ (`Table C4-6`):<br>• Dense pre-filter drops **$21.90\%$** of gold sentences (`Recall@5` $= 0.7810$).<br>• IDF lexical drops only **$2.41\%$** (`Recall@5` $= 0.9759$). | **Report the $21.90\%$ dense pre-filter bottleneck as a primary scientific finding** and score `lexical` exhaustively over all $m$ sentences in $1.31\text{ ms}$. |
| **3. Query Hypothesis Formation (Layer 4)** | (A) `span`: raw flagged span substring $s_i$<br>(B) `sentence`: enclosing answer sentence $h(s_i)$ | On `calib` (`Table C4-5`):<br>• `lexical\|sentence`: **$0.9095$** (Conv A), **$0.9629$** (Conv B)<br>• `lexical\|span`: $0.8886$ (Conv A), $0.9629$ (Conv B) | **Select `sentence` hypothesis offline on `calib`** for enclosing-context disambiguation; support direct `span_text` in `C4Attributor` for fast single-call API annotation. |
| **4. Conformal Prediction Formalism (Layer 6)** | (A) Least Ambiguous Set-Valued Classifiers (LAC) [13]<br>(B) Conformal Risk Control (CRC) [14] | At $\alpha = 0.10$ (`Table C4-10`):<br>• LAC: coverage $= 0.9003$, mean set $= 4.61$<br>• CRC: coverage $= 0.9003$, mean set $= 4.61$<br>• $\Delta = 0.0000$ | **Implement and report both LAC and CRC side-by-side (`evidence_conformal.py`)**, shipping LAC quantiles $\hat{q}_{1-\alpha}$ in `c4_artifact.json` due to their closed-form $O(m)$ evaluation. |

*Table 4.2: Architectural Trade-Off Matrix Across Attribution Scorer Backends and Conformal Formalisms.*

---

## 4.4 Detailed design

### 4.4.1 Component 4 internal pipeline and data flow
Figure 4.2 details the internal execution pipeline of Component 4 across its attribution, abstention, conformal prediction set, and multi-component tie-break stages.

![Figure 4.2: Component 4 Internal Attribution, Conformal Prediction Set, and Abstention Pipeline](file:///S:/Research%20New%20Project/TrustRAG/paper/figures/c4_arch_pipeline.png)
*Figure 4.2: Component 4 Internal Attribution, Conformal Prediction Set, and Abstention Pipeline.*

### 4.4.2 Frozen Pydantic data contract and serving artefact schema
Figure 4.3 presents the class and schema diagram connecting `AnalyzeRequest`, `AnalysisResult`, and `Span` in `src/common/schema.py` to Component 4's configuration artefact `results/c4/c4_artifact.json`.

![Figure 4.3: Frozen Pydantic Data Contract and C4 Serving Artefact Schema](file:///S:/Research%20New%20Project/TrustRAG/paper/figures/c4_arch_schema.png)
*Figure 4.3: Frozen Pydantic Data Contract (`src/common/schema.py`) and C4 Serving Artefact Schema.*

The production artefact `results/c4/c4_artifact.json` generated by Layer 12 contains the exact calibrated operating points fitted on the calibration split:
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
Algorithm 1 formalises the online execution of `C4Attributor.annotate` (`src/c4_attribution/serving.py`), combining IDF content-token containment scoring, calibrated no-evidence abstention, and LAC split conformal prediction set construction.

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

Algorithm 2 formalises the `c4_tie_break` multi-component fusion rule (`src/c4_attribution/fusion.py`) that resolves Component 2's conformal abstention band using Component 4's no-evidence signal.

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

The design and documentation of Component 4 adhere to four international software and AI engineering standards:
1. **ISO/IEC 25010:2023 (Systems and Software Quality Requirements and Evaluation — SQuaRE):** Guides our Non-Functional Requirements across *Time Behaviour* (`NFR-C4-02`, $1.312\text{ ms}$ p50 latency), *Resource Utilization* (`NFR-C4-03`, $0\text{ MB}$ model memory overhead), *Functional Correctness* (`NFR-C4-01`, $\le 10^{-9}$ offline/online parity), and *Maintainability* (modular 15-file package structure).
2. **IEEE Std 1016-2009 (Standard for Information Technology—Systems Design—Software Design Descriptions):** Governs the architectural decomposition into system context (**Figure 4.1**), component pipeline (**Figure 4.2**), and data contract schema (**Figure 4.3**).
3. **IEEE Std 829 / ISO/IEC/IEEE 29119 (Software Testing Standards):** Structures our five-level verification hierarchy (Unit, Contract, Parity, Integration, and UAT) detailed in Chapter 5.
4. **OpenAPI 3.1.0 & Pydantic v2 Schema Validation:** All REST payloads at `POST /api/analyze` and `GET /api/metrics` are strictly validated via Pydantic v2 models (`src/common/schema.py`) with explicit bounds (`ge=0.0, le=1.0`), ensuring type safety between the Python FastAPI backend and the TypeScript/React frontend.

---

# 5 IMPLEMENTATION INTEGRATION AND TESTING

## 5.1 Development environment and technology stack

Component 4 was developed and verified in a cross-platform Python/TypeScript environment designed to run offline batch evaluations on GPU hardware (NVIDIA Tesla T4 / RTX workstation) while serving live interactive requests on a resource-constrained laptop CPU. Table 5.1 lists the pinned technology stack.

| Layer / Role | Technology / Library | Version | Purpose in Component 4 |
| :--- | :--- | :--- | :--- |
| **Core Language** | Python | `3.10` / `3.11` | Primary implementation language for `src/c4_attribution/` and `backend/`. |
| **Numerical Computing** | NumPy | `1.26.4` | Vectorised quantile estimation, ECE binning,LAC/CRC conformal sets, and bootstrap CIs. |
| **Data Contracts** | Pydantic | `2.7.x` | Frozen schema validation (`Span`, `AnalysisResult`, `AnalyzeRequest` in `src/common/schema.py`). |
| **Offline Neural Scorers** | PyTorch + Hugging Face `transformers` | `2.2+` / `4.41+` | Offline benchmarking of `BAAI/bge-small-en-v1.5`, `DeBERTa-v3-base-mnli-fever-anli`, and `HHEM`. |
| **Backend API** | FastAPI + Uvicorn | `0.111+` | Asynchronous REST endpoints (`POST /api/analyze`, `GET /api/metrics`, `GET /api/figures/{name}`). |
| **Frontend UI** | React 18 + TypeScript + Vite + Tailwind | `5.x` | Interactive span highlighting, context sentence scroll-and-highlight, and conformal set display. |
| **Visualisation** | Matplotlib | `3.8.x` | Publication-grade vector `.pdf` and 300-DPI `.png` generation (`figures.py`, `arch_diagrams.py`). |
| **Automated Testing** | Pytest | `8.x` | Execution of $147$ C4 tests and $861$ total repository regression tests. |

*Table 5.1: Development Environment, Software Dependencies, and Pinning Strategy.*

---

## 5.2 Component implementation

The complete source code of Component 4 resides in `src/c4_attribution/` and comprises $18$ modular Python files totalling over $3,800$ lines of clean, documented, type-annotated code. Table 5.2 provides the complete module inventory.

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

Component 4 integrates with the rest of the TrustRAG platform through three tightly tested boundaries:

1. **Upstream Integration with Component 1 (`ModernBERT-base`) and Component 2 (Platt + Split Conformal):**
   * Offline, `src/c4_attribution/dataset.py` reads `results/c1/modernbert-base/split_ids.json` and imports `src.c1_detector.build_examples.build_context_and_question` so that every context string segmented by C4 is character-for-character identical to the context string seen by C1 and C2.
   * In Layer 11 (`src/c4_attribution/fusion.py`), C4 imports `calibrator_from_dict` (`src.c2_calibration.calibration`), `SplitConformal` (`src.c2_calibration.conformal`), and `span_units` (`src.c2_calibration.run_c2`) to rebuild C2's per-span decisions from `results/c2/c1/c2_artifact.json`. Before running fusion, `rebuild_c2_decisions` verifies that the rebuilt coverage matches C2's recorded test coverage ($0.8807531380753139$) to within $10^{-9}$ (`matches_recorded == True`).
2. **Peer Integration with Component 3 (Explanation Generation):**
   * Both C3 (`src/c3_explanation/grounding.py`) and C4 (`src/c4_attribution/sentences.py`) use the identical regex `(?<=[.!?])\s+` as `backend/app/services/detector.py`. Automated parity tests (`tests/test_c4_sentences.py` and `tests/test_c4_lexical.py`) assert that C4's splitter and unthresholded IDF overlap scorer agree with C3's functions on every input, while C4 extends the pipeline with calibrated no-evidence abstention ($\tau = 0.1973$) and LAC conformal evidence sets.
3. **Live FastAPI Backend and React Frontend Integration:**
   * In `backend/app/services/detector.py`, `C4Attributor.load("results/c4/c4_artifact.json")` is initialised at startup at $0\text{ MB}$ additional model memory. During `POST /api/analyze`, each detected span is passed through `C4Attributor.annotate(span.text, context_sentences, alpha=request.alpha)`, populating `span.evidence_sentence`, `span.evidence_index`, and `span.entailment_score` and setting `result.attribution_checked = True`.
   * In `backend/app/services/metrics.py`, all seven empirical C4 figure names (`c4_attribution_accuracy`, `c4_score_calibration`, `c4_conformal_coverage`, `c4_conformal_set_size`, `c4_no_evidence_pr`, `c4_transfer_matrix`, `c4_review_cost_curve`) are registered in `FIGURE_ORDER` and served via `GET /api/figures/{name}`.

---

## 5.4 Verification and software testing

To guarantee software reliability and prevent silent regressions across the four components, Component 4 implements a **five-level verification strategy** comprising **$147$ dedicated pytest tests** across $13$ test modules (`tests/test_c4_*.py`), within the group's **$861$-test repository suite**. All $147$ C4 tests pass ($0$ failures). Table 5.3 details the test suite across all five verification levels.

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

Component 4 is designed for zero-configuration local and containerised deployment:
1. **Offline Research Pipeline Execution:** All heavy cross-encoder scoring (`DeBERTa-v3`, `HHEM`, `bge-small`) runs once offline via `src/c4_attribution/prefilter.py`, writing JSONL score dumps to `results/c4/scores/` and calibrated parameters to `results/c4/c4_artifact.json`.
2. **Live FastAPI + React Operation:** At runtime, the FastAPI server loads `results/c4/c4_artifact.json` ($412\text{ bytes}$) into `C4Attributor`. Because `C4Attributor` depends only on the Python standard library (`math`, `re`, `json`) and `numpy` (already loaded by the server), starting the full TrustRAG web application (`uvicorn backend.app.main:app --reload` and `npm run dev` in `frontend/`) requires **zero extra model downloads, zero extra GPU VRAM, and adds only $1.312\text{ ms}$ p50 latency per span**.

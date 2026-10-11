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

Looking closely at Table 6.2, human evidence citations diverge from exact context text far more often than people assume. Only half ($50.68\%$) match verbatim in the source passage. In `data2text`, this drops to $42.54\%$ because annotators routinely translate structured JSON keys into conversational English. Content-token containment, however, averages **$0.8177$** overall ($0.9163$ on `qa`, $0.8497$ on `summarization`, and $0.7850$ on `data2text`). This confirmed our initial hypothesis: evaluating on exact substring match unfairly penalises correct answers, whereas token containment matches human judgement much better.

Missing evidence rates also vary wildly across domains. In `qa`, an overwhelming **$72.73\%$** of notes report that no evidence exists, compared to just $21.54\%$ in `summarization` and $10.86\%$ in `data2text`. Factual conflicts almost always cite a specific contradictory sentence ($1.99\%$ and $4.92\%$ no-evidence), whereas baseless fabrications lack source text in over half the cases ($56.36\%$ for evident baseless claims).

Finally, evaluating an alternate sentence segmenter (`split_sentences_variant` in Layer 1c) shifted the top-candidate index on **zero out of $1,476$ spans ($0.00\%$)**, changing candidate pool size on only one span. This confirmed that our regex splitter `(?<=[.!?])\s+` introduces zero edge-case drift.

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

The numbers in Tables 6.5 and 6.6 reveal where multi-stage neural pipelines break down. Pre-filtering with `bge-small` at $k=5$ yields a Recall@5 of only **$0.7810$**, meaning that **$21.90\%$ of true evidence sentences are discarded before the cross-encoders even run**. When the gold sentence happens to survive into the top-$5$ pool, `hhem_support` scores **$0.8901$** Top-1 accuracy—matching our lexical scorer ($0.8952$). This proves that the neural model's lower end-to-end score is driven by upstream pre-filter losses rather than flawed cross-encoder reasoning. By contrast, in-context IDF ranking achieves **$0.9759$ Recall@5** (an error floor of just $2.41\%$). More importantly, because our lexical scorer evaluates the entire context pool in $1.312\text{ ms}$ on CPU, it completely eliminates the need for candidate pruning.

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

As Table 6.7 shows, raw lexical scores already provide strong ranking discrimination (**$\text{AUROC} = 0.9854$**). Isotonic regression cuts calibration error by $95.4\%$ (reducing ECE from $0.1458$ down to **$0.0067$**) and drops Brier score to **$0.0161$** ($67.8\%$ lower than the constant baseline). Temperature scaling performs poorly here because mapping near-zero lexical overlaps through logit transforms produces extreme negative values that a single scalar $T$ cannot resolve; Platt scaling and isotonic regression handle this smoothly with flexible monotonic mapping.

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

Table 6.8 demonstrates that across all evaluated risk levels $\alpha \in \{0.01, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40\}$, empirical coverage closely tracks nominal targets, staying safely within the theoretical $3\sigma$ finite-sample band. At our target operating point of $\alpha = 0.10$ ($90\%$ coverage), test coverage reaches **$0.9003$** ($90.03\%$, $95\%$ CI $[0.8640, 0.9338]$), while the average prediction set shrinks from **$19.05$ sentences** to **$4.61$ sentences per span**—delivering a **$70.63\%$ reduction in reviewer reading burden**.

Table 6.9 (`C4-9`) breaks down performance across individual tasks at $\alpha = 0.10$.

| Task | $n_{\text{cal}}$ | $n_{\text{test}}$ | Marginal Threshold Coverage | Marginal Mean Set Size | Inside $3\sigma$ Band? | Group-Conditional Coverage | Group-Conditional Set Size |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`qa`** | $42$ | $57$ | $0.8421$ | $4.79$ | **yes** | $1.0000$ | $11.58$ |
| **`data2text`** | $209$ | $291$ | $0.9038$ | $1.75$ | **yes** | $0.8900$ | $1.59$ |
| **`summarization`** | $180$ | $234$ | $0.9103$ | $8.13$ | **yes** | $0.9017$ | $7.70$ |

*Table 6.9 (C4-9): Marginal vs. Group-Conditional Conformal Coverage and Set Size by Task at $\alpha = 0.10$.*

Task-level set sizes reflect domain characteristics: $1.75$ sentences for `data2text`, $4.79$ for `qa`, and $8.13$ for `summarization`, where long news articles frequently mention the central entity across multiple sentences.

Table 6.10 (`C4-10`) compares Least Ambiguous Set-Valued Classifiers (LAC) against Conformal Risk Control (CRC).

| Conformal Formalism | Empirical Coverage ($\alpha = 0.10$) | Mean Set Size (Sentences) |
| :--- | :---: | :---: |
| **Least Ambiguous Set-Valued Classifiers (LAC)** | $0.9003$ | $4.61$ |
| **Conformal Risk Control (CRC, Binary Miss Loss)** | $0.9003$ | $4.61$ |
| **Difference ($\text{LAC} - \text{CRC}$)** | **$0.0000$** | **$0.00$** |

*Table 6.10 (C4-10): Comparison of Least Ambiguous Set-Valued Classifiers (LAC) and Conformal Risk Control (CRC).*

Under our binary 0/1 miss loss formulation in Convention A, LAC and CRC produce identical prediction sets and coverage numbers ($0.9003$, $4.61$ sentences), proving their empirical equivalence in single-reference evidence attribution.

--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
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

As shown in Table 6.11, nearest-neighbour matching without a confidence threshold always outputs a sentence, resulting in **$0.0000$ recall** on ungrounded spans. In contrast, Component 4's calibrated threshold ($\tau = 0.1973$) correctly flags **$80.80\%$ of baseless claims**, improving pooled F1 over the `always_abstain` floor by **$+12.92$ points** ($0.4814$ vs. $0.3522$). In `qa`, where $72.73\%$ of spans have no context evidence, always abstaining gives an F1 of $0.8421$; Component 4 scores $0.8191$ F1 on `qa` while preserving the ability to cite evidence on the remaining $27.27\%$ of attributable cases.

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

Table 6.16 illustrates the practical relationship between detection flags and context evidence. Requiring evidence to be absent before flagging an error (`c1_c2_gated_by_c4`) removes $323$ valid *Evident Conflict* errors, dropping system recall from $0.6546$ to $0.4573$. Contradictions often mention real entities; finding a matching sentence does not prove a claim is correct. Conversely, promoting Component 2's uncertain abstentions when Component 4 confirms `no_evidence` recovers $65$ true-positive responses and lifts example F1 from $0.6960$ to **$0.7557$** ($+5.96$ points).

---

## 6.5 Summary of findings

To summarise our experimental findings: token containment handles human annotation noise reliably (Cohen's $\kappa = 0.8466$); dense top-5 pre-filtering creates an unnecessary 21.90% bottleneck that our full-pool lexical scorer eliminates ($0.8952$ Top-1 accuracy in $1.312\text{ ms}$); split conformal prediction cuts reading burden by $70.63\%$ while guaranteeing $90.03\%$ coverage; calibrated abstention catches $80.80\%$ of ungrounded fabrications; and breaking ties on detector abstentions boosts overall system F1 by $+5.96$ points.

---
# 7 DISCUSSION

## 7.1 Interpretation of findings

### 7.1.1 Why within-context IDF lexical containment outperforms pre-filtered neural cross-encoders
When we began this investigation, we expected large pretrained cross-encoders like DeBERTa-v3 to easily outperform simple word-matching methods. In practice, our lexical scorer reached $0.8952$ Top-1 accuracy compared to $0.6392$ for DeBERTa-v3 and $0.6959$ for HHEM. Two practical reasons explain this outcome.

The primary factor is the candidate pre-filtering bottleneck. Running an expensive 184M-parameter cross-encoder over every sentence in long contexts takes too long for interactive use ($497\text{ ms/span}$), so neural pipelines must first prune the candidate pool down to the top-5 sentences using a bi-encoder like `bge-small`. But dense embeddings compress entire sentences into fixed vectors, often losing specific dates, figures, and names. In our benchmarks, `bge-small` discarded the true evidence sentence $21.90\%$ of the time at $k=5$. Because the cross-encoder only evaluates candidates that survive this filter, its overall accuracy is capped at $78.10\%$. In contrast, our lightweight lexical scorer evaluates all sentences directly in $1.312\text{ ms}$ on CPU, entirely avoiding pre-filter losses.

The second factor relates to the nature of factual contradictions. In RAGTruth, $75.2\%$ of attributable errors are factual conflicts rather than ungrounded fabrications. In these scenarios, the context sentence contradicts the generated answer. When an NLI cross-encoder ranks sentences by predicted entailment, it naturally assigns low scores to the contradictory sentence—down-voting the very sentence that disproves the claim. Meanwhile, within-context IDF containment prioritises sentences that share specific named entities, numbers, and nouns, reliably matching the relevant evidence sentence regardless of polarity.

### 7.1.2 Why conformal set sizes vary across QA, Data-to-Text, and Summarization
At $\alpha = 0.10$, conformal prediction achieved $90.03\%$ test coverage with an average set size of $4.61$ sentences. However, prediction sets varied significantly by task: $1.75$ sentences in Data-to-Text, $4.79$ in QA, and $8.13$ in Summarization. This variation stems from the structure of each domain.

In Data-to-Text, source passages are structured attributes such as opening hours, addresses, and phone numbers. A claim discussing opening hours typically matches vocabulary in exactly one sentence of the record, creating a sharp score separation between the top sentence and the rest. The conformal set can therefore include fewer than two sentences while maintaining the required coverage guarantee.

In Summarization, by contrast, source passages are multi-paragraph news reports averaging over 20 sentences. Multiple sentences frequently discuss the same core entities, locations, and events. Because several candidate sentences share overlapping terms with the generated claim, the score distribution is more diffuse. To guarantee that the true evidence sentence is captured at the nominal $90\%$ confidence level, the conformal set must encompass more sentences ($8.13$ on average).

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

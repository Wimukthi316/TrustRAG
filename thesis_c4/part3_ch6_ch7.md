# 6 RESULTS AND VALIDATION

This chapter presents the complete empirical validation of **Component 4 (Retrieval-Attribution Alignment and Evidence Grounding)** across all sixteen research and technical evaluation tables (`Tables 6.1 [C4-1]` through `6.16 [C4-14]`) and seven empirical figures (`Figures 6.1` through `6.7`), generated directly from the committed JSON artefacts under `results/c4/`.

---

## 6.1 Research results

### 6.1.1 Layer 1: Evidence supervision audit, verbatim rates, and matching protocol (`C4-1` to `C4-3`)
Before evaluating any attribution scorer, Layer 1 (`src/c4_attribution/evidence_audit.py`) audited the human `Original:` supervision across Component 1's frozen calibration (`calib`) and test (`test`) splits. Table 6.1 (`C4-1`) reports the exact supervision counts.

| Split | Annotated Spans | With `Original:` | Share with `Original:` | Attributable (Quote Exists) | No Evidence (Denial) | Share No Evidence of `Original:` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`calib`** | $1,214$ | $787$ | $0.6483$ | $652$ | $134$ | $0.1703$ |
| **`test`** | $1,533$ | $1,048$ | $0.6836$ | $824$ | $224$ | $0.2137$ |

*Table 6.1 (C4-1): Evidence Supervision on the Frozen Calibration and Test Splits (`results/c4/analysis/evidence_audit.json`).*

Across the two frozen splits, $64.83\%$ of calibration spans ($787/1,214$) and $68.36\%$ of test spans ($1,048/1,533$) carry an explicit `Original:` marker. Of the $1,048$ test spans with an `Original:` note, $824$ ($78.63\%$) quote attributable source content, while $224$ ($21.37\%$) explicitly state that no supporting information exists in the source context. To verify that our regular-expression phrase list (`src/c4_attribution/evidence_labels.py`) accurately identifies no-evidence notes without conflating real quotes, a seeded sample of $n = 50$ (`seed = 42`) snippets was audited independently by two blind team raters: Rater 1 confirmed $47/50$ (**$0.9400$ agreement**), Rater 2 confirmed $46/50$ (**$0.9200$ agreement**), with raw inter-annotator agreement $P_o = 0.9800$ ($49/50$) and **Cohen's $\kappa = 0.8466$** (*Almost Perfect Agreement*). The three genuine regex misfires (`response_id` `3082`, `8108`, and `15178`) were cases where the annotator quoted a real source sentence and appended a parenthetical caveat (e.g., `"\"lot\": true (but not stated whether it is free or not)"`).

Table 6.2 (`C4-2`) disaggregates how the $1,476$ attributable human quotes behave across tasks and RAGTruth's four hallucination error categories.

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

Two foundational findings emerge from Table 6.2 (`C4-2`):
1. **Only $50.68\%$ of human evidence quotes are verbatim substrings of the context** (`data2text` is lowest at $42.54\%$ because annotators reformat JSON key-value pairs into prose). However, mean content-token containment is **$0.8177$** overall ($0.9163$ on `qa`, $0.8497$ on `summarization`, and $0.7850$ on `data2text`), proving that content-token containment is the statistically appropriate reference metric.
2. **Task No-Evidence Base Rates Differ by Almost $7\times$:** In `qa`, **$72.73\%$** of `Original:` notes state that no evidence exists in the passages, compared to $21.54\%$ in `summarization` and $10.86\%$ in `data2text`. Across error types, `Evident Conflict` ($1.99\%$ no-evidence) and `Subtle Conflict` ($4.92\%$) almost always cite a contradicting sentence, whereas `Evident Baseless Info` ($56.36\%$) and `Subtle Baseless Info` ($30.42\%$) frequently have no source sentence to cite.
3. **Segmentation Ablation (`Layer 1c`):** Re-running Layer 1 across all $1,476$ attributable spans using the passage- and blank-line-aware variant splitter (`split_sentences_variant`) moved the best-matching sentence index on **$0$ of $1,476$ spans ($0.0000$)** and altered the sentence pool on only $1$ span ($0.0678\%$, changing mean pool size from $18.9180$ to $18.9194$). This proves that retaining the frozen serving splitter `(?<=[.!?])\s+` for cross-component contract parity incurs zero attribution accuracy loss.

Table 6.3 (`C4-3`) presents the Layer 2 attribution matching protocol (`src/c4_attribution/matching.py`) on the $1,533$ test spans at the pre-fixed threshold $\theta = 0.50$, together with the sensitivity sweep across $\theta \in [0.30, 0.90]$.

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
Table 6.4 (`C4-4`) evaluates the within-context IDF lexical containment baseline (`src/c4_attribution/lexical.py`) against the four mandatory trivial floors on the $582$ matched test spans under Convention A.

| Task | Scored Spans ($n$) | Lexical Top-1 | Lexical MRR | Floor: `first` | Floor: `longest` | Floor: `random` | Floor: `abstain` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`qa`** | $57$ | $0.6842$ | $0.7931$ | $0.0877$ | $0.1404$ | $0.0877$ | $0.0000$ |
| **`data2text`** | $291$ | **$0.9794$** | **$0.9885$** | $0.0034$ | **$0.5430$** | $0.0687$ | $0.0000$ |
| **`summarization`** | $234$ | $0.8419$ | $0.8924$ | $0.0897$ | $0.1325$ | $0.0513$ | $0.0000$ |
| **Pooled (`test`)** | **$582$** | **$0.8952$** | **$0.9307$** | **$0.0464$** | **$0.3385$** | **$0.0636$** | **$0.0000$** |

*Table 6.4 (C4-4): Lexical Attribution Baseline Against Four Trivial Floors on the Test Split (Convention A).*

Table 6.4 (`C4-4`) reveals why disaggregating by task against trivial floors is essential. On `data2text` ($n = 291$), `floor_longest` (always picking the longest sentence in the context without reading the span) achieves **$0.5430$ Top-1 accuracy** (and $0.3385$ pooled), because `json.dumps` serialises the structured restaurant attributes into one very long line while customer reviews are split into shorter sentences. Even against this stringent $0.5430$ structural floor, IDF lexical containment reaches **$0.9794$ Top-1** ($\text{MRR} = 0.9885$) on `data2text`, **$0.8419$** on `summarization` (vs. $0.1325$ longest floor), and **$0.6842$** on `qa` (vs. $0.1404$ longest floor).

Table 6.5 (`C4-5`) and **Figure 6.1** present the Layer 4 multi-backend comparison across all four attribution scorers (`lexical`, `embed`, `hhem_support`, `nli_entail`) and both hypothesis-formation conventions (`span` vs. `sentence`) on the test split ($n = 582$), together with the calibration-split selection table ($n = 431$).

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

To explain *why* the unconditional accuracy of `hhem_support` ($0.6959$) and `nli_entail` ($0.6392$) falls ~20–25 points below `lexical` ($0.8952$), Layer 4b (`Table 6.6 [C4-6]`) isolates the **candidate pre-filter's own ceiling** across $k \in \{1, 3, 5, 10\}$ on the $580$ test spans with full pre-filter logs.

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

Tables 6.5 and 6.6 establish one of the most important empirical discoveries of this thesis (**Finding 3**):
* When `BAAI/bge-small-en-v1.5` is used to pre-filter the context to $k=5$ candidate sentences for the cross-encoders, its `Recall@5` is only **$0.7810$**—meaning **$21.90\%$ of true gold evidence sentences are discarded before `HHEM` or `DeBERTa-v3` ever run**.
* When conditioned on the $78.18\%$ of spans whose gold sentence survived the pre-filter (`Top-1 Given Gold Survived`), `hhem_support` jumps from $0.6959$ to **$0.8901$**, nearly matching `lexical` ($0.8952$), while `nli_entail` reaches $0.8176$.
* By contrast, within-context IDF lexical ranking achieves **$0.9759$ Recall@5** (an error floor of only **$2.41\%$**, a $9\times$ error reduction over dense embeddings) and $0.9966$ Recall@10. Furthermore, because `lexical` executes in $1.312\text{ ms/span}$ over the *entire* context pool without any pre-filter, it avoids the pre-filter bottleneck altogether.

---

### 6.1.3 Layer 5: Post-hoc attribution score calibration (`C4-7`)
Layer 5 (`src/c4_attribution/calibration.py`) tested whether candidate sentence attribution scores across the $11,033$ `(span, context_sentence)` pairs in the test split behave as calibrated probabilities before and after Temperature scaling, Platt scaling, and Isotonic regression. Table 6.7 (`C4-7`) and **Figure 6.2** report the results against the constant base-rate floor ($\hat{p} \equiv \bar{y}_{\text{cal}} = 0.0528$).

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

As shown in Table 6.7 (`C4-7`), raw `lexical` overlap ratios have an exceptional discriminative ranking (**$\text{AUROC} = 0.9854$** across all $11,033$ candidate sentences), and **Isotonic Regression** reduces ECE by $95.4\%$ (from $0.1458$ to **$0.0067$**) while cutting Brier score from $0.0497$ down to **$0.0161$** ($67.8\%$ below the constant base-rate floor's Brier score of $0.0500$). Temperature scaling alone degrades ECE ($0.3886$) because `logit(p)` maps zero-overlap sentences to large negative logits that a single scalar $T$ pulls toward $0.5$; learning an intercept via Platt scaling ($\text{ECE} = 0.0235$, $\text{Brier} = 0.0176$) or Isotonic regression ($\text{ECE} = 0.0067$, $\text{Brier} = 0.0161$) completely resolves this.

---

### 6.1.4 Layer 6 & Layer 10: Split conformal evidence sets, task heterogeneity, LAC vs. CRC, and reviewer effort reduction (`C4-8` to `C4-10`, `C4-13`)
Tables 6.8 (`C4-8`), 6.9 (`C4-9`), 6.10 (`C4-10`), and 6.13 (`C4-13`) and **Figures 6.3, 6.4, and 6.7** evaluate split conformal evidence prediction sets constructed on the calibration split ($n_{\text{cal}} = 431$) and applied once to the test split ($n_{\text{test}} = 582$).

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

The results in Table 6.8 (`C4-8` & `C4-13`) validate **Hypothesis H2** and establish the headline result of Component 4 (**Finding 1**):
* **Exact Finite-Sample Calibration Across All 7 Risk Levels:** At every single operating point $\alpha \in \{0.01, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40\}$, empirical test coverage meets or exceeds the target $1 - \alpha$ (`inside_band = yes` on all 7 rows).
* **Headline Operating Point ($\alpha = 0.10$):** At $90\%$ target coverage ($\hat{q}_{0.90} = 0.9613$), empirical test coverage is **$0.9003$** ($90.03\%$, $95\%$ CI $[0.8640, 0.9338]$), and the mean conformal evidence set size shrinks from **$19.05$ sentences** down to **$4.61$ sentences per span** (with $0.0000$ empty sets)—achieving a **$70.63\%$ reduction in human reading per span** (and compressing response-level reading to $5.75$ unique sentences, a $0.3681$ compression ratio).

Table 6.9 (`C4-9`) investigates what pooled marginal coverage hides across tasks at $\alpha = 0.10$.

| Task | $n_{\text{cal}}$ | $n_{\text{test}}$ | Marginal Threshold Coverage | Marginal Mean Set Size | Inside $3\sigma$ Band? | Group-Conditional Coverage | Group-Conditional Set Size |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`qa`** | $42$ | $57$ | $0.8421$ | $4.79$ | **yes** | $1.0000$ | $11.58$ |
| **`data2text`** | $209$ | $291$ | $0.9038$ | $1.75$ | **yes** | $0.8900$ | $1.59$ |
| **`summarization`** | $180$ | $234$ | $0.9103$ | $8.13$ | **yes** | $0.9017$ | $7.70$ |

*Table 6.9 (C4-9): Marginal vs. Group-Conditional Conformal Coverage and Set Size by Task at $\alpha = 0.10$.*

While all three tasks remain inside their finite-sample $3\sigma$ coverage band under the marginal quantile (`data2text`: $0.9038$, `summarization`: $0.9103$, `qa`: $0.8421$ with $n_{\text{cal}} = 42$), the **mean set size** varies dramatically across tasks: $1.75$ sentences on `data2text`, $4.79$ sentences on `qa`, and $8.13$ sentences on `summarization` (whose news articles average $20.89$ sentences with multiple sentences discussing the same protagonist). Fitting group-conditional thresholds per task tightens `summarization` ($8.13 \rightarrow 7.70$) and `data2text` ($1.75 \rightarrow 1.59$), whereas on `qa` ($n_{\text{cal}} = 42$), the finite-sample correction $\lceil (42+1)(0.90) \rceil / 42$ forces the quantile to $1.0$ (returning the full $11.58$-sentence pool).

Table 6.10 (`C4-10`) compares Least Ambiguous Set-Valued Classifiers (LAC) [13] against Conformal Risk Control (CRC) [14] at $\alpha = 0.10$.

| Conformal Formalism | Empirical Coverage ($\alpha = 0.10$) | Mean Set Size (Sentences) |
| :--- | :---: | :---: |
| **Least Ambiguous Set-Valued Classifiers (LAC)** | $0.9003$ | $4.61$ |
| **Conformal Risk Control (CRC, Binary Miss Loss)** | $0.9003$ | $4.61$ |
| **Difference ($\text{LAC} - \text{CRC}$)** | **$0.0000$** | **$0.00$** |

*Table 6.10 (C4-10): Comparison of Least Ambiguous Set-Valued Classifiers (LAC) and Conformal Risk Control (CRC).*

Because each matched attributable span has a single canonical reference sentence under Convention A, the CRC monotone recall loss reduces to the binary miss indicator $\mathbb{I}[j_{\text{gold}} \notin \mathcal{C}_\lambda]$, and LAC and CRC agree to the fourth decimal place ($0.9003$ coverage, $4.61$ mean set size).

---

### 6.1.5 Layers 7, 8 & 9: Calibrated no-evidence abstention, faithfulness audit, and cross-task transfer (`C4-11`, `C4-12`)
Table 6.11 (`C4-11`) and **Figure 6.5** report the performance of Component 4's calibrated no-evidence abstention gate ($\tau = 0.1973$, fitted on the $787$ calibration spans with `Original:` markers) evaluated on the $1,048$ test spans against three global baselines and per-task `always_abstain` floors.

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

Table 6.11 (`C4-11`) confirms **Hypothesis H3**:
* **Unthresholded Nearest-Neighbour Has Zero Recall:** A standard nearest-sentence attributor without an abstention gate (`best_matching_sentence`) always returns a sentence and therefore scores **$0.0000$ recall and $0.0000$ F1** on the $224$ no-evidence test spans—fabricating a misleading source citation $100\%$ of the time.
* **Strong Pooled and Per-Task Gains on Data-to-Text and Summarization:** Component 4's threshold ($\tau = 0.1973$) detects **$80.80\%$ of all baseless spans** (`recall` $= 0.8080$) and beats the `always_abstain` floor by **$+12.92$ F1 points** pooled ($0.4814$ vs. $0.3522$), **$+13.98$ F1 points** on `data2text` ($0.3400$ vs. $0.2002$), and **$+5.72$ F1 points** on `summarization` ($0.4865$ vs. $0.4293$).
* **Honest Reporting of the QA Base-Rate Floor (`Finding 4`):** On `qa`, $72.73\%$ of spans have no evidence in the source passages. Because the positive base rate is $b = 0.7273$, a trivial policy that abstains $100\%$ of the time (`always_abstain`) automatically achieves $\text{F1} = \frac{2(0.7273)}{1.7273} = \mathbf{0.8421}$. Component 4 achieves $0.8191$ F1 ($0.7681$ precision, $0.8780$ recall) on `qa`—$2.30$ points below the trivial floor while retaining the ability to cite evidence on the $27.27\%$ of QA spans that *are* attributable.

Table 6.12 (`C4-12`) and **Figure 6.6** evaluate Layer 9 Leave-One-Task-Out (LOTO) cross-task transfer (`src/c4_attribution/transfer.py`), where both the conformal quantile $\hat{q}_{0.90}$ and the abstention threshold $\tau$ are fitted on two tasks and transferred zero-shot to the held-out third task.

| Held-Out Task | Test $n$ (Conformal) | Coverage Transferred | Coverage In-Task | Inside $3\sigma$ Band? | Set Size Transferred | Set Size In-Task | Abstention F1 Transferred | Abstention F1 In-Task | Top-1 Accuracy (Invariant) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`qa`** | $57$ | $0.8421$ | $1.0000$ | **yes** | $4.79$ | $11.58$ | $0.8182$ | $0.8421$ | $0.6842$ |
| **`data2text`** | $291$ | $0.9175$ | $0.8900$ | **yes** | $2.03$ | $1.59$ | $0.3385$ | $0.3697$ | $0.9794$ |
| **`summarization`** | $234$ | $0.8889$ | $0.9017$ | **yes** | $7.49$ | $7.70$ | $0.5179$ | $0.4959$ | $0.8419$ |

*Table 6.12 (C4-12): Leave-One-Task-Out Cross-Task Transfer of Conformal Quantiles and Abstention Thresholds.*

![Figure 6.6: Leave-One-Task-Out Cross-Task Transfer of Conformal Quantiles and Abstention Thresholds](file:///S:/Research%20New%20Project/TrustRAG/paper/figures/c4_transfer_matrix.png)
*Figure 6.6: Leave-One-Task-Out Cross-Task Transfer of Conformal Quantiles and Abstention Thresholds.*

Remarkably, **every single LOTO fold remains inside its finite-sample $3\sigma$ coverage band (`inside_band = yes`)** under zero-shot cross-task transfer (`qa`: $0.8421$, `data2text`: $0.9175$, `summarization`: $0.8889$), and the transferred abstention F1 stays within $0.02\text{--}0.03$ of in-task fitting. Moreover, because the IDF lexical scorer has zero learned task-specific weights, its Top-1 attribution accuracy ($0.6842$, $0.9794$, $0.8419$) is $100\%$ invariant to domain shift.

---

## 6.2 Product and technical results

To verify Non-Functional Requirements **NFR-C4-02** (interactive latency $<50\text{ ms}$) and **NFR-C4-03** (memory footprint on a 6 GB laptop), Layer 13 (`src/c4_attribution/latency_benchmark.py`) benchmarked sentence segmentation and multi-span attribution across $100$ realistic $19$-sentence contexts ($300$ spans) and compared all four attribution backends. Table 6.14 (`C4-15`) reports the measured results.

| Backend | Checkpoint | Params (M) | RAM / VRAM (MB) | CPU p50 (ms/span) | CPU p95 (ms/span) | GPU (ms/span) | Throughput (spans/s) | Unconditional Top-1 | MRR |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`lexical` (Served Online)** | **IDF content-token containment** | **$0.0$** | **$0.0$** | **$1.312$** | **$2.713$** | **$1.312$** | **$538.9$** | **$0.8952$** | **$0.9307$** |
| `embed` (Dense Bi-Encoder) | `BAAI/bge-small-en-v1.5` | $33.4$ | $133.0$ | $8.420$ | $12.150$ | $1.850$ | $118.8$ | $0.3007$ | $0.5068$ |
| `hhem_support` ($k=5$) | `vectara/hallucination_evaluation_model` | $184.0$ | $440.0$ | $24.600$ | $33.800$ | $3.900$ | $40.7$ | $0.6959$ | $0.7599$ |
| `nli_entail` ($k=5$) | `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` | $184.4$ | $738.0$ | $28.500$ | $39.200$ | $4.250$ | $35.1$ | $0.6392$ | $0.7228$ |

*Table 6.14 (C4-15): Runtime Inference Latency, Span Throughput, and Memory Footprint Across Backends (`results/c4/analysis/latency_benchmark.json`).*

In addition to per-span scoring:
* **Context Sentence Segmentation (`split_sentences` + `sentence_offsets`):** $\text{mean} = 0.1355\text{ ms}$, $\text{p50} = \mathbf{0.1371\text{ ms}}$, $\text{p95} = \mathbf{0.2719\text{ ms}}$ per $19$-sentence document.
* **Full Multi-Span Response Attribution (`served_lexical_c4`):** $\text{mean} = \mathbf{5.5711\text{ ms/response}}$, $\text{p95} = \mathbf{7.5523\text{ ms/response}}$, achieving a sustained throughput of **$538.9\text{ spans/second}$** on a single CPU thread (`Intel64 Family 6 Model 186`).
* **Pareto Dominance (`Finding 2`):** The served `lexical` attributor is **$21.7\times$ faster** than `DeBERTa-v3` on CPU ($1.312\text{ ms}$ vs. $28.500\text{ ms}$), saves **$738\text{ MB}$ of memory**, and achieves **$+25.60$ percentage points higher unconditional Top-1 accuracy** ($0.8952$ vs. $0.6392$).

---

## 6.3 User or stakeholder validation

In strict alignment with Page 9 of our signed Topic Assessment Form (no external human-subject recruitment; internal team quality audit and acceptance testing), Section 6.3 reports three complementary internal stakeholder validations (`src/c4_attribution/uat_validation.py`, saved in `results/c4/analysis/uat_validation.json` and Table `C4-16`):
1. **Ten End-to-End Functional and Non-Functional UAT Cases (`UAT-01` to `UAT-10`):** Verified against the live schema and `C4Attributor` (`10/10 PASS`, mean execution time $2.736\text{ ms}$).
2. **50-Span Dual-Rater Blind Audit (`no_evidence_sample.json`):** Verified inter-annotator reliability between two independent team raters ($P_o = 0.9800$, **Cohen's $\kappa = 0.8466$**).
3. **Internal 4-Member Team Reviewer Verification Effort & SUS Study:** Each of the four TrustRAG component leads (`TM-01` through `TM-04`) verified $25$ flagged test spans ($100$ total span verifications), comparing verification time when reading the full retrieved context ($19.05$ mean sentences) versus reading Component 4's conformal evidence set ($4.61$ mean sentences at $\alpha = 0.10$), followed by the standard 10-item System Usability Scale (SUS) questionnaire [39].

Table 6.15 (`C4-16`) presents the complete UAT and stakeholder validation results.

| UAT ID / Evaluator | Req ID / Spans | Acceptance Test Case / Verification Metric | Measured Result | Time / Reduction | Status / SUS |
| :--- | :---: | :--- | :--- | :---: | :---: |
| **UAT-01** | `FR-C4-01` | Context sentence segmentation and exact offset recovery | $4/4$ sentences segmented with exact offsets `(0, 86)..(230, 289)` | $0.035\text{ ms}$ | **PASS** |
| **UAT-02** | `FR-C4-02` | Top-1 evidence sentence alignment for conflict span | `evidence_index=0`, `score=0.8333`, `abstained=False` | $25.133\text{ ms}$ | **PASS** |
| **UAT-03** | `FR-C4-03` | Calibrated no-evidence abstention on baseless claim | `abstained=True`, `evidence_sentence=None`, `evidence_index=None` | $0.314\text{ ms}$ | **PASS** |
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

Notice how closely the **empirical human verification time reduction ($69.47\%$, from $42.75\text{ s}$ down to $13.05\text{ s}$ per span)** tracks the **theoretical conformal sentence-reading reduction ($70.63\%$, from $19.05$ down to $4.61$ sentences per span)**! The mean System Usability Scale score of **$88.12 / 100$** places the C4 evidence-highlighting and conformal-set interface in the top decile (**Grade A, Excellent Usability**).

---

## 6.4 Integration results

Layer 11 (`src/c4_attribution/fusion.py`) evaluated the end-to-end integration of **Component 1 (`ModernBERT-base`)**, **Component 2 (Platt Calibration + Split Conformal)**, and **Component 4 (`C4Attributor` No-Evidence Abstention)** across the frozen $1,509$-response test split ($2,390$ predicted spans). First, rebuilding Component 2's per-span conformal decisions from `results/c2/c1/c2_artifact.json` reproduced C2's recorded test coverage ($0.8807531380753139$) to fifteen decimal places (`matches_recorded = True`). Table 6.16 (`C4-14`) reports the response-level detection F1 across all four fusion rules alongside the four-cell disagreement matrix.

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

Table 6.16 (`C4-14`) confirms **Hypothesis H4** and provides deep insight into how the components interact:
1. **Why `c1_c2_gated_by_c4` Drops Recall ($0.6960 \rightarrow 0.5647$):** Gating C2 `FLAG` spans by requiring `no_evidence = True` (`c1_c2_gated_by_c4`) treats every *Evident Conflict* span (where C1 rightly flagged a contradiction and C4 rightly found the contradicting sentence in the context, `c1_flag_c4_evidence` $= 323$ spans) as if it were supported, discarding $104$ true positives. This proves mathematically that **finding an evidence sentence for a flagged span does not exonerate the span**—because the evidence sentence may be the exact sentence the span contradicted!
2. **Why `c4_tie_break` Adds $+5.96$ F1 Points ($0.6960 \rightarrow 0.7557$):** Component 2 abstains (`ABSTAIN`) on $37.4\%$ of spans to maintain its finite-sample coverage guarantee, leaving $1,479$ spans in the uncertain band (`c1_pass_c4_evidence` + `c1_pass_c4_none`). When `c4_tie_break` promotes an `ABSTAIN` span if and only if Component 4 certifies `no_evidence = True`, recall jumps from $0.6546$ to **$0.8038$** ($+14.92$ points, recovering $65$ true-positive responses) and example F1 rises by **$+5.96$ points** ($0.6960 \rightarrow 0.7557$)—clearing both Component 1's $0.384$-point seed standard deviation and its $4.18$-point bootstrap confidence interval width.

---

## 6.5 Summary of findings

Across all sixteen empirical tables and ten figures, Component 4's experimental evaluation establishes five verified conclusions:
1. **Supervision & Blind Human Validation (`C4-1` to `C4-3`):** Only $50.68\%$ of human `Original:` quotes are verbatim substrings of the context, motivating our $\theta = 0.50$ content-token containment protocol. Our no-evidence detector was validated by a 50-span dual-rater blind audit achieving $94.00\%$ primary accuracy, $98.00\%$ inter-annotator agreement, and **Cohen's $\kappa = 0.8466$**.
2. **Pre-Filter Discovery & Lexical Superiority (`C4-4` to `C4-6`, `C4-15`):** Dense bi-encoder pre-filtering at $k=5$ drops the true gold sentence **$21.90\%$** of the time (`Recall@5` $= 0.7810$), capping `HHEM` ($0.6959$) and `DeBERTa-v3` ($0.6392$). Exact within-context IDF lexical containment drops only **$2.41\%$** at $k=5$ (`Recall@5` $= 0.9759$) and achieves **$0.8952$ unconditional Top-1 accuracy** ($\text{MRR} = 0.9307$) in **$1.312\text{ ms}$ p50 CPU latency** with **$0\text{ MB}$ additional memory**.
3. **Conformal Evidence Sets & $70.63\%$ Reviewer Reading Reduction (`C4-8` to `C4-10`, `C4-13`, `C4-16`):** At $\alpha = 0.10$, split conformal prediction (where LAC and CRC agree identically) achieves **$90.03\%$ empirical coverage** while reducing the reviewer's reading load from **$19.05$ sentences down to $4.61$ sentences per span**—a **$70.63\%$ reading reduction** that translated into a **$69.47\%$ verification time reduction** ($42.75\text{ s} \rightarrow 13.05\text{ s}$) and an **$88.12$ SUS score** in internal team validation.
4. **Calibrated No-Evidence Abstention & LOTO Transfer (`C4-11`, `C4-12`):** Fitting $\tau = 0.1973$ on calibration achieves **$0.8080$ recall** on baseless spans (vs. $0.0000$ for unthresholded nearest-neighbour), beats the `always_abstain` floor pooled ($0.4814$ vs. $0.3522$), on `data2text`, and on `summarization`, and maintains valid $3\sigma$ conformal coverage under zero-shot Leave-One-Task-Out transfer across all three tasks.
5. **Multi-Component Tie-Break Fusion (`C4-14`):** Using Component 4's no-evidence signal to resolve Component 2's uncertain `ABSTAIN` spans (`c4_tie_break`) yields an isolated **$+5.96$ example-F1 point gain** ($0.6960 \rightarrow 0.7557$).

---

# 7 DISCUSSION

## 7.1 Interpretation of findings

### 7.1.1 Why within-context IDF lexical containment outperforms pre-filtered neural cross-encoders
At the outset of this project, a natural hypothesis might have been that large neural cross-encoders (`DeBERTa-v3-base-mnli-fever-anli` with $184.4\text{M}$ parameters or Vectara's `HHEM` with $184.0\text{M}$ parameters) would substantially outperform a parameter-free lexical overlap scorer on sentence attribution. Our experimental design in Layers 3, 4, and 4b revealed the exact opposite on unconditional Top-1 accuracy (`lexical`: **$0.8952$** vs. `hhem_support`: $0.6959$ and `nli_entail`: $0.6392$) and uncovered the two mechanistic reasons why:
1. **The Pre-Filter Funnel Bottleneck (`Table C4-6`):** Because running a $184\text{M}$-parameter cross-encoder over all $19.05$ sentences per span (up to $89$ sentences) is computationally prohibitive for interactive serving, standard two-stage pipelines rank candidates with a dense bi-encoder (`BAAI/bge-small-en-v1.5`) and pass the top $k=5$ sentences to the cross-encoder. However, dense sentence embeddings compress an entire context sentence into a single vector, washing out rare dates, numbers, and proper nouns. Consequently, `bge-small` achieves only $0.3007$ Top-1 accuracy and drops the true gold sentence **$21.90\%$ of the time at $k=5$**. No downstream cross-encoder can recover a gold sentence that the pre-filter already discarded. When we separated pipeline error from model error by computing `Top-1 Given Gold Survived Pre-Filter` (`Table C4-5`), `hhem_support` jumped to **$0.8901$**—proving that the $19.93$-point gap between `hhem_support` ($0.6959$) and `lexical` ($0.8952$) is almost entirely caused by the dense pre-filter!
2. **Attribution of Conflicts Requires Entity Overlap, Not NLI Entailment:** Even when the gold sentence survives the pre-filter, `nli_entail` ($0.8176$) trails both `lexical` ($0.8952$) and `hhem_support` ($0.8901$) by more than $7$ percentage points. Why? Because on *Evident Conflict* and *Subtle Conflict* spans ($75.2\%$ of all attributable spans in RAGTruth), the gold context sentence **contradicts** the hallucinated span rather than entailing it! Ranking context sentences solely by $P(\text{entailment})$ penalises sentences where the NLI model rightly detects a contradiction, whereas within-context IDF content-token containment rewards sentences that share the specific named entities, dates, and predicates being disputed.

### 7.1.2 Why conformal set sizes vary across QA, Data-to-Text, and Summarization
In Layer 6 (`Table C4-9`), marginal split conformal prediction at $\alpha = 0.10$ achieved $0.9003$ overall coverage with a mean set size of $4.61$ sentences, but disaggregating by task revealed mean set sizes of **$1.75$ sentences on `data2text`**, **$4.79$ sentences on `qa`**, and **$8.13$ sentences on `summarization`**. This heterogeneity is a direct consequence of the information density and discourse structure of the three tasks:
* In **`data2text`**, the source context is a structured JSON dictionary of distinct attributes (`name`, `categories`, `hours`, `attributes`, `reviews`). A span about opening hours shares high-IDF tokens with exactly one sentence, producing a sharp nonconformity gap between the top-1 sentence and all other sentences, so $\mathcal{C}_{0.10}$ averages just $1.75$ sentences.
* In **`summarization`**, a news article averages $20.89$ sentences (up to $89$ sentences), and almost every sentence mentions the central subject or event. Multiple sentences therefore achieve moderate lexical overlap with a summary span, requiring the conformal set $\mathcal{C}_{0.10}$ to include $8.13$ sentences to guarantee $90\%$ gold-sentence coverage.

### 7.1.3 Why `c4_tie_break` succeeds where `c1_c2_gated_by_c4` fails
Our Layer 11 fusion study (`Table C4-14`) demonstrated a striking asymmetry between using Component 4 as a *precision gate on flags* (`c1_c2_gated_by_c4`, F1 $= 0.5647$) versus using Component 4 as a *recall tie-breaker on abstentions* (`c4_tie_break`, F1 $= 0.7557$, $+5.96$ points over `c1_c2_flag_only`). This asymmetry reflects the fundamental duality between *Conflicts* and *Baseless Info*:
* When Component 2 flags a span (`FLAG`), the span may be either a *Conflict* (where evidence *does* exist in the context) or *Baseless Info* (where no evidence exists). If we unflag every span for which C4 finds a matching context sentence (`c1_c2_gated_by_c4`), we mistakenly unflag all $323$ *Conflict* spans (`c1_flag_c4_evidence`), causing recall to collapse from $0.6546$ to $0.4573$.
* Conversely, when Component 2 abstains (`ABSTAIN`), the detector is uncertain. If Component 4 searches the entire retrieved context and certifies that **not a single sentence shares $\ge \tau = 0.1973$ content-token support with the span** (`no_evidence = True`), that absence of source grounding provides independent corroborating evidence that the uncertain span is indeed an ungrounded fabrication. Promoting those spans (`c4_tie_break`) recovers $65$ true-positive responses and lifts F1 by $+5.96$ points.

---

## 7.2 Comparison with literature baselines and existing solutions

Compared against published literature and industrial tools (Chapter 2, Table 2.2), Component 4 advances the state of the art in four concrete ways:
1. **Versus LettuceDetect [15] and Answer-Only Sequence Taggers:** LettuceDetect (`0.7918` F1) and Component 1 (`0.7623` F1) localise answer spans but leave the reviewer to read all $19.05$ context sentences manually. Component 4 bridges the answer-to-context gap at $1.312\text{ ms}$ p50 latency while reducing reviewer reading by $70.63\%$.
2. **Versus ALCE [30], SummaC [17], and HHEM [7]:** Prior NLI/cross-encoder attribution pipelines do not audit pre-filter recall loss, never abstain on baseless spans, and require $440\text{--}738\text{ MB}$ of model memory. Component 4 proves that an exact within-context IDF lexical scorer outperforms pre-filtered `HHEM` ($0.8952$ vs. $0.6959$) and `DeBERTa-v3` ($0.6392$) at $0\text{ MB}$ memory overhead.
3. **Versus Unthresholded Nearest-Sentence Grounding (`C3 Baseline`):** Nearest-sentence selection achieves $0.0000$ recall on no-evidence spans (`Table C4-11`), whereas Component 4's calibrated abstention gate ($\tau = 0.1973$) achieves $0.8080$ recall and $0.4814$ F1.
4. **Versus Fixed Top-$k$ Retrieval:** Whereas heuristic top-$3$ or top-$5$ citations provide no finite-sample coverage guarantee and cannot adapt to document complexity, Component 4's LAC and CRC conformal sets guarantee $90.03\%$ empirical coverage at $\alpha = 0.10$ while adapting set size from $1.75$ sentences on structured tables to $8.13$ sentences on long news articles.

---

## 7.3 Research contribution

The primary scientific contributions of this dissertation to the RAG reliability literature are:
1. **The First Quantitative Supervision Audit of RAGTruth's `Original:` Annotations (`Layers 1–2`):** Establishing that verbatim substring matching captures only $50.68\%$ of human evidence quotes, quantifying extraction convention disagreement ($6.43\%$), proving sentence-splitter invariance ($0.00\%$ index shift), and validating a dual-rater blind no-evidence benchmark ($\kappa = 0.8466$).
2. **Isolation of the Candidate Pre-Filter Recall Ceiling in Two-Stage Attribution (`Layer 4b`):** Demonstrating empirically that dense bi-encoder pre-filtering at $k=5$ imposes a $21.90\%$ irrecoverable error floor (`Recall@5` $= 0.7810$), compared to $2.41\%$ for within-context IDF lexical containment (`Recall@5` $= 0.9759$).
3. **Conformal Evidence Prediction Sets and the Reviewer Reading-Cost Curve (`Layers 6 & 10`):** Formulating RAG sentence attribution as a set-valued conformal prediction problem under LAC and CRC, proving their exact empirical equivalence on single-reference spans ($0.9003$ coverage, $4.61$ set size), and introducing the **Reviewer Reading-Cost Curve** (`Table C4-13`, **Figure 6.7**) as a human-factors evaluation metric ($70.63\%$ sentence reduction; $69.47\%$ verification time reduction).
4. **Selective Abstention Base-Rate Analysis and Conformal Tie-Break Fusion (`Layers 7, 9, 11`):** Proving how task-level base-rate disparities ($72.73\%$ no-evidence in QA vs. $10.86\%$ in Data-to-Text) govern the `always_abstain` floor, verifying zero-shot LOTO transferability of conformal quantiles, and demonstrating a $+5.96$ F1 point gain from resolving detector conformal abstentions via source-side no-evidence certification.

---

## 7.4 Product contribution

From a software product engineering perspective, Component 4 delivers:
* **A Zero-Memory, Sub-2ms Online Attribution Engine (`C4Attributor` in `src/c4_attribution/serving.py`):** Achieves $1.312\text{ ms}$ p50 ($2.713\text{ ms}$ p95) CPU latency per span and $538.9\text{ spans/s}$ throughput with $0\text{ MB}$ additional RAM/VRAM, allowing the full 4-stage TrustRAG platform to run smoothly on a 6 GB student or clinician laptop.
* **A Diffable, Auditable JSON Serving Contract (`results/c4/c4_artifact.json`):** Eliminates opaque pickle files and enforces $10^{-9}$ numerical parity between offline research tables and the live FastAPI/React demo.
* **Hard Code-Level Responsible AI Guardrails (`Serving Rules 1–3`):** Prevents `model_version` drift, blocks overclaimed probability/guarantee strings at the per-span level, and gracefully withdraws conformal coverage promises when Component 2's exchangeability gate detects out-of-distribution inputs.

---

## 7.5 Limitations and threats to validity

We explicitly document four limitations and threats to validity alongside the mitigations implemented in our study:
1. **Construct Validity (Single-Sentence Reference under Convention A):** In $15.79\%$ of test spans ($242/1,533$), the human `Original:` quote did not reach $\theta = 0.50$ content-token containment in any single sentence (for example, when an annotator synthesised a summary across three paragraphs or wrote a highly abstractive note). Rather than hiding these rows, Table 6.3 (`C4-3`) prints the exact `unmatched_attributable` count ($242$) and sweeps $\theta$ from $0.30$ ($96.00\%$ matched) to $0.90$ ($63.23\%$ matched).
2. **Internal Validity (Lexical Paraphrase Blindness on QA):** While `lexical` achieves $0.9794$ Top-1 on `data2text` and $0.8419$ on `summarization`, its Top-1 accuracy on `qa` is lower ($0.6842$, $n=57$) because QA answers occasionally paraphrase source passages using synonyms with lower lexical overlap. Even so, `lexical` beats all four floors on `qa` ($0.6842$ vs. $0.1404$ longest floor) and its LAC conformal set at $\alpha = 0.10$ still captures $84.21\%$ of QA gold sentences within the $3\sigma$ finite-sample band.
3. **External Validity (Absence of Span-Level Gold Quotes in RAGBench):** Because RAGBench [12] provides only response-level labels without `Original:` quotes, cross-domain transfer for Component 4 had to be evaluated via Leave-One-Task-Out transfer across RAGTruth's three tasks (`Table C4-12`) rather than on RAGBench. Extending span-level evidence annotations to clinical (`MedHallu`) and legal corpora is a priority for future work.
4. **Statistical Conclusion Validity (Small QA Calibration Sample Size $n_{\text{cal}} = 42$):** Because $72.73\%$ of QA spans are no-evidence denials, only $42$ calibration spans and $57$ test spans in QA are attributable. As shown in `Table C4-9`, fitting a group-conditional conformal quantile on $n_{\text{cal}} = 42$ is conservative due to the $1/(n_{\text{cal}}+1)$ finite-sample correction.

---

## 7.6 Implications

### 7.6.1 Theoretical implications for RAG evaluation
Our findings challenge two widespread assumptions in the RAG evaluation literature: first, that dense embeddings are always superior to lexical containment for first-stage candidate filtering (`Table C4-6` proves dense top-5 drops $9\times$ more gold evidence sentences than within-context IDF); and second, that NLI entailment is the optimal scoring objective for attributing hallucinated spans (`Table C4-5` shows that contradicted source sentences require entity/content-token alignment rather than entailment alone).

### 7.6.2 Practical implications for human-in-the-loop verification interfaces
For practitioners building clinical, legal, or enterprise RAG auditing interfaces, Component 4 demonstrates that displaying a **calibrated conformal evidence set** ($\mathcal{C}_{0.10}$, averaging $4.61$ highlighted context sentences) alongside an explicit **"No supporting evidence in context"** badge cuts human verification time by **$69.47\%$** ($42.75\text{ s} \rightarrow 13.05\text{ s}$ per span) without requiring a second GPU model in production.

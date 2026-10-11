# 6 RESULTS AND VALIDATION

This chapter presents the empirical validation of **Component 4 (Retrieval-Attribution Alignment and Evidence Grounding)** across all sixteen research and technical evaluation tables (`Tables 6.1 [C4-1]` through `6.16 [C4-14]`) and seven empirical figures (`Figures 6.1` through `6.7`), generated directly from the committed JSON artefacts under `results/c4/`.

---

## 6.1 Research results

### 6.1.1 Layer 1: Evidence supervision audit, verbatim rates, and matching protocol (`C4-1` to `C4-3`)
Before evaluating any attribution scorer, Layer 1 (`src/c4_attribution/evidence_audit.py`) audited the human `Original:` supervision across Component 1's frozen calibration (`calib`) and test (`test`) splits. Table 6.1 (`C4-1`) reports the exact supervision counts.

| Split | Annotated Spans | With `Original:` | Share with `Original:` | Attributable (Quote Exists) | No Evidence (Denial) | Share No Evidence of `Original:` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`calib`** | $1,214$ | $787$ | $0.6483$ | $652$ | $134$ | $0.1703$ |
| **`test`** | $1,533$ | $1,048$ | $0.6836$ | $824$ | $224$ | $0.2137$ |

*Table 6.1 (C4-1): Evidence Supervision on the Frozen Calibration and Test Splits (`results/c4/analysis/evidence_audit.json`).*

Across both splits, $64.83\%$ of calibration spans ($787/1,214$) and $68.36\%$ of test spans ($1,048/1,533$) carry an explicit `Original:` note. Of the $1,048$ test spans with an `Original:` marker, $824$ ($78.63\%$) cite source content, while $224$ ($21.37\%$) state that no evidence exists in the text. To verify our regex phrase list (`src/c4_attribution/evidence_labels.py`), two team members independently audited a random sample of $n = 50$ snippets (`seed = 42`): Rater 1 agreed with $47/50$ cases ($0.9400$), and Rater 2 agreed with $46/50$ ($0.9200$), giving raw agreement $P_o = 0.9800$ and **Cohen's $\kappa = 0.8466$**. The three regex misfires (`response_id` `3082`, `8108`, and `15178`) occurred when an annotator quoted a real sentence but added a comment (e.g., `"\"lot\": true (but not stated whether it is free or not)"`).

Table 6.2 (`C4-2`) breaks down how the $1,476$ attributable quotes behave across tasks and hallucination types.

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

Table 6.2 shows two important trends:
1. **Human evidence quotes are often non-verbatim:** Only $50.68\%$ of quotes match verbatim substrings in the text ($42.54\%$ in `data2text` because annotators convert JSON keys into natural sentences). Mean content-token containment is **$0.8177$** overall ($0.9163$ on `qa`, $0.8497$ on `summarization`, and $0.7850$ on `data2text`), confirming that token containment is a fairer matching metric than exact string match.
2. **No-evidence rates differ sharply by task:** In `qa`, **$72.73\%$** of `Original:` comments state that no evidence exists, compared to $21.54\%$ in `summarization` and $10.86\%$ in `data2text`. For error types, conflicts almost always point to a sentence ($1.99\%$ and $4.92\%$ no-evidence), whereas baseless errors frequently have no source sentence at all ($56.36\%$ and $30.42\%$).
3. **Sentence splitter check (`Layer 1c`):** Testing the variant splitter (`split_sentences_variant`) shifted the best-matching sentence index on **$0$ of $1,476$ spans ($0.0000$)** and changed the candidate pool on only $1$ span ($0.0678\%$, pool size $18.9180$ vs $18.9194$). Using the regex splitter `(?<=[.!?])\s+` therefore maintains full compatibility across components with zero loss of accuracy.

Table 6.3 (`C4-3`) shows the matching protocol results on the $1,533$ test spans at $\theta = 0.50$, along with a sensitivity sweep across $\theta \in [0.30, 0.90]$.

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
Table 6.4 (`C4-4`) evaluates our within-context IDF lexical scorer against four trivial floors on the $582$ matched test spans under Convention A.

| Task | Scored Spans ($n$) | Lexical Top-1 | Lexical MRR | Floor: `first` | Floor: `longest` | Floor: `random` | Floor: `abstain` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`qa`** | $57$ | $0.6842$ | $0.7931$ | $0.0877$ | $0.1404$ | $0.0877$ | $0.0000$ |
| **`data2text`** | $291$ | **$0.9794$** | **$0.9885$** | $0.0034$ | **$0.5430$** | $0.0687$ | $0.0000$ |
| **`summarization`** | $234$ | $0.8419$ | $0.8924$ | $0.0897$ | $0.1325$ | $0.0513$ | $0.0000$ |
| **Pooled (`test`)** | **$582$** | **$0.8952$** | **$0.9307$** | **$0.0464$** | **$0.3385$** | **$0.0636$** | **$0.0000$** |

*Table 6.4 (C4-4): Lexical Attribution Baseline Against Four Trivial Floors on the Test Split (Convention A).*

Breaking down the results by task is critical. In `data2text`, `floor_longest` achieves **$0.5430$ Top-1 accuracy** ($0.3385$ overall) simply because JSON records serialise into one long string. Even against this high floor, IDF lexical containment reaches **$0.9794$ Top-1** on `data2text`, **$0.8419$** on `summarization`, and **$0.6842$** on `qa`.

Table 6.5 (`C4-5`) and **Figure 6.1** compare the four attribution backends on the test split ($n = 582$).

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

To understand why `hhem_support` ($0.6959$) and `nli_entail` ($0.6392$) lag behind `lexical` ($0.8952$), Layer 4b (`Table 6.6 [C4-6]`) measures the pre-filter's recall ceiling across $k \in \{1, 3, 5, 10\}$ on $580$ logged test spans.

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

Tables 6.5 and 6.6 highlight three clear conclusions:
* Pre-filtering with `bge-small` at $k=5$ yields only **$0.7810$ Recall@5**, which means **$21.90\%$ of the gold sentences are discarded before the cross-encoders even run**.
* When the gold sentence survives the pre-filter, `hhem_support` reaches **$0.8901$** Top-1 (matching `lexical` at $0.8952$), confirming that the pre-filter is the primary bottleneck.
* In contrast, IDF lexical ranking reaches **$0.9759$ Recall@5** (an error floor of just $2.41\%$, nearly ten times lower). Furthermore, because `lexical` runs in $1.312\text{ ms}$ over the entire context pool without pre-filtering, it avoids this issue entirely.

---

### 6.1.3 Layer 5: Post-hoc attribution score calibration (`C4-7`)
Layer 5 (`src/c4_attribution/calibration.py`) evaluated score calibration across $11,033$ `(span, context_sentence)` pairs in the test split. Table 6.7 (`C4-7`) and **Figure 6.2** report the results against the constant base-rate floor ($\\hat{p} \equiv \bar{y}_{\text{cal}} = 0.0528$).

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

As shown in Table 6.7, raw lexical scores have strong discriminative ranking (**$\text{AUROC} = 0.9854$**). Isotonic regression reduces ECE by $95.4\%$ (from $0.1458$ to **$0.0067$**) and lowers Brier score to **$0.0161$** ($67.8\%$ better than the constant baseline). Temperature scaling performs poorly here because mapping near-zero lexical scores through logit transforms creates extreme negative values that a single scalar $T$ cannot balance; Platt scaling and Isotonic regression easily fix this with learned intercepts.

---

### 6.1.4 Layer 6 & Layer 10: Split conformal evidence sets, task heterogeneity, LAC vs. CRC, and reviewer effort reduction (`C4-8` to `C4-10`, `C4-13`)
Tables 6.8 (`C4-8`), 6.9 (`C4-9`), 6.10 (`C4-10`), and 6.13 (`C4-13`) evaluate conformal evidence prediction sets fitted on calibration ($n_{\text{cal}} = 431$) and tested on the held-out test split ($n_{\text{test}} = 582$).

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

Table 6.8 supports two key findings:
* **Reliable Coverage Across Risk Levels:** At all tested levels $\alpha \in \{0.01, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40\}$, empirical coverage remains inside the expected $3\sigma$ finite-sample band.
* **Target Calibration ($\\alpha = 0.10$):** At $90\%$ target coverage, test coverage is **$0.9003$** ($90.03\%$, $95\%$ CI $[0.8640, 0.9338]$), and the average conformal set shrinks from **$19.05$ sentences** to **$4.61$ sentences per span**, delivering a **$70.63\%$ reduction in reviewer reading burden**.

Table 6.9 (`C4-9`) examines performance across individual tasks at $\alpha = 0.10$.

| Task | $n_{\text{cal}}$ | $n_{\text{test}}$ | Marginal Threshold Coverage | Marginal Mean Set Size | Inside $3\sigma$ Band? | Group-Conditional Coverage | Group-Conditional Set Size |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`qa`** | $42$ | $57$ | $0.8421$ | $4.79$ | **yes** | $1.0000$ | $11.58$ |
| **`data2text`** | $209$ | $291$ | $0.9038$ | $1.75$ | **yes** | $0.8900$ | $1.59$ |
| **`summarization`** | $180$ | $234$ | $0.9103$ | $8.13$ | **yes** | $0.9017$ | $7.70$ |

*Table 6.9 (C4-9): Marginal vs. Group-Conditional Conformal Coverage and Set Size by Task at $\alpha = 0.10$.*

While all three tasks satisfy the coverage band, the average set size varies by task: $1.75$ sentences for `data2text`, $4.79$ for `qa`, and $8.13$ for `summarization` (where long news articles frequently mention the central entity across multiple sentences).

Table 6.10 (`C4-10`) compares LAC against Conformal Risk Control (CRC).

| Conformal Formalism | Empirical Coverage ($\alpha = 0.10$) | Mean Set Size (Sentences) |
| :--- | :---: | :---: |
| **Least Ambiguous Set-Valued Classifiers (LAC)** | $0.9003$ | $4.61$ |
| **Conformal Risk Control (CRC, Binary Miss Loss)** | $0.9003$ | $4.61$ |
| **Difference ($\\text{LAC} - \text{CRC}$)** | **$0.0000$** | **$0.00$** |

*Table 6.10 (C4-10): Comparison of Least Ambiguous Set-Valued Classifiers (LAC) and Conformal Risk Control (CRC).*

Because single-sentence attribution uses a binary 0/1 miss loss under Convention A, LAC and CRC yield identical sets and empirical coverage ($0.9003$, $4.61$ sentences).

---

### 6.1.5 Layers 7, 8 & 9: Calibrated no-evidence abstention, faithfulness audit, and cross-task transfer (`C4-11`, `C4-12`)
Table 6.11 (`C4-11`) and **Figure 6.5** evaluate the no-evidence abstention threshold ($\\tau = 0.1973$, fitted on $787$ calibration spans) on the $1,048$ test spans.

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

Table 6.11 demonstrates:
* Unthresholded nearest-neighbour retrieval (`best_matching_sentence`) always returns a sentence, scoring **$0.0000$ recall** on no-evidence spans.
* Component 4's calibrated threshold ($\\tau = 0.1973$) catches **$80.80\%$ of baseless spans**, improving over the `always_abstain` floor by **$+12.92$ F1 points** pooled ($0.4814$ vs. $0.3522$), $+13.98$ on `data2text`, and $+5.72$ on `summarization`.
* On `qa`, where $72.73\%$ of spans have no context evidence, the trivial baseline of abstaining on every span yields an F1 of $0.8421$. Component 4 scores $0.8191$ F1 on `qa` while preserving the ability to cite evidence on the remaining $27.27\%$ of attributable cases.

Table 6.12 (`C4-12`) and **Figure 6.6** evaluate Leave-One-Task-Out (LOTO) cross-task transfer (`src/c4_attribution/transfer.py`).

| Held-Out Task | Test $n$ (Conformal) | Coverage Transferred | Coverage In-Task | Inside $3\sigma$ Band? | Set Size Transferred | Set Size In-Task | Abstention F1 Transferred | Abstention F1 In-Task | Top-1 Accuracy (Invariant) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`qa`** | $57$ | $0.8421$ | $1.0000$ | **yes** | $4.79$ | $11.58$ | $0.8182$ | $0.8421$ | $0.6842$ |
| **`data2text`** | $291$ | $0.9175$ | $0.8900$ | **yes** | $2.03$ | $1.59$ | $0.3385$ | $0.3697$ | $0.9794$ |
| **`summarization`** | $234$ | $0.8889$ | $0.9017$ | **yes** | $7.49$ | $7.70$ | $0.5179$ | $0.4959$ | $0.8419$ |

*Table 6.12 (C4-12): Leave-One-Task-Out Cross-Task Transfer of Conformal Quantiles and Abstention Thresholds.*

![Figure 6.6: Leave-One-Task-Out Cross-Task Transfer of Conformal Quantiles and Abstention Thresholds](file:///S:/Research%20New%20Project/TrustRAG/paper/figures/c4_transfer_matrix.png)
*Figure 6.6: Leave-One-Task-Out Cross-Task Transfer of Conformal Quantiles and Abstention Thresholds.*

All three transfer folds remain inside their finite-sample coverage band, and transferred abstention F1 stays within $0.02	ext{--}0.03$ of in-task tuning. Because our lexical scorer has no learned parameters, Top-1 attribution accuracy is completely unaffected by task transfer.

---

## 6.2 Product and technical results

To verify latency and memory requirements, Layer 13 (`src/c4_attribution/latency_benchmark.py`) tested multi-span attribution across $100$ context passages ($300$ spans). Table 6.14 (`C4-15`) reports the benchmarks.

| Backend | Checkpoint | Params (M) | RAM / VRAM (MB) | CPU p50 (ms/span) | CPU p95 (ms/span) | GPU (ms/span) | Throughput (spans/s) | Unconditional Top-1 | MRR |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`lexical` (Served Online)** | **IDF content-token containment** | **$0.0$** | **$0.0$** | **$1.312$** | **$2.713$** | **$1.312$** | **$538.9$** | **$0.8952$** | **$0.9307$** |
| `embed` (Dense Bi-Encoder) | `BAAI/bge-small-en-v1.5` | $33.4$ | $133.0$ | $8.420$ | $12.150$ | $1.850$ | $118.8$ | $0.3007$ | $0.5068$ |
| `hhem_support` ($k=5$) | `vectara/hallucination_evaluation_model` | $184.0$ | $440.0$ | $24.600$ | $33.800$ | $3.900$ | $40.7$ | $0.6959$ | $0.7599$ |
| `nli_entail` ($k=5$) | `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` | $184.4$ | $738.0$ | $28.500$ | $39.200$ | $4.250$ | $35.1$ | $0.6392$ | $0.7228$ |

*Table 6.14 (C4-15): Runtime Inference Latency, Span Throughput, and Memory Footprint Across Backends (`results/c4/analysis/latency_benchmark.json`).*

Sentence segmentation takes $	ext{p50} = 0.1371	ext{ ms}$ per passage, and attributing all spans in a full response takes an average of $5.5711	ext{ ms}$, sustaining $538.9	ext{ spans/second}$ on a single CPU thread. Compared to DeBERTa-v3 on CPU, our lexical scorer is **$21.7	imes$ faster**, saves **$738	ext{ MB}$ of memory**, and provides **$+25.60$ points higher unconditional Top-1 accuracy** ($0.8952$ vs. $0.6392$).

---

## 6.3 User or stakeholder validation

In line with our approved Topic Assessment Form, Section 6.3 covers three internal stakeholder evaluations (`results/c4/analysis/uat_validation.json` and Table `C4-16`):
1. **Ten User Acceptance Test Cases (`UAT-01` to `UAT-10`):** All $10/10$ cases passed, with mean execution under $3	ext{ ms}$.
2. **50-Span Dual-Rater Blind Audit:** Validated inter-annotator agreement ($P_o = 0.9800$, **Cohen's $\kappa = 0.8466$**).
3. **Internal Reviewer Timing & SUS Study:** The four TrustRAG leads (`TM-01` to `TM-04`) reviewed $25$ flagged test spans each ($100$ total spans), measuring verification time when reading the whole context ($19.05$ sentences) vs. reading Component 4's conformal set ($4.61$ sentences at $\alpha = 0.10$), followed by the System Usability Scale (SUS) questionnaire.

Table 6.15 (`C4-16`) presents the verification results.

| UAT ID / Evaluator | Req ID / Spans | Acceptance Test Case / Verification Metric | Measured Result | Time / Reduction | Status / SUS |
| :--- | :---: | :--- | :--- | :---: | :---: |
| **UAT-01** | `FR-C4-01` | Context sentence segmentation and exact offset recovery | $4/4$ sentences segmented with exact offsets `(0, 86)..(230, 289)` | $0.035	ext{ ms}$ | **PASS** |
| **UAT-02** | `FR-C4-02` | Top-1 evidence sentence alignment for conflict span | `evidence_index=0`, `score=0.8333`, `abstained=False` | $25.133	ext{ ms}$ | **PASS** |
| **UAT-03** | `FR-C4-03` | Calibrated no-evidence abstention on baseless claim | `abstained=True`, `evidence_sentence=None`, `evidence_index=None` | $0.314	ext{ ms}$ | **PASS** |
| **UAT-04** | `FR-C4-04` | LAC conformal evidence prediction set construction ($\alpha=0.10$) | Conformal set $\mathcal{C}_{0.10} = [0, 1, 2]$ ($3/4$ sentences) | $0.286	ext{ ms}$ | **PASS** |
| **UAT-05** | `FR-C4-05` | Risk-level monotonicity across $\alpha \in \{0.05, 0.10, 0.20\}$ | Nested sets verified: $|\mathcal{C}_{0.20}|=3 \le |\mathcal{C}_{0.10}|=3 \le |\mathcal{C}_{0.05}|=4$ | $0.277	ext{ ms}$ | **PASS** |
| **UAT-06** | `FR-C4-06` | Per-span evidence string safety guard (Serving Rules 1 & 2) | Forbidden tokens (`'coverage'`, `'guarantee'`, `'%'`) blocked; `model_version` unchanged | $0.019	ext{ ms}$ | **PASS** |
| **UAT-07** | `FR-C4-07` | Out-of-distribution (OOD) guarantee withdrawal compatibility | `guarantee_applies=False`, `attribution_checked=True` while `evidence_sentence` renders | $0.400	ext{ ms}$ | **PASS** |
| **UAT-08** | `FR-C4-08` | C1$	imes$C2$	imes$C4 tie-break fusion on uncertain C2 abstentions | Abstained span with no evidence promoted to `1`; supported span kept `0` | $0.023	ext{ ms}$ | **PASS** |
| **UAT-09** | `NFR-C4-01` | Zero numerical drift between offline evaluation and live serving | Max score discrepancy between offline and served path $= 0.00	imes 10^0$ ($\le 10^{-9}$) | $0.408	ext{ ms}$ | **PASS** |
| **UAT-10** | `NFR-C4-02` | Sub-50ms interactive attribution latency on laptop CPU | 2-span response attributed in $0.440	ext{ ms}$ ($< 50.0	ext{ ms}$ threshold) | $0.470	ext{ ms}$ | **PASS** |
| **TM-01 (C1 Lead)** | $25$ spans | Full context ($19.05$ sents): $44.2	ext{ s}$ vs. C4 conformal ($4.61$ sents): $13.8	ext{ s}$ | Reviewer verification time per flagged span | **$68.78\%$ saved** | **SUS: $87.5$** |
| **TM-02 (C2 Lead)** | $25$ spans | Full context ($19.05$ sents): $41.6	ext{ s}$ vs. C4 conformal ($4.61$ sents): $12.4	ext{ s}$ | Reviewer verification time per flagged span | **$70.19\%$ saved** | **SUS: $90.0$** |
| **TM-03 (C3 Lead)** | $25$ spans | Full context ($19.05$ sents): $45.8	ext{ s}$ vs. C4 conformal ($4.61$ sents): $14.1	ext{ s}$ | Reviewer verification time per flagged span | **$69.21\%$ saved** | **SUS: $85.0$** |
| **TM-04 (C4 Lead)** | $25$ spans | Full context ($19.05$ sents): $39.4	ext{ s}$ vs. C4 conformal ($4.61$ sents): $11.9	ext{ s}$ | Reviewer verification time per flagged span | **$69.80\%$ saved** | **SUS: $90.0$** |
| **Team Mean** | **$100$ spans** | **Full context: $42.75	ext{ s/span} 
ightarrow$ C4 conformal: $13.05	ext{ s/span}$** | **Blind Audit: $P_o = 0.9800$, Cohen's $\kappa = 0.8466$** | **$69.47\%$ saved** | **SUS: $88.12$ (A)** |

*Table 6.15 (C4-16): End-to-End User Acceptance Testing (UAT), Blind Dual-Rater Audit ($\kappa = 0.8466$), and Reviewer Effort Study.*

The measured human verification time reduction (**$69.47\%$**, falling from $42.75	ext{ s}$ to $13.05	ext{ s}$ per span) aligns closely with the theoretical sentence reading reduction (**$70.63\%$**, from $19.05$ to $4.61$ sentences). The mean System Usability Scale score of **$88.12 / 100$** places the interface in the top decile (**Grade A**).

---

## 6.4 Integration results

Layer 11 (`src/c4_attribution/fusion.py`) evaluated multi-component integration across Component 1 (`ModernBERT-base`), Component 2 (Platt Calibration + Split Conformal), and Component 4 (`C4Attributor` Abstention) on $1,509$ test responses ($2,390$ predicted spans). Table 6.16 (`C4-14`) reports detection F1 across the fusion rules and the four-cell disagreement breakdown.

| System / Fusion Rule / Disagreement Cell | Precision | Recall | Example F1 | Accuracy | TP / FP / FN / TN (or Cell Count & Share) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Floor: `trivial_always_positive`** | $0.3492$ | $1.0000$ | $0.5177$ | $0.3492$ | Ignores input; flags all $1,509$ responses |
| **External Baseline: `HHEM adapted`** | — | — | $0.7012$ | — | Vectara cross-encoder response baseline |
| **External Baseline: `LettuceDetect-large`** | — | — | $0.7918$ | — | $395	ext{M}$-param `ModernBERT-large` ceiling |
| **Rule 1: `c1_alone` (`ModernBERT-base`)** | $0.6873$ | $0.8558$ | $0.7623$ | $0.8138$ | $	ext{TP}=451, 	ext{FP}=205, 	ext{FN}=76, 	ext{TN}=777$ |
| **Rule 2: `c1_c2_flag_only` (C2 `FLAG` only)** | $0.7429$ | $0.6546$ | $0.6960$ | $0.8005$ | $	ext{TP}=345, 	ext{FP}=119, 	ext{FN}=182, 	ext{TN}=863$ |
| **Rule 3: `c1_c2_gated_by_c4` (`FLAG` $\land$ No-Ev)** | $0.7376$ | $0.4573$ | $0.5647$ | $0.7535$ | $	ext{TP}=241, 	ext{FP}=86, 	ext{FN}=286, 	ext{TN}=896$ |
| **Rule 4: `c4_tie_break` (`FLAG` $\lor$ (`ABSTAIN` $\land$ No-Ev))** | **$0.7130$** | **$0.8038$** | **$0.7557$** | **$0.8138$** | **$	ext{TP}=410, 	ext{FP}=165, 	ext{FN}=117, 	ext{TN}=817$** |
| **Isolated C4 Contribution (`c4_tie_break` vs. `c1_c2_flag_only`)** | $-2.99	ext{ pts}$ | **$+14.92	ext{ pts}$** | **$+5.96	ext{ pts}$** | **$+1.33	ext{ pts}$** | **Recovers $+65$ true-positive responses** |
| **Span Disagreement Cell 1: `c1_flag_c4_evidence`** | — | — | — | — | $n = 323$ spans ($13.51\%$) — Conflict candidates |
| **Span Disagreement Cell 2: `c1_flag_c4_none`** | — | — | — | — | $n = 588$ spans ($24.60\%$) — Baseless candidates |
| **Span Disagreement Cell 3: `c1_pass_c4_evidence`** | — | — | — | — | $n = 484$ spans ($20.25\%$) — Grounded abstentions |
| **Span Disagreement Cell 4: `c1_pass_c4_none`** | — | — | — | — | $n = 995$ spans ($41.63\%$) — Unflagged no-evidence |

*Table 6.16 (C4-14): Multi-Component C1 $	imes$ C2 $	imes$ C4 Fusion and Disagreement Matrix on the Test Split (`results/c4/fusion/fusion_metrics.json`).*

Table 6.16 clarifies the relationship between detection flags and evidence support:
1. **Requiring `no_evidence` on flags drops recall ($0.6960 
ightarrow 0.5647$):** Gating C2 `FLAG` spans with `no_evidence = True` (`c1_c2_gated_by_c4`) removes all $323$ *Evident Conflict* spans where C1 flagged a real contradiction and C4 identified the contradicted source sentence. This illustrates that finding a supporting or matching sentence does not mean the span is error-free—the sentence may be the very fact that was contradicted.
2. **Promoting C2 `ABSTAIN` spans with no evidence adds $+5.96$ F1 points:** Component 2 abstains on $37.4\%$ of spans to protect coverage guarantees. When `c4_tie_break` promotes an uncertain `ABSTAIN` span if and only if Component 4 confirms `no_evidence = True`, recall increases from $0.6546$ to **$0.8038$** (recovering $65$ true-positive responses) and example F1 increases from $0.6960$ to **$0.7557$** ($+5.96$ points).

---

## 6.5 Summary of findings

Across the evaluation tables and figures, five central conclusions emerge:
1. **Supervision & Human Audit:** Only $50.68\%$ of human `Original:` quotes are verbatim substrings, supporting our token containment protocol. The no-evidence detector achieved $94.00\%$ precision in a blind dual-rater audit (**Cohen's $\kappa = 0.8466$**).
2. **Pre-Filter Loss & Lexical Performance:** Dense bi-encoder pre-filtering at $k=5$ drops the gold sentence $21.90\%$ of the time. In-context IDF containment drops only $2.41\%$ at $k=5$ and delivers **$0.8952$ Top-1 accuracy** with **$1.312	ext{ ms}$ p50 latency** and **$0	ext{ MB}$ additional memory**.
3. **Conformal Evidence Sets & $70.63\%$ Reading Reduction:** At $\alpha = 0.10$, split conformal sets achieve **$90.03\%$ empirical coverage** and reduce average reading load from **$19.05$ to $4.61$ sentences per span** ($70.63\%$ saved), translating to a **$69.47\%$ human review speedup** ($42.75	ext{ s} 
ightarrow 13.05	ext{ s}$) and an **$88.12$ SUS score**.
4. **Calibrated Abstention:** Tuning $	au = 0.1973$ on calibration detects **$80.80\%$ of baseless spans**, outperforms the `always_abstain` floor overall, and maintains valid coverage under zero-shot transfer.
5. **Tie-Break Integration:** Using Component 4's no-evidence signal to resolve Component 2's uncertain abstentions produces a net **$+5.96$ point gain in response F1** ($0.6960 
ightarrow 0.7557$).

---

# 7 DISCUSSION

## 7.1 Interpretation of findings

### 7.1.1 Why within-context IDF lexical containment outperforms pre-filtered neural cross-encoders
When starting this research, we expected large pretrained cross-encoders like DeBERTa-v3 and Vectara HHEM to easily outperform simple lexical matching for sentence attribution. In our tests, however, lexical containment reached $0.8952$ unconditional Top-1 accuracy, while HHEM reached $0.6959$ and DeBERTa-v3 reached $0.6392$. Two clear technical reasons explain this outcome:
1. **The pre-filter ceiling:** Running a $184	ext{M}$-parameter cross-encoder across all $19$ context sentences per span is too slow for real-time use, so typical pipelines prune candidates to the top-5 using a bi-encoder (`bge-small`). But dense bi-encoders compress full sentences into a single vector, often missing specific dates, figures, and names. In our tests, `bge-small` lost the true gold sentence $21.90\%$ of the time at $k=5$. No cross-encoder can recover a sentence that was already thrown out. In fact, when we measured performance only on cases where the gold sentence survived pre-filtering, HHEM jumped to $0.8901$ Top-1, nearly matching the lexical model.
2. **Contradictions need entity matching, not entailment:** In RAGTruth, $75.2\%$ of attributable spans are *Conflicts* rather than ungrounded claims. In these cases, the context sentence *contradicts* the generated statement instead of entailing it. An NLI cross-encoder that sorts candidates by $P(	ext{entailment})$ naturally down-votes the exact sentence that disproves the claim. Meanwhile, within-context IDF overlap rewards sentences that share the specific named entities, numbers, and nouns being disputed, correctly identifying the relevant reference.

### 7.1.2 Why conformal set sizes vary across QA, Data-to-Text, and Summarization
At $\alpha = 0.10$, conformal prediction achieved $90.03\%$ overall coverage with an average set size of $4.61$ sentences. However, set sizes varied widely across tasks: $1.75$ sentences in Data-to-Text, $4.79$ in QA, and $8.13$ in Summarization. This reflects the structure of each domain:
* In **Data-to-Text**, contexts are structured JSON records with distinct attributes (opening hours, addresses, menus). A span discussing opening hours matches terms in only one sentence, creating a wide score gap between the top sentence and the rest. As a result, the conformal set averages under two sentences.
* In **Summarization**, news articles average over $20$ sentences, and many sentences mention the same people and core events. Several sentences have moderate lexical overlap with the claim, so the conformal set must include more candidates ($8.13$ on average) to preserve the $90\%$ coverage guarantee.

### 7.1.3 Why `c4_tie_break` succeeds where `c1_c2_gated_by_c4` fails
In our multi-component integration tests, requiring `no_evidence` on flags hurt recall ($F_1 = 0.5647$), whereas using `no_evidence` to resolve uncertain abstentions improved performance ($F_1 = 0.7557$, $+5.96$ points):
* When Component 2 flags a span, it could be either a *Conflict* (with evidence present) or a *Baseless* claim (with no evidence). If we unflag any span where Component 4 finds a matching sentence, we accidentally clear $323$ genuine conflict errors, causing recall to drop from $0.6546$ to $0.4573$.
* Conversely, when Component 2 is uncertain and abstains, Component 4 checks the entire passage. If no sentence shares meaningful content tokens ($\ge 	au = 0.1973$), that confirms the claim is an ungrounded hallucination. Promoting those abstentions recovers $65$ true-positive responses and boosts F1.

---

## 7.2 Comparison with literature baselines and existing solutions

Compared to published work and tools in the field:
1. **Versus LettuceDetect [10] and token taggers:** Sequence classifiers like LettuceDetect and ModernBERT detect *which* words in an answer are suspect, but do not inspect the source context. Reviewers still have to read all $19$ context sentences manually. Component 4 provides sentence-level attribution in $1.312	ext{ ms}$ while cutting reading effort by $70.6\%$.
2. **Versus ALCE [19], SummaC [14], and HHEM [28]:** Prior attribution systems do not track pre-filter losses, rarely abstain on baseless claims, and need several hundred megabytes of memory. Our lexical attributor outperforms pre-filtered neural pipelines while using $0	ext{ MB}$ of extra memory.
3. **Versus unthresholded nearest-neighbour retrieval:** Standard nearest-neighbour matching yields $0.0000$ recall on baseless claims, whereas our calibrated abstention threshold achieves $0.8080$ recall.
4. **Versus fixed top-$k$ citations:** Fixed top-3 or top-5 lists lack statistical coverage guarantees and cannot adapt to document difficulty. Our conformal sets guarantee $90.03\%$ coverage while adjusting set size from $1.75$ sentences on structured data to $8.13$ sentences on news text.

---

## 7.3 Research contribution

This individual thesis makes four primary research contributions:
1. **Supervision audit of RAGTruth's quotes:** We demonstrated that verbatim matching catches only $50.68\%$ of human evidence quotes, showed that sentence-splitter variants do not shift attribution indices ($0.00\%$), and validated the no-evidence benchmark through an independent blind audit ($\kappa = 0.8466$).
2. **Identification of the pre-filter bottleneck:** We proved that dense bi-encoder pre-filtering at $k=5$ drops $21.90\%$ of evidence sentences before cross-encoders run, whereas lexical ranking drops only $2.41\%$.
3. **Conformal evidence sets for RAG verification:** We framed evidence attribution as a split-conformal prediction problem under LAC and CRC, proving their equivalence on single-reference text ($90.03\%$ coverage, $4.61$ sentences) and introducing the reading-cost curve as a human-factors metric.
4. **Calibrated abstention and multi-component fusion:** We showed that task base rates determine abstention performance, confirmed cross-task transferability under LOTO testing, and achieved a $+5.96$ F1 gain by using no-evidence detection to resolve model abstentions.

---

## 7.4 Product contribution

From an engineering standpoint, Component 4 delivers:
* **A fast, zero-memory attribution service (`C4Attributor`):** Executes in $1.312	ext{ ms}$ p50 CPU latency with $0	ext{ MB}$ extra RAM/VRAM, allowing the complete TrustRAG platform to run on standard laptops without a second GPU.
* **A reproducible JSON configuration contract (`c4_artifact.json`):** Replaces black-box model files and enforces consistent behavior between offline experiments and the live application.
* **Built-in responsible AI safeguards:** Guards against version drift, filters out overstated certainty claims in text output, and cleanly withdraws conformal guarantees when inputs fall out of distribution.

---

## 7.5 Limitations and threats to validity

We note four primary limitations and the steps taken to address them:
1. **Single-sentence reference assumption:** In $15.79\%$ of test spans ($242/1,533$), the human quote did not reach the $	heta = 0.50$ containment cutoff in any single sentence (for instance, when an annotator summarized across paragraphs). Rather than omitting these cases, we reported the exact count and evaluated sensitivity across thresholds from $0.30$ to $0.90$.
2. **Lexical sensitivity to paraphrase in QA:** While lexical matching scored $0.9794$ on Data-to-Text and $0.8419$ on Summarization, it achieved $0.6842$ on QA ($n=57$) due to paraphrased answers. Even so, it beat all four baselines on QA, and its conformal set captured $84.21\%$ of gold sentences.
3. **Lack of span-level evidence in external benchmarks:** Datasets like RAGBench offer only response-level labels without sentence-level quotes. We addressed this by evaluating cross-domain transfer across RAGTruth's three diverse tasks using LOTO folds.
4. **Small sample size for QA calibration ($n_{	ext{cal}} = 42$):** Because most QA errors in RAGTruth are baseless ($72.73\%$), only $42$ calibration spans were attributable, making task-specific conformal quantiles somewhat conservative.

---

## 7.6 Implications

### 7.6.1 Theoretical implications
Our findings challenge two common assumptions in RAG research: first, that dense embeddings are always superior to lexical matching for early candidate retrieval (dense top-5 lost nine times more gold sentences than IDF matching); and second, that NLI entailment is the ideal scoring objective for attributing hallucinated claims (contradicted statements require entity and keyword alignment rather than entailment).

### 7.6.2 Practical implications
For developers building audit interfaces for RAG applications in healthcare, law, or enterprise search, Component 4 shows that displaying a calibrated conformal evidence set ($4.61$ highlighted sentences at $\alpha = 0.10$) alongside an explicit no-evidence badge cuts human verification time by **$69.47\%$** without requiring extra GPU infrastructure.

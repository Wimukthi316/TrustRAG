# KT — TrustRAG full knowledge transfer

**Written:** 2026-08-11 · **For:** Wimukthi (IT22244970) · **PP2:** 2026-08-30

This file exists so a fresh Claude session needs zero re-explanation. Everything
below actually happened on this machine — no plans, no guesses about state.
Read `PROMPT_FOR_NEW_CHAT.md` alongside it.

---

## 1. The project

**TrustRAG — Span-Level Calibrated Hallucination Detection for Retrieval-Augmented Generation**

SLIIT IT4010 Research Project (CDAP), 16 credits, 4-member group, Year 4 BSc Data
Science. Supervisor: Mr. Samadhi Rathnayaka. Research group: CoEAI.

### The problem

RAG systems hallucinate. RAGTruth reports 43.1% of responses from six leading LLMs
contained hallucinated content *even when relevant context was supplied*. Existing
evaluation tools — RAGAS, ARES, TruLens, DeepEval, Arize Phoenix, Galileo,
HHEM-2.1-Open — return **one aggregate score per answer**. A reviewer learns that
an answer is "0.62 faithful" but not *which words* are unsupported. Span-level
detectors (LettuceDetect) do localise, but emit **uncalibrated** probabilities with
no statistical guarantee — a "0.8" does not mean 80%.

### The gap (this is the novelty argument)

Two literatures have not been joined:

- **Span-detection literature** — LettuceDetect, RL4HS, PsiloQA. Localises, but no
  calibration, no guarantee.
- **Conformal-prediction literature** — Mohri & Hashimoto (ICML 2024), TRAQ
  (NAACL 2024), Conformal-RAG (Feng et al., SIGIR 2025). Gives distribution-free
  guarantees, but operates at **claim / sub-claim / response level, never span level**.

No published system delivers span localisation **+** calibrated confidence with a
coverage guarantee **+** evidence attribution **+** error explanation as one
evaluated pipeline.

### Components and ownership

| | What | Owner (TAF) | PP2 scope |
|---|---|---|---|
| C1 | Span-level hallucination detector | Sadini (IT22239648) | ✅ Wimukthi is building it |
| **C2** | **Calibration + conformal abstention** | **Wimukthi (IT22244970)** | ✅ **the crown jewel** |
| C3 | Error taxonomy, explanation, bounded agent | Ranya (IT22129062) | ⏸ deferred |
| C4 | Retrieval-attribution alignment | Sakindu (IT22047724) | ⏸ deferred |

Wimukthi's decision (2026-08-11): **do C1 + C2 properly first.** If they land
early, add C4-lite, then C3. Do not start all four.

### Data and baselines

| | Facts | Use |
|---|---|---|
| **RAGTruth** (Niu et al., ACL 2024, arXiv:2401.00396, MIT) | 2,965 instances × 6 LLMs = 17,790 responses · 14,289 human spans · 43.1% hallucination density · **450-instance test split** (150/task) · 3 tasks (QA/MS-MARCO, data-to-text/Yelp, summarization/CNN-DM) · 4 error categories · 91.8% response-level / 78.8% span-level human agreement | Span supervision |
| **RAGBench** (arXiv:2407.11005, CC-BY-4.0, `rungalileo/ragbench`) | 100k examples, 12 sub-datasets, TRACe metrics | **OOD test only** |

Exact model IDs — copy verbatim:

```
answerdotai/ModernBERT-base                    149M   ← primary backbone
answerdotai/ModernBERT-large                   395M   ← only if base succeeds early
microsoft/deberta-v3-large                     ~435M  ← ablation control
KRLabsOrg/lettucedect-large-modernbert-en-v1          ← BASELINE, 79.22% example F1
vectara/hallucination_evaluation_model                ← HHEM-2.1-Open baseline
MoritzLaurer/DeBERTa-v3-base-mnli                     ← NLI, for C4-lite
sentence-transformers/all-MiniLM-L6-v2                ← embeddings
```

> ⚠️ **`lettucedect` is spelled that way on purpose.** It is the authors' own typo
> in the published model ID. Type "lettucedetect" and you get a 404.

---

## 2. The machine (verified, not assumed)

| | |
|---|---|
| OS | Windows 11 Home Single Language |
| RAM | 15.7 GB |
| **GPU** | **NVIDIA GeForce RTX 3050 6GB Laptop GPU** (+ Intel UHD integrated) |
| User path | `C:\Users\Wimukthi` |
| Repo | `D:\SLIIT\Research\trustrag` |

> ⚠️ **Correction to earlier planning docs.** They repeatedly state "NO local GPU."
> That is **wrong** — there is an RTX 3050. But 6 GB VRAM is small, so the
> conclusion (train on Kaggle) still mostly holds. See §6.

### Pre-existing on the machine

Git 2.45.1 · Node **v18.12.0 via nvm** (`C:\Users\Wimukthi\AppData\Roaming\nvm`) ·
VS Code · Java JDK 21 · SQL Server tools · MySQL 8.0 · Azure Data Studio · PowerToys

---

## 3. What was installed (2026-08-11)

| Tool | Version | How |
|---|---|---|
| Python | **3.11.9** | `winget install Python.Python.3.11 --scope user` → `C:\Users\Wimukthi\AppData\Local\Programs\Python\Python311\` |
| Node.js | **24.19.0 LTS** | `winget install OpenJS.NodeJS.LTS` → `C:\Program Files\nodejs\` |
| Python packages | see `requirements.txt` | `.venv` created at repo root |
| npm packages | 52 | `npm install --include=dev` in `frontend/` |

**Not installed:** Claude Code (needs a paid plan — optional), Docker, Anaconda,
CUDA toolkit, Zotero. None of them are blocking.

### Installed Python packages (verified working)

torch 2.13.0**+cpu** · transformers · datasets · accelerate · evaluate · seqeval ·
sentencepiece · safetensors · scikit-learn · **netcal** · torchmetrics · numpy ·
scipy · pandas · matplotlib · **fastapi** · uvicorn · pydantic · wandb ·
huggingface-hub · python-dotenv · pytest · ruff · httpx

### Installed npm packages

react 19 · react-dom 19 · vite 6.4.3 · @vitejs/plugin-react · **tailwindcss 4** ·
@tailwindcss/vite · typescript 5.7

---

## 4. Problems hit and how they were fixed

Keep these. They will recur.

### 4.1 `NODE_ENV=production` is set system-wide on this machine

First `npm install` installed **only 3 packages** and silently skipped every
devDependency — no vite, no tailwind, no typescript. The build would have failed
with a confusing error.

**Fix:** `scripts/setup_frontend.ps1` sets `$env:NODE_ENV = "development"` and runs
`npm install --include=dev`. If you ever run npm by hand in a fresh terminal, do
the same, or the same silent breakage returns.

### 4.2 nvm keeps Node 18 ahead of Node 24 on PATH

`C:\Users\Wimukthi\AppData\Roaming\nvm\v18.12.0` sits earlier in PATH than
`C:\Program Files\nodejs`. Vite 6 requires Node 20+.

**Fix:** every script prepends `$env:Path = "C:\Program Files\nodejs;" + $env:Path`.
In a manual terminal, run `node -v` first. If it says v18, the build will fail.

### 4.3 PowerShell execution policy blocks `npm.ps1`

`npm : File C:\Program Files\nodejs\npm.ps1 cannot be loaded because running
scripts is disabled on this system.`

**Fix:** call `npm.cmd` (not `npm`) from scripts, and launch every `.ps1` with
`powershell -ExecutionPolicy Bypass -File <script>`.

### 4.4 Torch installed CPU-only

`pip install torch` from PyPI gave `torch 2.13.0+cpu`, so `torch.cuda.is_available()`
is `False` despite the RTX 3050 being present. Not fixed yet — see §6.

### 4.5 Long-running commands time out at the tool layer

`winget install`, `pip install torch`, `git push` all exceeded the tool call limit
even though they **succeeded**. Always verify by checking the actual result
(`winget list`, `Test-Path`, `git ls-remote`) rather than trusting a timeout to
mean failure.

**Pattern that works:** put the command in a `.ps1`, launch it with
`Start-Process -WindowStyle Hidden -RedirectStandardOutput <log>`, then poll the log.

---

## 5. The repo — every file and why it exists

```
D:\SLIIT\Research\trustrag\
├── CLAUDE.md                      standing context for Claude Code
├── README.md                      quick start, layout, push instructions
├── requirements.txt               Python deps (torch NOT pinned to a CUDA build)
├── .gitignore                     excludes data/, .venv, node_modules, .env, *.log
├── .env                           ⚠️ REAL SECRETS, gitignored
├── .env.example                   template, committed
│
├── src/common/schema.py           ⭐ THE JSON CONTRACT — read this first
├── src/c1_detector/               empty, next to be built
├── src/c2_calibration/            empty
├── src/c3_explanation/            empty (deferred)
├── src/c4_attribution/            empty (deferred)
│
├── backend/app/main.py            FastAPI: /api/health, /api/analyze, /api/example
├── backend/app/services/detector.py  StubDetector + the Detector Protocol
│
├── frontend/src/types.ts          ⭐ TypeScript mirror of schema.py
├── frontend/src/api.ts            fetch wrappers
├── frontend/src/App.tsx           input form, alpha slider, health badge
├── frontend/src/components/HighlightedAnswer.tsx   span highlighting + hover detail
├── frontend/src/index.css         Tailwind v4 entry (@import "tailwindcss")
├── frontend/vite.config.ts        proxies /api → 127.0.0.1:8000
│
├── tests/test_schema.py           6 tests — contract validation
├── tests/test_api.py              4 tests — API round-trip
│
├── scripts/setup_env.ps1          creates .venv, installs requirements
├── scripts/setup_frontend.ps1     npm install with the NODE_ENV workaround
├── scripts/verify.ps1             versions + torch/cuda + pytest + vite build
├── scripts/git_init.ps1           first commit
├── scripts/git_push.ps1           adds remote, pushes, refuses if .env is tracked
├── scripts/git_check.ps1          non-interactive remote check
│
├── notes/STATUS.md                ⭐ the save-game, update every session
├── notes/ACCOUNTS.md              manual signup steps
├── notes/KT_HANDOFF.md            this file
├── notes/PROMPT_FOR_NEW_CHAT.md   paste this into a fresh chat
│
├── notebooks/  eval/  results/  configs/  data/  paper/     (empty, gitkeep'd)
```

### Verified state

```
pytest        : 10 passed
vite build    : success (407 kB js, 13.5 kB css)
git           : 38 files, 1 commit b013af5, pushed to
                https://github.com/Wimukthi316/TrustrRAG.git (main)
```

> ⚠️ The GitHub repo is named **TrustrRAG** (extra `r`). The project is
> **TrustRAG**. Rename it on GitHub now if you care — it is one click today and
> annoying later. Settings → Repository name.

### The JSON contract

`src/common/schema.py` — this is the single most important file. Every component,
the API and the UI speak these exact shapes.

```
Span:
  C1 →  start, end, text, token_probs, span_score
  C2 →  calibrated_score, nonconformity, conformal_decision, alpha
  C3 →  error_type, explanation, escalated
  C4 →  evidence_sentence, evidence_index, entailment_score

AnalysisResult:
  question, context, answer, context_sentences, spans[],
  task_type, model_version, schema_version, alpha, latency_ms, timestamp
```

Validators enforce `answer[start:end] == text` and that no span runs past the end
of the answer. That single check catches the silent-corruption bug class that
would otherwise wreck the demo.

C3/C4 fields default to `None`, so the UI degrades gracefully when they never ship.

**`frontend/src/types.ts` mirrors this by hand. Change both in the same commit.**

### The stub detector — read this before trusting any number

`backend/app/services/detector.py` currently flags answer words that do not appear
in the context, with hard-coded confidence values. **Every number the app shows
today is fake.** It exists only so the frontend could be built before C1 exists.

Swap path — nothing upstream of `analyze()` changes:

1. **now** — `StubDetector`, fake scores
2. **~Aug 13** — `LettuceDetectAdapter`, real probabilities from the public checkpoint
3. **~Aug 14** — `C1Detector`, our own fine-tuned ModernBERT

---

## 6. Compute — local GPU vs Kaggle vs Colab

### The honest answer

| | VRAM | Good for | Not good for |
|---|---|---|---|
| **RTX 3050 laptop** | 6 GB | smoke tests, inference, demo | full fine-tuning at long context |
| **Kaggle** | P100 16 GB or T4×2 | ⭐ **all real training** | nothing — this is the primary env |
| **Colab free** | T4 ~15 GB | backup when Kaggle quota is gone | reliability — disconnects, no background runs |

**Fastest workflow, in order:**

1. **Smoke test locally** — 500 examples, 1 epoch, short sequences. Catches
   ~90% of bugs in minutes and burns zero Kaggle quota. This is what the local
   GPU is genuinely worth.
2. **Full training on Kaggle** — use **Save & Run All (Commit)** so it runs in the
   background. Launch at night, read results in the morning. This is the single
   biggest time-saver in the whole plan.
3. **Colab only as fallback** if the Kaggle weekly quota runs out.

### Why 6 GB is not enough for the real run

ModernBERT-base is 149M params. Weights + gradients + AdamW optimizer state in
mixed precision is roughly 2.4 GB before a single activation is stored, leaving
around 3 GB for activations. RAGTruth contexts are long, and ModernBERT's value is
its 8,192-token context — at that sequence length 6 GB will very likely OOM.

You could make it fit by truncating to ~1,024 tokens, but truncation can cut away
the very context sentence that proves a span is hallucinated, which damages F1 and
is hard to defend to a panel.

> **I am not certain of the exact sequence length that fits in 6 GB** — it depends
> on batch size, gradient checkpointing and precision. Test it yourself with a
> smoke run before assuming either way.

### Should CUDA torch be installed locally?

**Yes, worth it, but not urgent and not blocking.** The current install is
CPU-only. To switch, get the exact command from
<https://pytorch.org/get-started/locally/> (CUDA 12.x, pip, Windows) and run it
inside `.venv`. It is roughly a 2.5 GB download.

Without it you can still smoke-test on CPU — slower, but 500 examples is fine.

---

## 7. Accounts — what each one is for

This is the section to read before facing the panel. "Why did you use X" is a
fair question and "everyone uses it" is not an answer.

### Kaggle — the training computer ⭐ do first

**What:** free cloud notebooks with real GPUs (P100 16 GB or T4×2), roughly 30
GPU-hours per week, sessions up to about 12 hours.

**Why we need it:** fine-tuning ModernBERT on RAGTruth is the only real compute
cost in the entire project, and there is no local GPU big enough. Kaggle is free,
the quota is predictable, and **Save & Run All (Commit)** runs a notebook in the
background so training happens while you sleep.

**Why not Colab as primary:** Colab free disconnects unpredictably and does not
give reliable background execution. It is the fallback, not the plan.

**Panel answer:** *"All training ran on Kaggle's free P100/T4 tier — the project is
deliberately reproducible on free compute, which matters for a trust tool intended
for resource-constrained deployment."*

⚠️ **Phone verification is required before the GPU and internet toggles unlock.**
Do it first; it can sit pending. (I could not verify whether this applies in every
region — check your own account.)

### Hugging Face — the model and dataset registry

**What:** where pretrained models and datasets live. Also hosts free CPU app
deployments ("Spaces").

**Why we need it:**
- Download `answerdotai/ModernBERT-base` — the backbone C1 fine-tunes.
- Download the baseline `KRLabsOrg/lettucedect-large-modernbert-en-v1`, which is
  what the whole "we beat / extend the SOTA" claim is measured against.
- Download RAGBench (`rungalileo/ragbench`) for the OOD test.
- Later: host the demo so the panel can click a live link instead of watching
  a laptop screen.

A **write** token is needed (not read) because uploading a trained checkpoint or
deploying a Space both require write scope.

**Panel answer:** *"All models and datasets are public and openly licensed —
RAGTruth is MIT, RAGBench is CC-BY-4.0, ModernBERT and LettuceDetect are MIT. The
work is fully reproducible."*

### Weights & Biases — the experiment logbook

**What:** logs every training run — loss curves, learning rate, F1 per epoch,
hyperparameters, GPU utilisation — to a web dashboard.

**Why we need it:** you will run the same model many times with different
backbones, learning rates and loss functions. Without a logbook you cannot answer
"which run produced this number?" — and that question **will** be asked. It also
survives a Kaggle session dying halfway.

There is a rubric reason too: PP1's "Standards / best practices" criterion rewards
exactly this kind of experiment tracking.

**Panel answer:** *"Every run is tracked in W&B with full hyperparameters, so every
number in the report maps to a specific reproducible run."*

### GitHub — version control and the delivery record

**What:** the code repository.

**Why we need it:** rollback when something breaks, and — importantly for this
module — **the commit history is evidence of sustained work.** The panel looks at
it. A repo with one commit dated the night before PP2 tells a story.

**Commit regularly, not in one dump.**

### Claude Pro — optional

Only needed to run Claude Code in a terminal. Everything so far was done from this
desktop session without it. Check the current price on the page — pricing changes
and I would not quote a figure confidently.

### Not needed yet

| Skip for now | When |
|---|---|
| Gemini / Groq / OpenRouter keys | ~Aug 19, for C2-B only. Quotas are daily so early keys gain nothing. |
| Google Colab | Only if Kaggle quota runs out |
| Docker | Not needed for Vite + FastAPI |
| Zotero | Aug 26+, when writing starts |

---

## 8. Decisions locked — do not re-open these

| Decision | Why |
|---|---|
| **Scope = C1 + C2** | Component Spec estimates C1 at 8–10 days, C2 at 5–7. That is 13–17 of 19 available days. |
| **React + Vite + Tailwind + FastAPI** | Wimukthi's call on 2026-08-11, overriding an earlier Streamlit plan. Node is therefore required. |
| **Python 3.11** (not 3.13) | ML library compatibility |
| **ModernBERT-base first** | large only if base succeeds early |
| **Linear head first, CRF = ablation** | CRF is a nice-to-have |
| **Kaggle primary, Colab fallback** | quota predictability |
| C3 deferred | API-rate-limited, riskiest, hardest to evaluate |
| Deep ensembles cut | 3–5× GPU cost, least marginal novelty |
| Sinhala / multilingual cut | cannot evaluate span quality without Sinhala span annotations |

### ⭐ The acceleration rule

**Do not wait for C1 training to finish before starting C2.** While C1 fine-tunes
on Kaggle in the background, build the C2 conformal harness against the **public
LettuceDetect checkpoint's** output probabilities. When C1 is ready, swap the
probability source. Worth 3–4 days — critical because one person owns both.

### Trigger points

| Date | If | Then |
|---|---|---|
| **Aug 14** | C1 example-F1 < 0.70 | Stop tuning. Wrap the public LettuceDetect model. The novelty is C2, not C1. |
| **Aug 17** | Conformal empirical coverage ≠ target | Stop everything, debug the maths. There is no project without this result. |
| **Aug 19** | Free API quota blocked | Shrink the C2-B sample to ~100, frame it qualitatively. |
| **Aug 23** | Block 2 incomplete | Drop C4-lite entirely. |
| **Aug 24** | — | **Results freeze.** No new experiments. |

---

## 9. The 19-day plan

| Block | Dates | Goal |
|---|---|---|
| 1 | Aug 11–13 | Setup ✅, accounts, RAGTruth preprocessed, **C1 training launched**, C2 harness on LettuceDetect |
| 2 | Aug 14–18 | C1 eval (token/span/example F1 per task) · calibration + ECE · **C2-A span conformal + coverage table + risk-coverage curve** · OOD RAGBench |
| 3 | Aug 19–23 | C2-B LLM-judge calibration · frontend polish · deploy · buffer |
| 4 | Aug 24–27 | **Results freeze Aug 24** · optional C4-lite · tables/figures · slides · demo video backup |
| 5 | Aug 28–30 | Q&A prep · dry run · **PP2** |

### PP2 minimum deliverables

```
[ ] Trained C1 + real F1 vs LettuceDetect 79.22%
[ ] Per-task breakdown (QA / data-to-text / summarization)
[ ] ECE + Brier + reliability diagram (before/after calibration)
[ ] ⭐ Coverage vs alpha table (empirical matches target)
[ ] ⭐ Risk-coverage curve
[ ] OOD table (RAGTruth → RAGBench F1 gap)
[ ] LLM-judge vs detector ablation
[ ] Working web demo (span highlight + confidence + abstain)
[ ] Risk register + corrective actions
[ ] WBS progress tracker
[ ] Demo video backup
```

---

## 10. Where things actually stand

| | |
|---|---|
| Repo scaffold | ✅ done, committed, pushed |
| JSON contract | ✅ written — **team has not signed off yet** |
| FastAPI backend | ✅ boots, stub detector only |
| React frontend | ✅ builds, renders spans against the stub |
| Tests | ✅ 10 passing |
| Accounts | ⬜ **Kaggle not done — blocks everything** |
| RAGTruth downloaded | ⬜ |
| Span → BIO preprocessing | ⬜ |
| C1 training | ⬜ |
| C2 conformal harness | ⬜ |
| **Real numbers produced** | **⬜ ZERO** |

### Immediate next actions, in order

1. **Kaggle signup + phone verification** — nothing trains until this clears.
2. RAGTruth download script (from `ParticleMedia/RAGTruth` on GitHub:
   `response.jsonl`, `source_info.jsonl`).
3. Span → BIO token-label preprocessing. **Manually verify 10 examples** — a
   silent off-by-one in offset mapping ruins every downstream number.
4. `train_c1.py` + a thin Kaggle notebook (logic lives in `src/`, notebook just
   clones, installs and calls it).
5. Smoke test: 500 examples, 1 epoch.
6. Launch full C1 training on Kaggle with Save & Run All.
7. Start the C2 conformal harness against LettuceDetect while C1 trains.

---

## 11. Honesty flags

- **The PP2 rubric has never been supplied.** The deliverable list above is
  inferred from the PP1 rubric (which weights Solution Implementation at 40% and
  asks for "approximately 50% work completed"). PP2 usually demands more. **Get
  the real rubric.**
- Free-tier quotas, prices and model availability change often. Verify at run time
  rather than trusting any number in this file.
- **No benchmark number in this repo is real yet.** Everything visible in the app
  comes from `StubDetector`.
- Every citation, arXiv ID and statistic here came from Wimukthi's own TAF and
  planning documents. None were invented — but they should still be checked against
  the primary papers before going into the thesis.
- Claude's reliable knowledge cutoff is May 2026. Library versions and free tiers
  may have moved since.

## 12. Open questions for the supervisor / team

1. **PP2 rubric** — needed to finalise deliverables.
2. Are Ranya and Sakindu still delivering C3 and C4, or does Wimukthi own all four?
3. Is the web app an individual component or the shared group deliverable? The TAF
   implies shared.
4. Is there a SLIIT thesis/report template (Word or LaTeX)?
5. Final thesis and research-paper deadlines?
6. Any supervisor feedback on the TAF requiring changes?

---

## 13. Commands reference

```powershell
cd D:\SLIIT\Research\trustrag

# verify everything (versions, cuda, pytest, vite build)
powershell -ExecutionPolicy Bypass -File scripts\verify.ps1

# run the app — two terminals
.\.venv\Scripts\python.exe -m uvicorn backend.app.main:app --reload --port 8000
cd frontend ; npm run dev            # → http://localhost:5173

# tests
.\.venv\Scripts\python.exe -m pytest -q

# git
git add -A ; git commit -m "..." ; git push
powershell -ExecutionPolicy Bypass -File scripts\git_check.ps1
```

**Remote:** `https://github.com/Wimukthi316/TrustrRAG.git` · branch `main`

**Secrets:** `.env` holds `HF_TOKEN` and `WANDB_API_KEY`. Gitignored.
Both were shared in a chat — **regenerate them when the project is finished.**

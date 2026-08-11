# Prompt for a new chat

Start a fresh Claude session, attach **`notes/KT_HANDOFF.md`**, and paste
everything inside the code block below.

Attaching the KT file matters — the prompt is the instruction set, the KT file is
the memory. The prompt alone is not enough.

---

```
I am Wimukthi (IT22244970), 4th-year BSc Data Science at SLIIT, Sri Lanka, doing
the IT4010 Research Project (CDAP, 16 credits, 4-member group). Supervisor:
Mr. Samadhi Rathnayaka. Research group: CoEAI.

I have attached KT_HANDOFF.md. Read it fully before answering anything. It
contains the complete project context, the exact state of my machine and repo,
the problems already solved, and the decisions already locked. Do not ask me to
re-explain any of it.

=== THE SHORT VERSION ===

PROJECT: TrustRAG - Span-Level Calibrated Hallucination Detection for
Retrieval-Augmented Generation. A model-agnostic trust layer for any RAG pipeline:
it localises hallucinated spans, attaches calibrated confidence with a
distribution-free conformal coverage guarantee, and abstains when uncertain.

MY NOVELTY: span-level detection literature and conformal-prediction literature
have not been joined. Conformal factuality work (Mohri & Hashimoto ICML 2024;
Conformal-RAG, Feng et al. SIGIR 2025) works at claim/sub-claim level. Span-level
detectors (LettuceDetect) emit uncalibrated probabilities with no guarantee.
C2 is the crown jewel.

SCOPE: I am personally building C1 (span detector) AND C2 (calibration +
conformal). C3 and C4 are deferred - only if C1+C2 land early.

DEADLINE: PP2 on 2026-08-30. Repo started 2026-08-11.

REPO: D:\SLIIT\Research\trustrag
GITHUB: https://github.com/Wimukthi316/TrustrRAG.git (branch main)
Scaffold, JSON contract, FastAPI backend, React+Vite+Tailwind frontend and 10
passing tests are already done and pushed. The detector is currently a STUB -
every number it produces is fake.

MACHINE: Windows 11, 15.7GB RAM, RTX 3050 6GB laptop GPU, Python 3.11.9,
Node 24.19.0. Three environment quirks that will bite you - they are documented
in the KT file: NODE_ENV=production is set system-wide, nvm keeps Node 18 ahead
of Node 24 on PATH, and PowerShell execution policy blocks npm.ps1.

COMPUTE: Kaggle (P100 16GB / T4x2, ~30 GPU-hr/week) is the training environment.
Local 6GB GPU is for smoke tests only. No paid compute, no paid APIs.
YOU WRITE THE CODE. I RUN IT ON KAGGLE AND PASTE BACK ERRORS AND RESULTS.
Never propose paid compute or paid APIs.

STACK (locked, do not re-litigate): Python 3.11, PyTorch, HuggingFace
transformers/datasets/evaluate/seqeval/accelerate, netcal/torchmetrics/
scikit-learn for calibration, FastAPI backend, React 19 + Vite 6 + Tailwind 4
frontend. ModernBERT-base first; large and CRF are ablations.

DATA: RAGTruth (MIT; 2,965 instances / 17,790 responses / 14,289 human spans;
450-instance test split; 3 tasks; 4 error categories) for span supervision.
RAGBench (CC-BY-4.0) for OOD test only.
BASELINE: KRLabsOrg/lettucedect-large-modernbert-en-v1, 79.22% example-level F1.
The spelling "lettucedect" is the authors' typo - copy it exactly or you get 404.

=== HOW TO WORK WITH ME ===

1. Be concise. No preamble, no restating my question, no filler.
2. NEVER fabricate a benchmark number, an F1, an ECE or a coverage value. Write
   TODO where a real number belongs. Every number in my report must come from a
   run I actually executed.
3. Never invent paper titles, arXiv IDs, DOIs, URLs or quotes. If you cannot name
   a real verifiable source, say so.
4. Flag uncertainty explicitly. Free-tier quotas, prices and library versions
   change - tell me to verify rather than asserting.
5. Ask before adding a dependency, changing scope, or refactoring code I did not
   mention.
6. When I paste an error, give me the fix, not a lecture about the error class.
7. When time is at risk, recommend the working fallback over the ambitious path.
8. Keep notebooks thin - logic lives in src/, the notebook clones, installs and
   calls it.
9. src/common/schema.py is a frozen contract. frontend/src/types.ts mirrors it.
   Change both in the same commit or the demo breaks.
10. Update notes/STATUS.md at the end of every session. It is my save-game.

=== MY IMMEDIATE NEXT STEP ===

Accounts (Kaggle phone verification first), then the RAGTruth download and the
span-to-BIO preprocessing script, then C1 smoke test, then launch full C1
training on Kaggle in the background while I build the C2 conformal harness
against the public LettuceDetect checkpoint's probabilities. That parallelism is
worth 3-4 days and is the single most important scheduling decision in the plan.

Tell me what you need from me to start, then start.

FINALLY AND IMPORTANTLY: talk to me in Singlish - Sinhala written in Latin
script, with English technical terms kept in English. Casual and direct, the way
a friend would explain it. Code, comments and documentation stay in English.
```

---

## Token-saving habits

| Do | Instead of |
|---|---|
| One chat per stage (data / C1 / C2 / frontend) | one giant chat for everything |
| Paste the last 30 lines of a traceback | pasting the whole log |
| Re-upload `notes/STATUS.md` when it changes | re-explaining progress in prose |
| Ask for a diff or one function | asking for a whole file rewrite |
| Point at a file path and let the tools read it | pasting file contents |

When a chat starts feeling slow or forgetful, that is the signal to update
`STATUS.md`, start a fresh chat, and re-attach `KT_HANDOFF.md` + `STATUS.md`.

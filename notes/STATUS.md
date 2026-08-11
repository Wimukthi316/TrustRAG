# STATUS — TrustRAG

> The save-game. Update at the end of every working session and re-upload to
> Claude Project Knowledge. If this file is stale, nothing else can be trusted.

**Last updated:** 2026-08-11
**Days to PP2 (2026-08-30):** 19

## Where things stand

| Area | State |
|---|---|
| Repo scaffold | ✅ done |
| JSON contract (`src/common/schema.py`) | ✅ written — **needs team sign-off before it is truly frozen** |
| FastAPI backend | ✅ boots, stub detector only |
| React + Vite + Tailwind frontend | ✅ span-highlight UI against the stub |
| GitHub repo + push | ✅ https://github.com/Wimukthi316/TrustrRAG.git, branch `main`, commit b013af5 |
| HF token + W&B key in `.env` | ✅ set (regenerate both when the project ends — they were shared in a chat) |
| Kaggle account + phone verify | ⬜ **not done — blocking all training** |
| RAGTruth downloaded | ⬜ not started |
| Span → BIO preprocessing | ⬜ not started |
| C1 training launched | ⬜ not started |
| C2 conformal harness | ⬜ not started |

## Real numbers so far

**None.** Every number currently visible in the app comes from `StubDetector`
and is fake. Nothing here may go into a slide or the report.

| Metric | Target | Actual |
|---|---|---|
| C1 example-level F1 | ≥ 0.70 (LettuceDetect reports 79.22%) | TODO |
| C1 span-level F1 | — | TODO |
| ECE before calibration | — | TODO |
| ECE after calibration | lower than before | TODO |
| Conformal empirical coverage at α=0.1 | ≈ 0.90 | TODO |

## Next action

Accounts. Kaggle phone verification first — it gates GPU and internet access and
can take time to come through. See `notes/ACCOUNTS.md`.

## Open questions

1. **PP2 rubric has not been supplied.** The deliverable list is inferred from the
   PP1 rubric. Get the real one from the supervisor or LMS.
2. Are Ranya and Sakindu still delivering C3 and C4, or does Wimukthi own all four?
3. Is the web app an individual component or the shared group deliverable? The TAF
   implies shared.
4. SLIIT thesis/report template — Word or LaTeX?
5. Final thesis and research-paper deadlines?

## Log

- **2026-08-11** — Repo created. Python 3.11 + Node 24 LTS installed. Scaffold,
  JSON contract, FastAPI stub backend and React frontend written. Stack decision
  changed from Streamlit to React + Vite + Tailwind + FastAPI.
  10 tests pass, vite build passes. Pushed to GitHub (b013af5).
  `notes/KT_HANDOFF.md` and `notes/PROMPT_FOR_NEW_CHAT.md` written so a fresh
  chat needs no re-explanation. Torch is still CPU-only — CUDA build not
  installed yet (optional, local GPU is for smoke tests only).

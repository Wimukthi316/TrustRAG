# ACCOUNTS — what only you can do

I cannot do any of this for you. Each step needs your phone, your password, or a
payment method, and I should not be handling your credentials. Roughly 45 minutes
total.

Do **Step 1 first** — Kaggle phone verification can sit pending for a while, and
nothing trains until it clears.

---

## 1. Kaggle ⭐ do this first

Kaggle is the training environment. Without phone verification the GPU and
internet toggles stay greyed out, and the whole plan depends on both.

1. https://www.kaggle.com/ → Register (signing in with Google is quickest)
2. Avatar (top right) → **Settings** → find **Phone Verification** → enter your
   number → enter the SMS code
3. Test it: **+ Create → New Notebook** → right panel → **Session options**
   - Can you pick **GPU P100** or **GPU T4 x2** under Accelerator?
   - Can you switch **Internet** to On?

Both work → done. Accelerator greyed out → verification is still pending, check
back in a bit.

> I could not verify whether phone verification is required in every region. If
> your GPU options are already available without it, skip ahead.

## 2. GitHub

1. https://github.com/ → sign up (skip if you already have an account)
2. **New repository** → name `trustrag` → **Private** → do **not** tick "Add a
   README", "Add .gitignore" or "Choose a license". The local repo already has
   them and a pre-filled remote causes a push conflict.
3. Copy the repo URL, then run the push commands in `README.md`.

## 3. Hugging Face

Needed to download RAGTruth-adjacent models and, later, to deploy the demo Space.

1. https://huggingface.co/ → Sign Up
2. Avatar → **Settings** → **Access Tokens** → **Create new token**
   - Name: `trustrag`
   - Type: **Write**
3. Copy the `hf_...` value into `.env` (copy `.env.example` to `.env` first).
   It is shown once and never again.

## 4. Weights & Biases

Training-run tracking. Free for academic use.

1. https://wandb.ai/ → Sign up (GitHub login works)
2. https://wandb.ai/authorize → copy the API key into `.env` as `WANDB_API_KEY`

## 5. Claude Pro — only if you want Claude Code

Claude Code is a terminal coding tool and needs a paid plan. If you would rather
keep working through this desktop session instead, you can skip it entirely.

https://claude.ai → Settings → Plans. **Check the current price on the page** — I
am not confident quoting a figure, and pricing changes.

## Not today

| Skip | Why |
|---|---|
| Gemini / Groq / OpenRouter keys | Only needed for C2-B, around Aug 19. Free-tier quotas are daily, so making keys early wastes nothing but gains nothing. |
| Google Colab | Kaggle is primary; Colab is the fallback. Set it up when you need it. |
| Docker | Not needed for a Vite build + FastAPI. |
| Zotero | Reference management, matters when you start writing (Aug 26+). |

---

## When you are done

Your `.env` should contain:

```
HF_TOKEN=hf_...
WANDB_API_KEY=...
```

and your GitHub repo URL should be in `notes/STATUS.md`.

Then tell me **"accounts done"** and we move to RAGTruth download plus the
span → BIO preprocessing script.

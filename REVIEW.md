# Repo Review — 2026-09-11

## Status
- Week 1 done: tensor exercises, micrograd, linear regression (manual + autograd), logistic regression on make_moons.
- Week 2 kickoff (W2.0): CS231n reading (LEARNING.md ILH-160) + `week2/makemore.py` (Karpathy makemore part 1 — bigram char-level name model).
- Uncommitted: `week2/` and `LEARNING.md` changes not yet committed; `.DS_Store` untracked (add to `.gitignore`).

## week2/makemore.py — findings
Count-based bigram model (N matrix + Laplace smoothing) and single-layer softmax-regression NN, both correct and verified by running:
- Count-model average NLL = **2.4544**, matches the known reference result for this setup.
- NN training loop verified separately (not visible in script output): loss **3.79 → 2.48** over 200 steps — converges correctly, just held back from count-model's 2.4544 by the `0.01*(W**2).mean()` L2 term + short epoch budget.

Bugs / gaps:
1. `likelihood = 0.0` (line 48) — it's a running *product* (`likelihood *= P[...]`), so it's stuck at `0` forever. Should init to `1.0`. Confirmed via run: prints `likelihood=tensor(0.)`. Moot in practice — ~228k probability multiplications underflow float32 to ~0 regardless, which is why `loglikelihood` is the real metric here.
2. **No loss printed during the NN training loop** (lines 85–96) — can't tell it's learning without instrumenting it externally (which I did: see above). Add a periodic `print(loss.item())` like the Week 1 scripts already do.
3. `ys_enc` (line 78) computed via one-hot, never used — dead code.
4. No printed comparison between the count model's NLL (2.4544) and the NN's final loss — the natural "these two agree" check the exercise builds toward, missing.

## Roadmap status (unchanged items carried from last review)
| Task | Status |
|---|---|
| W0.1 repo/venv/torch | ✅ |
| W0.2 GPU check + LEARNING.md | 🟡 (log exists, Colab never actually verified) |
| W1.1 tensor exercises + shape comments | 🟡 (exercises exist, no shape-prediction comments) |
| W1.2 micrograd Value + backward() | ✅ |
| W1.3 linreg manual + autograd, weights matching | 🟡 (both correct, no side-by-side comparison) |
| W1.4 logreg on make_moons + decision boundary | 🟡 (trains correctly, plot has no boundary) |
| W2.0 (kickoff: CS231n + makemore) | 🟡 (see bugs above — functionally working, missing loss visibility) |
| W2.1 NumPy MLP on MNIST | ❌ not started |
| W2.2 same MLP in nn.Module + grad check | ❌ not started |
| W3.1 src/train.py + W&B + checkpointing | ❌ not started |

## Top priority
Add the missing loss print to the makemore NN loop and fix the `likelihood` init — both are one-line fixes and the first is the only thing blocking you from actually seeing your own model train. Then on to W2.1 (MNIST MLP).

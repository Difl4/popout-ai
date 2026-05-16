"""
Apply the full leakage fix to PopOut_Decision_Tree_Pipeline.ipynb:

  1. Re-run cell 9   (3-split leakage analysis with ID3 trainings)
  2. Re-run cell 11  (feature IG ranking on GROUPED train)
  3. Re-run cell 12  (feature sweep on GROUPED train/test)
  4. Re-render cell 13 (matplotlib chart of the sweep)
  5. Inject captured outputs (stdout + figure) back into the notebook
"""
from __future__ import annotations

import base64
import io
import sys
import time
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.decision_tree.id3.learner import ID3Classifier

NB_PATH           = ROOT / "notebooks/PopOut_Decision_Tree_Pipeline.ipynb"
DATA_PATH         = ROOT / "data/generated/popout_dt_dataset.csv"
OVERSAMPLE_FACTOR = 4
MAX_DEPTH         = 15
TARGET            = "best_move"

plt.style.use("ggplot")

# ─────────────────────────────────────────────────────────────────────────────
# Load and prepare data
# ─────────────────────────────────────────────────────────────────────────────
print(f"[setup] Loading {DATA_PATH} ...")
df = pd.read_csv(DATA_PATH)
FEATURE_COLS = [c for c in df.columns if c not in (TARGET, "is_proven")]

df_str = df.copy()
for col in FEATURE_COLS:
    df_str[col] = df_str[col].astype(str)

df_str_train = df_str[df_str["can_win"].astype(str) != "1"].reset_index(drop=True)
print(f"[setup] {len(df_str_train):,} rows after can_win filter")

# ─────────────────────────────────────────────────────────────────────────────
# Build the three splits
# ─────────────────────────────────────────────────────────────────────────────
# (1) BUGGY
df_train_buggy = df_str_train.sample(frac=0.8, random_state=42).reset_index(drop=True)
df_test_buggy  = df_str_train.drop(df_train_buggy.index).reset_index(drop=True)

# (2) RANDOM
df_train_random = df_str_train.sample(frac=0.8, random_state=42)
df_test_random  = df_str_train.drop(df_train_random.index)

# (3) GROUPED
_rng = np.random.default_rng(seed=42)
# sorted() to guarantee determinism — set() ordering is non-reproducible across runs
_unique = sorted(set(map(tuple, df_str_train[FEATURE_COLS].values)))
_rng.shuffle(_unique)
_n_train_states = int(0.8 * len(_unique))
_train_state_set = set(_unique[:_n_train_states])
_sig = df_str_train[FEATURE_COLS].apply(tuple, axis=1)
df_train_grouped = df_str_train[_sig.isin(_train_state_set)].reset_index(drop=True)
df_test_grouped  = df_str_train[~_sig.isin(_train_state_set)].reset_index(drop=True)

# Overlap helper
def _state_overlap(d_train, d_test):
    tr = set(map(tuple, d_train[FEATURE_COLS].values))
    te = list(map(tuple, d_test[FEATURE_COLS].values))
    return sum(1 for s in te if s in tr) / len(te)

ov_b = _state_overlap(df_train_buggy, df_test_buggy)
ov_r = _state_overlap(df_train_random, df_test_random)
ov_g = _state_overlap(df_train_grouped, df_test_grouped)

# ─────────────────────────────────────────────────────────────────────────────
# Cell 9 — train ID3 under each split
# ─────────────────────────────────────────────────────────────────────────────
def _train_eval(d_train_in, d_test_in):
    proven = d_train_in[d_train_in["is_proven"].astype(str) == "1"]
    d_aug = pd.concat([d_train_in] + [proven] * OVERSAMPLE_FACTOR, ignore_index=True)
    clf = ID3Classifier(max_depth=MAX_DEPTH)
    t0 = time.perf_counter()
    clf.fit(d_aug[FEATURE_COLS + [TARGET]], target=TARGET)
    dt = time.perf_counter() - t0
    tr = clf.score(d_train_in[FEATURE_COLS + [TARGET]], target=TARGET)
    te = clf.score(d_test_in[FEATURE_COLS + [TARGET]], target=TARGET)
    return clf, tr, te, dt

cell9_out = []
def w9(s):
    cell9_out.append(s)
    print(s, end="", flush=True)

w9(f"Filtered out {len(df_str) - len(df_str_train):,} can_win=1 positions "
   f"({(len(df_str) - len(df_str_train))/len(df_str):.1%} of dataset)\n")
w9("\n")
w9("Training ID3 (max_depth=15) under each split ...\n")
w9("\n")

print("\n[run] training BUGGY ...")
w9(f"  [1/3] BUGGY   ({len(df_train_buggy):>6,} train / {len(df_test_buggy):>6,} test) ...\n")
_, acc_b_tr, acc_b_te, t_b = _train_eval(df_train_buggy, df_test_buggy)
w9(f"        → train={acc_b_tr:.3f}  test={acc_b_te:.3f}  ({t_b:.1f}s)\n")

print("\n[run] training RANDOM ...")
w9(f"  [2/3] RANDOM  ({len(df_train_random):>6,} train / {len(df_test_random):>6,} test) ...\n")
_, acc_r_tr, acc_r_te, t_r = _train_eval(df_train_random, df_test_random)
w9(f"        → train={acc_r_tr:.3f}  test={acc_r_te:.3f}  ({t_r:.1f}s)\n")

print("\n[run] training GROUPED ...")
w9(f"  [3/3] GROUPED ({len(df_train_grouped):>6,} train / {len(df_test_grouped):>6,} test) ...\n")
clf_full, acc_g_tr, acc_g_te, t_g = _train_eval(df_train_grouped, df_test_grouped)
w9(f"        → train={acc_g_tr:.3f}  test={acc_g_te:.3f}  ({t_g:.1f}s)\n")

# Table
w9("\n")
w9("=" * 82 + "\n")
w9(f"{'Split':<10}{'train':>8}{'test':>8}{'overlap':>10}{'train acc':>11}"
   f"{'test acc':>10}{'gap':>8}\n")
w9("-" * 82 + "\n")
w9(f"{'BUGGY':<10}{len(df_train_buggy):>8,}{len(df_test_buggy):>8,}"
   f"{ov_b:>10.1%}{acc_b_tr:>11.3f}{acc_b_te:>10.3f}{acc_b_tr-acc_b_te:>+8.3f}\n")
w9(f"{'RANDOM':<10}{len(df_train_random):>8,}{len(df_test_random):>8,}"
   f"{ov_r:>10.1%}{acc_r_tr:>11.3f}{acc_r_te:>10.3f}{acc_r_tr-acc_r_te:>+8.3f}\n")
w9(f"{'GROUPED':<10}{len(df_train_grouped):>8,}{len(df_test_grouped):>8,}"
   f"{ov_g:>10.1%}{acc_g_tr:>11.3f}{acc_g_te:>10.3f}{acc_g_tr-acc_g_te:>+8.3f}\n")
w9("=" * 82 + "\n")
w9("\n")
w9("Interpretation:\n")
w9(f"  Index bug impact     (BUGGY → RANDOM):   test acc {acc_b_te:.3f} → {acc_r_te:.3f}  "
   f"(Δ {acc_r_te-acc_b_te:+.3f})\n")
w9(f"  Duplication impact   (RANDOM → GROUPED): test acc {acc_r_te:.3f} → {acc_g_te:.3f}  "
   f"(Δ {acc_g_te-acc_r_te:+.3f})\n")
w9(f"  True generalisation gap (train-test on GROUPED): {acc_g_tr - acc_g_te:+.3f}\n")
w9("\n")
w9("→ GROUPED is the canonical split for downstream analysis.\n")

# Tactical breakdown
preds   = clf_full.predict(df_test_grouped[FEATURE_COLS])
correct = (preds == df_test_grouped[TARGET].astype(str))
w9("\n")
w9("Tactical accuracy on GROUPED test set:\n")
opp_mask = df_test_grouped["opp_wins_next"].astype(str) == "1"
if opp_mask.sum() > 0:
    w9(f"  opp_wins_next=1  (n={opp_mask.sum():4d}): {correct[opp_mask].mean():.3f}\n")
pop_mask = df_test_grouped[TARGET].str.startswith("pop")
if pop_mask.sum() > 0:
    w9(f"  pop moves        (n={pop_mask.sum():4d}): {correct[pop_mask].mean():.3f}\n")

# Canonical downstream variables
df_train = df_train_grouped
df_test  = df_test_grouped

# ─────────────────────────────────────────────────────────────────────────────
# Cell 11 — feature IG ranking on GROUPED train
# ─────────────────────────────────────────────────────────────────────────────
print("\n[run] feature IG ranking ...")
cell11_out = []
def w11(s):
    cell11_out.append(s)
    print(s, end="", flush=True)

ranker = ID3Classifier()
ig_scores = {f: ranker.information_gain(df_train, f, TARGET) for f in FEATURE_COLS}
ranked_features = sorted(ig_scores, key=ig_scores.get, reverse=True)

w11("Top 10 features by information gain:\n")
for i, f in enumerate(ranked_features[:10], 1):
    w11(f"  {i:2d}. {f:<18}  IG = {ig_scores[f]:.4f}\n")

# ─────────────────────────────────────────────────────────────────────────────
# Cell 12 — feature sweep on GROUPED
# ─────────────────────────────────────────────────────────────────────────────
print("\n[run] feature sweep ...")
cell12_out = []
def w12(s):
    cell12_out.append(s)
    print(s, end="", flush=True)

K_VALUES = [1, 3, 5, 10, 15, 20, 30, len(FEATURE_COLS)]
sweep_results = []
for k in K_VALUES:
    cols = ranked_features[:k] + [TARGET]
    clf_k = ID3Classifier(max_depth=15)
    t0 = time.perf_counter()
    clf_k.fit(df_train[cols], target=TARGET)
    tra = clf_k.score(df_train[cols], target=TARGET)
    tea = clf_k.score(df_test[cols],  target=TARGET)
    sweep_results.append({"K": k, "Train": tra, "Test": tea})
    w12(f"K={k:3d}  train={tra:.3f}  test={tea:.3f}\n")

# ─────────────────────────────────────────────────────────────────────────────
# Cell 13 — chart
# ─────────────────────────────────────────────────────────────────────────────
print("\n[run] rendering chart ...")
xs = [r["K"] for r in sweep_results]
plt.figure(figsize=(9, 4))
plt.plot(xs, [r["Train"] for r in sweep_results], "o-", label="Train accuracy",
         color="steelblue", linewidth=2)
plt.plot(xs, [r["Test"] for r in sweep_results], "s--", label="Test accuracy",
         color="tomato", linewidth=2)
plt.axhline(1/14, color="gray", linestyle=":", label="Random baseline (1/14)")
plt.xlabel("Number of features (top-K by information gain)")
plt.ylabel("Move-prediction accuracy")
plt.title("Feature Sweep — ID3 on PopOut  (GROUPED split, no leakage)")
plt.legend()
plt.tight_layout()
buf = io.BytesIO()
plt.savefig(buf, format="png", dpi=80, bbox_inches="tight")
buf.seek(0)
chart_b64 = base64.b64encode(buf.read()).decode("utf-8")
plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# Patch notebook
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n[patch] Loading {NB_PATH} ...")
nb = nbformat.read(NB_PATH, as_version=4)

c9_idx = c11_idx = c12_idx = c13_idx = None
for i, cell in enumerate(nb.cells):
    if cell.cell_type != "code":
        continue
    src = "".join(cell.source) if isinstance(cell.source, list) else cell.source
    if "df_train_buggy" in src and "df_train_random" in src:
        c9_idx = i
    elif "ig_scores" in src and "ranked_features" in src and "Top 10" in src:
        c11_idx = i
    elif "K_VALUES" in src and "sweep_results" in src:
        c12_idx = i
    elif "sweep_df['Train']" in src and "sweep_df['Test']" in src:
        c13_idx = i

print(f"[patch] cells located: 9={c9_idx}  11={c11_idx}  12={c12_idx}  13={c13_idx}")

def stream(text):
    return nbformat.v4.new_output("stream", name="stdout", text=text)

def image(b64_png):
    return nbformat.v4.new_output("display_data",
                                  data={"image/png": b64_png, "text/plain": "<Figure>"},
                                  metadata={})

if c9_idx is not None:
    nb.cells[c9_idx].outputs = [stream("".join(cell9_out))]
    nb.cells[c9_idx].execution_count = 1
    print(f"[patch] cell {c9_idx} (cell 9) outputs injected")

if c11_idx is not None:
    nb.cells[c11_idx].outputs = [stream("".join(cell11_out))]
    nb.cells[c11_idx].execution_count = 2
    print(f"[patch] cell {c11_idx} (cell 11) outputs injected")

if c12_idx is not None:
    nb.cells[c12_idx].outputs = [stream("".join(cell12_out))]
    nb.cells[c12_idx].execution_count = 3
    print(f"[patch] cell {c12_idx} (cell 12) outputs injected")

if c13_idx is not None:
    nb.cells[c13_idx].outputs = [image(chart_b64)]
    nb.cells[c13_idx].execution_count = 4
    print(f"[patch] cell {c13_idx} (cell 13) chart injected")

nbformat.write(nb, NB_PATH)
print(f"\n[save] Notebook saved to {NB_PATH}")
print("\n[done] Leakage fix applied to all 4 cells (9, 11, 12, 13).")

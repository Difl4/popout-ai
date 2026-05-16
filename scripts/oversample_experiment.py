"""Test OVERSAMPLE_FACTOR ∈ {4, 8, 16} on the GROUPED split.

Reports for each: overall test acc + tactical accuracy (opp_wins_next, pop moves).
Helps decide whether to switch from 4 to a larger oversample.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.decision_tree.id3.learner import ID3Classifier

# ── Setup ─────────────────────────────────────────────────────────────────────
DATA_PATH = ROOT / "data/generated/popout_dt_dataset.csv"
MAX_DEPTH = 15
TARGET    = "best_move"

df = pd.read_csv(DATA_PATH)
FEATURE_COLS = [c for c in df.columns if c not in (TARGET, "is_proven")]

df_str = df.copy()
for col in FEATURE_COLS:
    df_str[col] = df_str[col].astype(str)
df_str_train = df_str[df_str["can_win"].astype(str) != "1"].reset_index(drop=True)

# Same GROUPED split as canonical
_rng = np.random.default_rng(seed=42)
_unique = sorted(set(map(tuple, df_str_train[FEATURE_COLS].values)))
_rng.shuffle(_unique)
_n_train_states = int(0.8 * len(_unique))
_train_state_set = set(_unique[:_n_train_states])
_sig = df_str_train[FEATURE_COLS].apply(tuple, axis=1)
df_train = df_str_train[_sig.isin(_train_state_set)].reset_index(drop=True)
df_test  = df_str_train[~_sig.isin(_train_state_set)].reset_index(drop=True)

print(f"GROUPED split: {len(df_train):,} train  /  {len(df_test):,} test")
print()

# ── Run trials ────────────────────────────────────────────────────────────────
results = []
for OVERSAMPLE in [4, 8, 16]:
    print(f"  [OVERSAMPLE={OVERSAMPLE:2d}] ", end="", flush=True)
    proven = df_train[df_train["is_proven"].astype(str) == "1"]
    d_aug = pd.concat([df_train] + [proven] * OVERSAMPLE, ignore_index=True)
    print(f"aug-rows={len(d_aug):>7,}  ", end="", flush=True)

    clf = ID3Classifier(max_depth=MAX_DEPTH)
    t0 = time.perf_counter()
    clf.fit(d_aug[FEATURE_COLS + [TARGET]], target=TARGET)
    dt = time.perf_counter() - t0

    train_acc = clf.score(df_train[FEATURE_COLS + [TARGET]], target=TARGET)
    test_acc  = clf.score(df_test[FEATURE_COLS + [TARGET]], target=TARGET)

    preds = clf.predict(df_test[FEATURE_COLS])
    correct = (preds == df_test[TARGET].astype(str))
    opp_mask = df_test["opp_wins_next"].astype(str) == "1"
    pop_mask = df_test[TARGET].str.startswith("pop")
    opp_acc = correct[opp_mask].mean() if opp_mask.sum() > 0 else float("nan")
    pop_acc = correct[pop_mask].mean() if pop_mask.sum() > 0 else float("nan")

    results.append({
        "oversample": OVERSAMPLE,
        "train_acc":  train_acc,
        "test_acc":   test_acc,
        "opp_acc":    opp_acc,
        "pop_acc":    pop_acc,
        "time":       dt,
    })
    print(f"train={train_acc:.3f}  test={test_acc:.3f}  "
          f"opp_wins={opp_acc:.3f}  pop={pop_acc:.3f}  ({dt:.1f}s)")

# ── Summary table ─────────────────────────────────────────────────────────────
print()
print("=" * 78)
print(f"{'OVERSAMPLE':>11}{'train':>9}{'test':>9}{'opp_wins':>11}{'pop':>9}{'time':>9}")
print("-" * 78)
for r in results:
    print(f"{r['oversample']:>11d}{r['train_acc']:>9.3f}{r['test_acc']:>9.3f}"
          f"{r['opp_acc']:>11.3f}{r['pop_acc']:>9.3f}{r['time']:>8.1f}s")
print("=" * 78)
print()
print("Decision criteria:")
print("  - If higher OVERSAMPLE improves opp_wins / pop accuracy by >3pp without")
print("    hurting overall test acc → switch")
print("  - If improvements are <2pp → keep 4 (Occam's razor)")

"""
Compara ID3 depth=10 (actual) vs depth=15 (novo).

Métricas:
  1. Train / test accuracy (80/20 split)
  2. Win-rate vs FlatNumbaMCTS e FlatNumbaSolverMCTS (100 jogos por config)

Resultados depth=10 são os do notebook (hardcoded para comparação directa).
"""

from __future__ import annotations

import gc
import math
import os
import sys
import time
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.decision_tree.id3.learner import ID3Classifier
from src.decision_tree.id3_agent import ID3Agent
from src.decision_tree.id3_agent_raw import ID3AgentRaw
from src.engine.standard.bitboard import PopOutBoard
from src.engine.standard.rules import evaluate_after_move

# ── Configuração ──────────────────────────────────────────────────────────────

DATA_CSV       = ROOT / "data/generated/popout_dt_dataset.csv"
PICKLE_D15     = ROOT / "data/generated/v3_5000games_tiered/id3_model_d15.pkl"
PICKLE_RAW_D15 = ROOT / "data/generated/v3_5000games_tiered/id3_model_raw_d15.pkl"

TARGET         = "best_move"
OVERSAMPLE_FACTOR = 4
RAW_OVERSAMPLE    = 10
EVAL_GAMES        = 100
NEW_DEPTH         = 15
RAW_DEPTH_NEW     = 18     # raw já estava a 18, mantém-se

_EXCLUDE_RAW = {"best_move", "is_proven",
                "threats_me", "threats_opp",
                "center_me", "center_opp",
                "phase", "can_win", "opp_wins_next"}

# ── Resultados depth=10 do notebook (referência) ──────────────────────────────

DEPTH10_ACCURACY = {
    "train": 0.877,
    "test":  0.839,
}

# (model, engine, iters) → win_rate
DEPTH10_WINRATES: dict[tuple[str, str, int], float] = {
    ("ID3 (features)", "FlatNumbaMCTS",       100):     0.89,
    ("ID3 (features)", "FlatNumbaMCTS",       500):     0.69,
    ("ID3 (features)", "FlatNumbaMCTS",     1_000):     0.50,
    ("ID3 (features)", "FlatNumbaMCTS",     5_000):     0.37,
    ("ID3 (features)", "FlatNumbaMCTS",    10_000):     0.27,
    ("ID3 (features)", "FlatNumbaSolverMCTS",  1_000):  0.46,
    ("ID3 (features)", "FlatNumbaSolverMCTS",  5_000):  0.32,
    ("ID3 (features)", "FlatNumbaSolverMCTS", 10_000):  0.35,
    ("ID3 (features)", "FlatNumbaSolverMCTS", 50_000):  0.54,
    ("ID3 (features)", "FlatNumbaSolverMCTS",100_000):  0.51,
    ("ID3 Raw",        "FlatNumbaMCTS",       100):     0.94,
    ("ID3 Raw",        "FlatNumbaMCTS",       500):     0.66,
    ("ID3 Raw",        "FlatNumbaMCTS",     1_000):     0.62,
    ("ID3 Raw",        "FlatNumbaMCTS",     5_000):     0.51,
    ("ID3 Raw",        "FlatNumbaMCTS",    10_000):     0.46,
    ("ID3 Raw",        "FlatNumbaSolverMCTS",  1_000):  0.63,
    ("ID3 Raw",        "FlatNumbaSolverMCTS",  5_000):  0.61,
    ("ID3 Raw",        "FlatNumbaSolverMCTS", 10_000):  0.67,
    ("ID3 Raw",        "FlatNumbaSolverMCTS", 50_000):  0.54,
    ("ID3 Raw",        "FlatNumbaSolverMCTS",100_000):  0.54,
}

CONFIGS = {
    "FlatNumbaMCTS":       [100, 500, 1_000, 5_000, 10_000],
    "FlatNumbaSolverMCTS": [1_000, 5_000, 10_000, 50_000, 100_000],
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def _filter_can_win(df: pd.DataFrame) -> pd.DataFrame:
    if "can_win" not in df.columns:
        return df
    return df[df["can_win"] != 1].reset_index(drop=True)


def _make_agent(clf, raw: bool = False):
    agent = ID3AgentRaw() if raw else ID3Agent()
    agent._classifier = clf
    return agent


def play_one_game(tree_agent, mcts_agent, tree_plays_as: int,
                  mcts_iterations: int, max_moves: int = 120) -> int:
    board = PopOutBoard()
    for _ in range(max_moves):
        legal = board.legal_moves()
        if not legal:
            return 0
        mover = board.current_player
        move = (tree_agent.run(board)
                if mover == tree_plays_as
                else mcts_agent.run(board, iterations=mcts_iterations))
        board.apply_move(move)
        winner = evaluate_after_move(board, mover=mover)
        if winner != 0:
            return 1 if winner == tree_plays_as else -1
    return 0


# ── 1. Carregar dataset ───────────────────────────────────────────────────────

print("=" * 70)
print(f"Carregando dataset: {DATA_CSV}")
df = pd.read_csv(DATA_CSV)
print(f"  {len(df):,} linhas × {df.shape[1]} colunas")

FEATURE_COLS = [c for c in df.columns if c not in (TARGET, "is_proven")]

# ── 2. Treinar clf_full a depth=15 (split 80/20) ─────────────────────────────

print(f"\n{'='*70}")
print(f"TREINO  —  depth={NEW_DEPTH}  (split 80/20, oversampling proven × {OVERSAMPLE_FACTOR})")
print("=" * 70)

df_str = df.copy()
for col in FEATURE_COLS:
    df_str[col] = df_str[col].astype(str)

df_str_train = df_str[df_str["can_win"].astype(str) != "1"].reset_index(drop=True)
print(f"  can_win=1 filtrados: {len(df_str) - len(df_str_train):,}")

df_train = df_str_train.sample(frac=0.8, random_state=42).reset_index(drop=True)
df_test  = df_str_train.drop(df_train.index).reset_index(drop=True)

proven      = df_train[df_train["is_proven"].astype(str) == "1"]
df_train_aug = pd.concat([df_train] + [proven] * OVERSAMPLE_FACTOR, ignore_index=True)

print(f"  Train (aug): {len(df_train_aug):,} linhas  |  Test: {len(df_test):,} linhas")

clf_full = ID3Classifier(max_depth=NEW_DEPTH)
t0 = time.perf_counter()
clf_full.fit(df_train_aug[FEATURE_COLS + [TARGET]], target=TARGET)
train_time = time.perf_counter() - t0

train_acc_d15 = clf_full.score(df_train[FEATURE_COLS + [TARGET]], target=TARGET)
test_acc_d15  = clf_full.score(df_test[FEATURE_COLS  + [TARGET]], target=TARGET)

print(f"\n  Tempo de treino : {train_time:.1f}s")
print(f"  Train accuracy  : {train_acc_d15:.3f}  (depth=10: {DEPTH10_ACCURACY['train']:.3f})  "
      f"Δ={train_acc_d15 - DEPTH10_ACCURACY['train']:+.3f}")
print(f"  Test  accuracy  : {test_acc_d15:.3f}  (depth=10: {DEPTH10_ACCURACY['test']:.3f})  "
      f"Δ={test_acc_d15 - DEPTH10_ACCURACY['test']:+.3f}")

# Tactical breakdown
preds   = clf_full.predict(df_test[FEATURE_COLS])
correct = (preds == df_test[TARGET].astype(str))
opp_mask = df_test["opp_wins_next"].astype(str) == "1"
pop_mask  = df_test[TARGET].str.startswith("pop")
print(f"\n  Tactical accuracy (depth=15):")
if opp_mask.sum() > 0:
    print(f"    opp_wins_next=1  (n={opp_mask.sum():4d}): {correct[opp_mask].mean():.3f}  "
          f"(depth=10: 0.762)  Δ={correct[opp_mask].mean()-0.762:+.3f}")
if pop_mask.sum() > 0:
    print(f"    pop moves        (n={pop_mask.sum():4d}): {correct[pop_mask].mean():.3f}  "
          f"(depth=10: 0.707)  Δ={correct[pop_mask].mean()-0.707:+.3f}")

del df_train_aug, df_str_train, df_str, df_train, df_test, proven
gc.collect()

# ── 3. Treinar clf_game e clf_game_raw a depth=15 ────────────────────────────

print(f"\n{'='*70}")
print(f"TREINO clf_game (features, depth={NEW_DEPTH}) para avaliação win-rate")
print("=" * 70)

if PICKLE_D15.exists():
    print(f"  Carregando pickle existente: {PICKLE_D15}")
    with open(PICKLE_D15, "rb") as f:
        clf_game = pickle.load(f)
else:
    _df = _filter_can_win(df.copy())
    _feat_cols = [c for c in _df.columns if c not in ("best_move", "is_proven")]
    _proven = _df[_df["is_proven"] == 1]
    _df = pd.concat([_df] + [_proven] * OVERSAMPLE_FACTOR, ignore_index=True)
    del _proven
    for c in _feat_cols:
        _df[c] = _df[c].astype(str)
    _df["best_move"] = _df["best_move"].astype(str)
    print(f"  Linhas após oversampling: {len(_df):,}")
    t0 = time.perf_counter()
    clf_game = ID3Classifier(max_depth=NEW_DEPTH)
    clf_game.fit(_df[_feat_cols + ["best_move"]], target="best_move")
    print(f"  Treino em {time.perf_counter()-t0:.1f}s")
    del _df; gc.collect()
    PICKLE_D15.parent.mkdir(parents=True, exist_ok=True)
    with open(PICKLE_D15, "wb") as f:
        pickle.dump(clf_game, f)
    print(f"  Guardado em {PICKLE_D15}")

print(f"\n{'='*70}")
print(f"TREINO clf_game_raw (raw cells, depth={RAW_DEPTH_NEW}) para avaliação win-rate")
print("=" * 70)

if PICKLE_RAW_D15.exists():
    print(f"  Carregando pickle existente: {PICKLE_RAW_D15}")
    with open(PICKLE_RAW_D15, "rb") as f:
        clf_game_raw = pickle.load(f)
else:
    _df_raw = _filter_can_win(df.copy())
    _raw_cols = [c for c in _df_raw.columns if c not in _EXCLUDE_RAW]
    _proven = _df_raw[_df_raw["is_proven"] == 1]
    _df_raw = pd.concat([_df_raw] + [_proven] * RAW_OVERSAMPLE, ignore_index=True)
    del _proven
    for c in _raw_cols:
        _df_raw[c] = _df_raw[c].astype(str)
    _df_raw["best_move"] = _df_raw["best_move"].astype(str)
    print(f"  Linhas após oversampling: {len(_df_raw):,}")
    t0 = time.perf_counter()
    clf_game_raw = ID3Classifier(max_depth=RAW_DEPTH_NEW)
    clf_game_raw.fit(_df_raw[_raw_cols + ["best_move"]], target="best_move")
    print(f"  Treino em {time.perf_counter()-t0:.1f}s")
    del _df_raw; gc.collect()
    with open(PICKLE_RAW_D15, "wb") as f:
        pickle.dump(clf_game_raw, f)
    print(f"  Guardado em {PICKLE_RAW_D15}")

# ── 4. Warmup Numba ───────────────────────────────────────────────────────────

print(f"\n{'='*70}")
print("WARMUP Numba JIT ...")
print("=" * 70)
from src.mcts.optimized.numba_mcts import warmup, FlatNumbaMCTS
from src.mcts.optimized.numba_solver import FlatNumbaSolverMCTS, warmup_solver
t0 = time.perf_counter()
warmup()
warmup_solver()
print(f"  Warmup concluído em {time.perf_counter()-t0:.1f}s")

# ── 5. Avaliação win-rate ─────────────────────────────────────────────────────

print(f"\n{'='*70}")
print(f"AVALIAÇÃO  —  {EVAL_GAMES} jogos por configuração")
print("=" * 70)

TREE_MODELS = {
    "ID3 (features)": _make_agent(clf_game,     raw=False),
    "ID3 Raw":        _make_agent(clf_game_raw, raw=True),
}

eval_results: dict[tuple[str, str, int], tuple[int, int, int]] = {}

t_eval = time.perf_counter()
for model_name, tree_agent in TREE_MODELS.items():
    print(f"\n  ── {model_name} ──")
    for engine_name, iter_list in CONFIGS.items():
        for iters in iter_list:
            mcts_agent = (FlatNumbaMCTS(max_nodes=iters + 256)
                          if engine_name == "FlatNumbaMCTS"
                          else FlatNumbaSolverMCTS(max_nodes=iters + 256))
            wins = losses = draws = 0
            for i in range(EVAL_GAMES):
                r = play_one_game(tree_agent, mcts_agent,
                                  1 if i % 2 == 0 else 2, iters)
                if r == 1:
                    wins += 1
                elif r == -1:
                    losses += 1
                else:
                    draws += 1
            eval_results[(model_name, engine_name, iters)] = (wins, losses, draws)

            wr_d15 = wins / EVAL_GAMES
            wr_d10 = DEPTH10_WINRATES.get((model_name, engine_name, iters), float("nan"))
            delta  = wr_d15 - wr_d10 if not math.isnan(wr_d10) else float("nan")
            sign   = "+" if delta >= 0 else ""
            print(f"    {engine_name:22s}  {iters:7,d} iter  "
                  f"d15={wr_d15:5.1%}  d10={wr_d10:5.1%}  Δ={sign}{delta:.1%}")
        print()

print(f"\n  Tempo total de avaliação: {time.perf_counter()-t_eval:.1f}s")

# ── 6. Tabela resumo final ────────────────────────────────────────────────────

print(f"\n{'='*70}")
print("RESUMO COMPARATIVO  —  depth=10 vs depth=15")
print("=" * 70)

print(f"\n  ── Accuracy (split 80/20) ──")
print(f"  {'Métrica':<20}  {'depth=10':>9}  {'depth=15':>9}  {'Δ':>8}")
print(f"  {'-'*50}")
for label, d10, d15 in [
    ("Train accuracy",     DEPTH10_ACCURACY['train'], train_acc_d15),
    ("Test  accuracy",     DEPTH10_ACCURACY['test'],  test_acc_d15),
    ("opp_wins_next=1",    0.762,  correct[opp_mask].mean() if opp_mask.sum() > 0 else float("nan")),
    ("Pop moves",          0.707,  correct[pop_mask].mean() if pop_mask.sum() > 0 else float("nan")),
]:
    delta = d15 - d10
    sign  = "+" if delta >= 0 else ""
    print(f"  {label:<20}  {d10:>9.1%}  {d15:>9.1%}  {sign}{delta:.1%}")

print(f"\n  ── Win-rate médio por modelo ──")
print(f"  {'Modelo':<20}  {'depth=10':>9}  {'depth=15':>9}  {'Δ':>8}")
print(f"  {'-'*50}")
for model_name in TREE_MODELS:
    keys_d10 = [(model_name, eng, it) for eng, lst in CONFIGS.items() for it in lst]
    avg_d10 = np.mean([DEPTH10_WINRATES[k] for k in keys_d10 if k in DEPTH10_WINRATES])
    avg_d15 = np.mean([eval_results[k][0] / EVAL_GAMES for k in keys_d10 if k in eval_results])
    delta   = avg_d15 - avg_d10
    sign    = "+" if delta >= 0 else ""
    print(f"  {model_name:<20}  {avg_d10:>9.1%}  {avg_d15:>9.1%}  {sign}{delta:.1%}")

print(f"\n  ── Win-rate vs FlatNumbaSolverMCTS (100k iters) — o adversário mais forte ──")
print(f"  {'Modelo':<20}  {'depth=10':>9}  {'depth=15':>9}  {'Δ':>8}")
print(f"  {'-'*50}")
for model_name in TREE_MODELS:
    k = (model_name, "FlatNumbaSolverMCTS", 100_000)
    d10 = DEPTH10_WINRATES.get(k, float("nan"))
    d15 = eval_results[k][0] / EVAL_GAMES if k in eval_results else float("nan")
    delta = d15 - d10
    sign  = "+" if delta >= 0 else ""
    print(f"  {model_name:<20}  {d10:>9.1%}  {d15:>9.1%}  {sign}{delta:.1%}")

print(f"\n{'='*70}")
print("Concluído.")

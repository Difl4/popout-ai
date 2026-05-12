"""Pre-train the raw-features ID3 agent and save to data/generated/id3_model_raw.pkl.

    python scripts/train_id3_raw.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.decision_tree.id3_agent_raw import ID3AgentRaw

if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parents[1]
    pickle_path  = ROOT / "data/generated/id3_model_raw.pkl"
    dataset_path = ROOT / "data/generated/popout_dt_dataset.csv"

    if pickle_path.exists():
        print(f"Apagando modelo antigo em {pickle_path} para retreinar...")
        pickle_path.unlink()

    print("A treinar ID3 Raw (features brutas, depth=20, oversampling 10x proven)...")
    print("(pode demorar bastante — paralelismo ativo)\n")

    t0 = time.perf_counter()
    agent = ID3AgentRaw(
        dataset_path=str(dataset_path),
        pickle_path=str(pickle_path),
    )
    agent._ensure_trained()
    elapsed = time.perf_counter() - t0

    print(f"\nPronto! Modelo guardado em {pickle_path} ({elapsed:.1f}s)")

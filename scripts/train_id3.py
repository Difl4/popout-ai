"""Pre-train the ID3 agent and save the model to data/generated/id3_model.pkl.

Run once before launching the GUI with the ID3 difficulty:

    python scripts/train_id3.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.decision_tree.id3_agent import ID3Agent

if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parents[1]
    pickle_path = ROOT / "data/generated/id3_model.pkl"
    dataset_path = ROOT / "data/generated/popout_dt_dataset.csv"

    if pickle_path.exists():
        print(f"Apagando modelo antigo em {pickle_path} para retreinar...")
        pickle_path.unlink()

    print("A treinar o modelo ID3 com o dataset do notebook...")
    print("(isto pode demorar alguns minutos)\n")

    t0 = time.perf_counter()
    agent = ID3Agent(
        dataset_path=str(dataset_path),
        pickle_path=str(pickle_path),
    )
    agent._ensure_trained()
    elapsed = time.perf_counter() - t0

    print(f"\nPronto! Modelo guardado em {pickle_path} ({elapsed:.1f}s)")
    print("Pode agora iniciar o jogo normalmente — a IA ID3 carregará instantaneamente.")

"""Render the current Iris ID3 tree as a PNG figure.

Output: data/figures/iris_id3_tree.png — overwrites the existing stale image.
Uses the same colour palette as the standalone slides for visual continuity.

Run:
    python scripts/render_iris_tree.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from src.decision_tree.id3.learner import ID3Classifier, DecisionNode
from src.decision_tree.discretizer import fit_quantile_bins, apply_bins

# ─── Palette (Ocean Gradient + class colours) ─────────────────────────────────
NAVY     = "#21295C"
DEEP     = "#065A82"
TEAL     = "#1C7293"
ICE      = "#E6F1F5"
DARK     = "#121E3F"
GRAY     = "#6B7A88"
WHITE    = "#FFFFFF"

CLASS_COLOURS = {
    "Iris-setosa":     "#2E8B57",   # forest
    "Iris-versicolor": "#1C7293",   # teal
    "Iris-virginica":  "#C8597A",   # rose
}


# ─── Tree layout: assign x-coordinate to every node ───────────────────────────
def assign_positions(node, depth, x_counter, positions):
    """Place leaves in left-to-right order; internal nodes are centred over
    their children."""
    if node.is_leaf():
        x = x_counter[0]
        x_counter[0] += 1.0
        positions[id(node)] = (x, -depth)
        return x
    child_xs = []
    for _, child in sorted(node.children.items()):
        cx = assign_positions(child, depth + 1, x_counter, positions)
        child_xs.append(cx)
    x = sum(child_xs) / len(child_xs)
    positions[id(node)] = (x, -depth)
    return x


def collect_edges(node, edges):
    """Yield (parent_id, child_id, branch_value) tuples."""
    if node.is_leaf():
        return
    for value, child in sorted(node.children.items()):
        edges.append((id(node), id(child), value))
        collect_edges(child, edges)


def draw_node(ax, x, y, text, *, fill, text_colour=WHITE,
              width=1.50, height=0.55, fontsize=11, bold=True):
    box = FancyBboxPatch((x - width / 2, y - height / 2),
                         width, height,
                         boxstyle="round,pad=0.02,rounding_size=0.10",
                         linewidth=0.0, facecolor=fill,
                         edgecolor="none", zorder=2)
    ax.add_patch(box)
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize,
            fontweight=("bold" if bold else "normal"),
            color=text_colour, zorder=3)


def draw_edge(ax, p1, p2, label, *, colour=GRAY):
    x1, y1 = p1
    x2, y2 = p2
    # straight line
    ax.plot([x1, x2], [y1 - 0.30, y2 + 0.30], color=colour, linewidth=1.2,
            zorder=1)
    # label at midpoint, slightly above the line
    mx = (x1 + x2) / 2
    my = (y1 + y2) / 2
    # background rectangle so the label is readable over the lines
    ax.text(mx, my, label, ha="center", va="center", fontsize=8.5,
            color=DARK, fontstyle="italic",
            bbox=dict(boxstyle="round,pad=0.15", facecolor=ICE,
                      edgecolor="none", alpha=0.95),
            zorder=2)


def render(node, save_path):
    positions: dict[int, tuple[float, float]] = {}
    assign_positions(node, depth=0, x_counter=[0.0], positions=positions)
    edges: list[tuple[int, int, str]] = []
    collect_edges(node, edges)

    # Figure proportional to the number of leaves
    n_leaves = max(p[0] for p in positions.values()) + 1
    depth_max = -min(p[1] for p in positions.values())
    fig_w = max(11, n_leaves * 1.05)
    fig_h = max(5.5, (depth_max + 1) * 1.20)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=140)

    # Edges first (drawn under nodes)
    id_to_node: dict[int, DecisionNode] = {}
    def _index(n):
        id_to_node[id(n)] = n
        if not n.is_leaf():
            for c in n.children.values():
                _index(c)
    _index(node)

    for parent_id, child_id, value in edges:
        draw_edge(ax, positions[parent_id], positions[child_id], value)

    # Nodes
    for node_id, (x, y) in positions.items():
        n = id_to_node[node_id]
        if n.is_leaf():
            label = str(n.label).replace("Iris-", "")
            fill = CLASS_COLOURS.get(str(n.label), TEAL)
            draw_node(ax, x, y, label, fill=fill, width=1.40, height=0.55,
                      fontsize=11)
        else:
            label = str(n.feature)
            draw_node(ax, x, y, label, fill=DEEP, width=1.60, height=0.55,
                      fontsize=11)

    # Title
    ax.set_title("Árvore ID3 — Dataset Iris  (max_depth = 10, q = 3 bins)",
                 fontsize=14, fontweight="bold", color=NAVY, pad=12)

    # Legend
    legend_y = -(depth_max + 0.7)
    legend_x_start = 0.5
    items = [("Setosa", CLASS_COLOURS["Iris-setosa"]),
             ("Versicolor", CLASS_COLOURS["Iris-versicolor"]),
             ("Virginica", CLASS_COLOURS["Iris-virginica"]),
             ("Split node", DEEP)]
    for i, (name, col) in enumerate(items):
        lx = legend_x_start + i * 2.2
        draw_node(ax, lx, legend_y, name, fill=col, width=1.40, height=0.40,
                  fontsize=10)

    ax.set_xlim(-1.0, n_leaves)
    ax.set_ylim(legend_y - 0.6, 0.8)
    ax.axis("off")
    plt.tight_layout()

    plt.savefig(save_path, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {save_path}")


def main():
    df = pd.read_csv(ROOT / "notebooks" / "iris.csv").drop(columns=["ID"])
    FEATURES = ["sepallength", "sepalwidth", "petallength", "petalwidth"]
    TARGET   = "class"

    bins   = fit_quantile_bins(df, FEATURES, q=3)
    df_d   = apply_bins(df, bins)
    clf    = ID3Classifier(max_depth=10)
    clf.fit(df_d, target=TARGET)

    out = ROOT / "data/figures/iris_id3_tree.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    render(clf.root, str(out))


if __name__ == "__main__":
    main()

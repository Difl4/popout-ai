"""Build the PopOut AI defence presentation (docs/presentation.pptx).

Produces a 10-slide deck for the AI 2025/2026 oral defence (10-minute budget).
Reuses existing benchmark figures from data/figures/. Re-run after editing.

Usage:
    python scripts/build_presentation.py
"""
from __future__ import annotations

from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Inches, Pt, Emu

# ─── Paths ────────────────────────────────────────────────────────────────────
ROOT     = Path(__file__).resolve().parents[1]
FIG_DIR  = ROOT / "data/figures"
OUT_PATH = ROOT / "docs/presentation.pptx"

# ─── Palette ("Ocean Gradient" — strategic / technical feel) ──────────────────
NAVY   = RGBColor(0x21, 0x29, 0x5C)   # primary dark
DEEP   = RGBColor(0x06, 0x5A, 0x82)   # primary mid
TEAL   = RGBColor(0x1C, 0x72, 0x93)   # accent
ICE    = RGBColor(0xE6, 0xF1, 0xF5)   # subtle bg fill
WHITE  = RGBColor(0xFF, 0xFF, 0xFF)
GRAY   = RGBColor(0x55, 0x6B, 0x78)   # muted text
DARK   = RGBColor(0x12, 0x1E, 0x3F)   # near-black
CORAL  = RGBColor(0xE5, 0x71, 0x4E)   # warm accent (for warnings / red flags)
GREEN  = RGBColor(0x2E, 0x8B, 0x57)   # success accent

HEADER_FONT = "Calibri"
BODY_FONT   = "Calibri"


# ─── Helpers ──────────────────────────────────────────────────────────────────
def set_slide_bg(slide, color: RGBColor) -> None:
    bg = slide.background
    fill = bg.fill
    fill.solid()
    fill.fore_color.rgb = color


def add_text(slide, x, y, w, h, text, *, size=16, bold=False, color=DARK,
             align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, font=BODY_FONT,
             italic=False) -> None:
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = Inches(0.05)
    tf.margin_top = tf.margin_bottom = Inches(0.02)
    tf.vertical_anchor = anchor
    p = tf.paragraphs[0]
    p.alignment = align
    run = p.add_run()
    run.text = text
    f = run.font
    f.name = font
    f.size = Pt(size)
    f.bold = bold
    f.italic = italic
    f.color.rgb = color


def add_bullets(slide, x, y, w, h, items, *, size=15, color=DARK,
                bullet_color=TEAL, indent=0.0) -> None:
    """Add bulleted text with custom round-disc bullet on the left."""
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = Inches(0.05)
    tf.margin_top = Inches(0.0)
    for i, item in enumerate(items):
        if i == 0:
            p = tf.paragraphs[0]
        else:
            p = tf.add_paragraph()
        p.alignment = PP_ALIGN.LEFT
        p.space_after = Pt(8)
        # bullet character
        b = p.add_run()
        b.text = "•  "
        b.font.name = BODY_FONT
        b.font.size = Pt(size)
        b.font.color.rgb = bullet_color
        b.font.bold = True
        # body
        r = p.add_run()
        r.text = item
        r.font.name = BODY_FONT
        r.font.size = Pt(size)
        r.font.color.rgb = color


def add_box(slide, x, y, w, h, fill_color, line_color=None) -> None:
    shape = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE,
                                   Inches(x), Inches(y),
                                   Inches(w), Inches(h))
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill_color
    if line_color is None:
        shape.line.fill.background()
    else:
        shape.line.color.rgb = line_color
        shape.line.width = Pt(0.5)
    shape.shadow.inherit = False
    return shape


def add_card(slide, x, y, w, h, title, body_items, *, accent=TEAL,
             title_size=15, body_size=13) -> None:
    """White card with coloured top stripe + title + bullet body."""
    add_box(slide, x, y, w, h, WHITE)
    add_box(slide, x, y, w, 0.08, accent)
    add_text(slide, x + 0.15, y + 0.15, w - 0.3, 0.4,
             title, size=title_size, bold=True, color=NAVY)
    if isinstance(body_items, str):
        add_text(slide, x + 0.15, y + 0.55, w - 0.3, h - 0.65,
                 body_items, size=body_size, color=GRAY)
    else:
        add_bullets(slide, x + 0.15, y + 0.55, w - 0.3, h - 0.65,
                    body_items, size=body_size, color=GRAY, bullet_color=accent)


def slide_header(slide, title, *, chip=None) -> None:
    """Top-bar header with optional small chip (eg. section number)."""
    if chip:
        add_box(slide, 0.5, 0.35, 0.7, 0.32, TEAL)
        add_text(slide, 0.5, 0.36, 0.7, 0.32, chip,
                 size=12, bold=True, color=WHITE, align=PP_ALIGN.CENTER,
                 anchor=MSO_ANCHOR.MIDDLE)
        add_text(slide, 1.35, 0.30, 8.0, 0.45, title,
                 size=24, bold=True, color=NAVY)
    else:
        add_text(slide, 0.5, 0.30, 9.0, 0.45, title,
                 size=24, bold=True, color=NAVY)
    # subtle bottom rule
    add_box(slide, 0.5, 0.85, 9.0, 0.02, ICE)


def slide_footer(slide, page_no, total=10) -> None:
    add_text(slide, 0.5, 5.30, 4.0, 0.25,
             "PopOut AI — IA 2025/2026", size=9, color=GRAY)
    add_text(slide, 5.5, 5.30, 4.0, 0.25,
             f"{page_no} / {total}", size=9, color=GRAY,
             align=PP_ALIGN.RIGHT)


# ─── Build presentation ──────────────────────────────────────────────────────
def build() -> None:
    prs = Presentation()
    prs.slide_width  = Inches(10)
    prs.slide_height = Inches(5.625)
    blank = prs.slide_layouts[6]  # fully blank

    # ── Slide 1: TITLE ───────────────────────────────────────────────────────
    s = prs.slides.add_slide(blank)
    set_slide_bg(s, NAVY)

    # decorative side stripe
    add_box(s, 0.0, 0.0, 0.25, 5.625, TEAL)

    add_text(s, 1.0, 1.20, 8.0, 0.4,
             "INTELIGÊNCIA ARTIFICIAL 2025/2026",
             size=14, color=ICE, font=HEADER_FONT, bold=True)
    add_text(s, 1.0, 1.65, 8.0, 1.0,
             "PopOut AI",
             size=54, bold=True, color=WHITE, font=HEADER_FONT)
    add_text(s, 1.0, 2.55, 8.0, 0.5,
             "MCTS + ID3 para uma variante do Connect-4",
             size=20, color=ICE, italic=True, font=BODY_FONT)
    add_box(s, 1.0, 3.15, 5.0, 0.03, TEAL)
    add_text(s, 1.0, 3.30, 8.0, 0.3,
             "Grupo:", size=11, bold=True, color=ICE)
    add_text(s, 1.0, 3.60, 8.0, 0.3,
             "Duarte Meneses dos Santos Sousa Gomes  ·  202409386",
             size=13, color=WHITE)
    add_text(s, 1.0, 3.90, 8.0, 0.3,
             "José Paulo Pacheco de Sousa  ·  202405046",
             size=13, color=WHITE)
    add_text(s, 1.0, 4.20, 8.0, 0.3,
             "Tiago Braga da Cruz Frada de Sousa  ·  202405406",
             size=13, color=WHITE)
    add_text(s, 1.0, 4.85, 8.0, 0.3,
             "github.com/Difl4/popout-ai   ·   Maio 2026",
             size=11, color=ICE, italic=True)

    # ── Slide 2: THE PROBLEM ─────────────────────────────────────────────────
    s = prs.slides.add_slide(blank)
    set_slide_bg(s, WHITE)
    slide_header(s, "O Problema: PopOut", chip="01")

    # Left column — game description
    add_text(s, 0.5, 1.05, 5.5, 0.4,
             "Connect-4 + jogada POP",
             size=20, bold=True, color=DEEP)
    add_bullets(s, 0.5, 1.55, 5.5, 1.5, [
        "Tabuleiro 7×6, dois jogadores alternam.",
        "POP: remover peça própria do fundo — peças acima descem.",
        "Vencedor: o primeiro a alinhar 4 (horiz / vert / diag).",
    ], size=14)

    add_text(s, 0.5, 3.05, 5.5, 0.35,
             "Três regras especiais",
             size=15, bold=True, color=DEEP)
    add_bullets(s, 0.5, 3.40, 5.5, 1.4, [
        "Pop simultâneo: ganha quem fez o pop.",
        "Tabuleiro cheio: a vez de jogar pode declarar empate.",
        "Repetição tripla: qualquer jogador pode declarar empate.",
    ], size=13, color=GRAY)

    # Right column — required scenarios
    add_box(s, 6.4, 1.05, 3.1, 3.95, ICE)
    add_box(s, 6.4, 1.05, 0.08, 3.95, TEAL)
    add_text(s, 6.6, 1.20, 2.8, 0.35,
             "Cenários obrigatórios", size=14, bold=True, color=NAVY)
    add_bullets(s, 6.6, 1.55, 2.8, 2.6, [
        "Humano vs Humano",
        "Humano vs Computador",
        "Computador vs Computador",
    ], size=13, color=DARK)
    add_text(s, 6.6, 3.50, 2.8, 0.3,
             "Cobertura no projecto",
             size=12, bold=True, color=NAVY)
    add_text(s, 6.6, 3.78, 2.8, 1.2,
             "GUI Pygame (PvP, HvC, Arena) +\nCLI com torneio configurável.",
             size=11, color=GRAY)

    slide_footer(s, 2)

    # ── Slide 3: ARCHITECTURE + CONTRIBUTIONS ───────────────────────────────
    s = prs.slides.add_slide(blank)
    set_slide_bg(s, WHITE)
    slide_header(s, "Arquitectura & Contribuições", chip="02")

    # Three architecture columns
    layers = [
        ("Engine", "Bitboard 7×6\nRegras (3 especiais)\nKernels Numba", TEAL),
        ("Agentes", "MCTS + UCT\nMCTS-Solver\nID3 from scratch", DEEP),
        ("Interface", "CLI (HvH/HvC/CvC)\nGUI Pygame\nGUI Arena", NAVY),
    ]
    for i, (title, body, color) in enumerate(layers):
        x = 0.5 + i * 3.05
        add_box(s, x, 1.10, 2.85, 1.85, ICE)
        add_box(s, x, 1.10, 2.85, 0.10, color)
        add_text(s, x + 0.2, 1.30, 2.5, 0.4, title,
                 size=16, bold=True, color=color)
        add_text(s, x + 0.2, 1.75, 2.5, 1.1, body, size=11, color=DARK)

    # Contributions table
    add_text(s, 0.5, 3.20, 9.0, 0.32, "Distribuição de tarefas",
             size=13, bold=True, color=NAVY)
    contribs = [
        ("Duarte",  "Bitboard · MCTS-Solver · Kernels Numba · Pipeline ID3"),
        ("José",    "ID3 from scratch · Dataset generator · Análise leakage"),
        ("Tiago",   "GUI Pygame · CLI/torneio · Benchmarks · Tests"),
    ]
    for i, (name, role) in enumerate(contribs):
        y = 3.55 + i * 0.45
        add_box(s, 0.5, y, 1.5, 0.35, NAVY)
        add_text(s, 0.5, y, 1.5, 0.35, name,
                 size=11, bold=True, color=WHITE,
                 align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
        add_text(s, 2.1, y + 0.02, 7.4, 0.35, role,
                 size=11, color=DARK, anchor=MSO_ANCHOR.MIDDLE)

    slide_footer(s, 3)

    # ── Slide 4: MCTS — UCT and Variants ─────────────────────────────────────
    s = prs.slides.add_slide(blank)
    set_slide_bg(s, WHITE)
    slide_header(s, "MCTS — UCT e Variantes", chip="03")

    # Left — 4 phases
    add_text(s, 0.5, 1.05, 4.5, 0.4,
             "As 4 fases canónicas", size=15, bold=True, color=DEEP)
    phases = [
        ("1", "Selection",     "UCT escolhe o filho"),
        ("2", "Expansion",     "Adiciona novo nó"),
        ("3", "Simulation",    "Rollout aleatório"),
        ("4", "Backpropagation","Actualiza visitas/valor"),
    ]
    for i, (n, name, desc) in enumerate(phases):
        y = 1.50 + i * 0.50
        add_box(s, 0.5, y, 0.45, 0.45, TEAL)
        add_text(s, 0.5, y, 0.45, 0.45, n,
                 size=14, bold=True, color=WHITE,
                 align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
        add_text(s, 1.05, y + 0.05, 1.6, 0.35, name,
                 size=13, bold=True, color=NAVY)
        add_text(s, 2.65, y + 0.06, 2.5, 0.35, desc,
                 size=11, color=GRAY)

    # Right — UCT formula + variants
    add_box(s, 5.5, 1.05, 4.0, 3.95, ICE)
    add_box(s, 5.5, 1.05, 0.08, 3.95, DEEP)
    add_text(s, 5.7, 1.20, 3.7, 0.35,
             "Fórmula UCT", size=13, bold=True, color=NAVY)
    add_text(s, 5.7, 1.55, 3.7, 0.5,
             "score = Q + C · √(ln N / n)",
             size=16, color=DEEP, font="Cambria", italic=True)
    add_text(s, 5.7, 2.10, 3.7, 0.3,
             "C = √2 ≈ 1.414 (rewards normalizados em [0, 1])",
             size=10, color=GRAY, italic=True)
    add_text(s, 5.7, 2.55, 3.7, 0.35,
             "5 variantes implementadas",
             size=13, bold=True, color=NAVY)
    add_bullets(s, 5.7, 2.90, 3.7, 2.0, [
        "StandardUCT  ·  baseline",
        "ExperimentalUCT  ·  ordem de filhos",
        "TopK-UCT  ·  k filhos seleccionados (req §4.1)",
        "MCTS-Solver  ·  ver slide 5",
        "Numba JIT (4 variantes)  ·  ver slide 5",
    ], size=11, color=DARK)

    slide_footer(s, 4)

    # ── Slide 5: DIFFERENTIATORS — Solver / Reuse / Numba ────────────────────
    s = prs.slides.add_slide(blank)
    set_slide_bg(s, WHITE)
    slide_header(s, "Diferenciadores: Solver, Reuse, Numba", chip="04")

    # Three cards
    add_card(s, 0.5, 1.05, 2.95, 3.0,
             "MCTS-Solver",
             [
                 "Etiquetas WIN/LOSS/DRAW provadas matematicamente",
                 "Propagação AND/OR + distância minimax",
                 "Win-fast / Lose-slow",
             ], accent=DEEP)

    add_card(s, 3.55, 1.05, 2.95, 3.0,
             "Tree Reuse",
             [
                 "Reaproveita sub-árvore entre jogadas",
                 "Flat arrays + BFS compaction",
                 "Visitas inherited × ~0.35² ≈ 12%",
             ], accent=TEAL)

    add_card(s, 6.6, 1.05, 2.95, 3.0,
             "Numba JIT",
             [
                 "Kernels @njit em arrays planos",
                 "Warmup explícito + cache",
                 "Performance: ver coluna ↓",
             ], accent=CORAL)

    # Big throughput callout strip
    add_box(s, 0.5, 4.20, 9.05, 0.85, NAVY)
    add_text(s, 0.7, 4.25, 4.5, 0.4,
             "Throughput medido", size=11, bold=True, color=ICE)
    add_text(s, 0.7, 4.55, 4.5, 0.5,
             "FlatNumba: ~13 M iter/s   ·   StandardUCT: ~9 k iter/s",
             size=14, bold=True, color=WHITE)
    add_text(s, 5.5, 4.40, 4.0, 0.5,
             "≈ 1 400× speedup",
             size=22, bold=True, color=WHITE, align=PP_ALIGN.CENTER,
             anchor=MSO_ANCHOR.MIDDLE)

    slide_footer(s, 5)

    # ── Slide 6: ID3 — Iris + PopOut Dataset ─────────────────────────────────
    s = prs.slides.add_slide(blank)
    set_slide_bg(s, WHITE)
    slide_header(s, "ID3 — Iris e Dataset PopOut", chip="05")

    # Left: implementation
    add_text(s, 0.5, 1.05, 4.5, 0.4,
             "ID3 from scratch", size=15, bold=True, color=DEEP)
    add_bullets(s, 0.5, 1.50, 4.5, 1.6, [
        "Entropia de Shannon: −Σ pᵢ·log₂(pᵢ)",
        "Information Gain + recursive build",
        "Predict com fallback para majority label",
        "Sem scikit-learn no treino (§4.2)",
    ], size=13)

    # Iris results
    add_box(s, 0.5, 3.15, 4.5, 1.85, ICE)
    add_box(s, 0.5, 3.15, 0.08, 1.85, GREEN)
    add_text(s, 0.7, 3.25, 4.2, 0.32,
             "Iris (warm-up)", size=13, bold=True, color=NAVY)
    add_text(s, 0.7, 3.62, 4.2, 0.4,
             "95.07 % ± 5.63 %", size=22, bold=True, color=GREEN)
    add_text(s, 0.7, 4.10, 4.2, 0.3,
             "5×10-fold cross-validation (50 runs)",
             size=10, color=GRAY, italic=True)
    add_text(s, 0.7, 4.45, 4.2, 0.45,
             "Discretização quantile  ·  F1 ≥ 0.92 para 3 classes",
             size=10, color=DARK)

    # Right: PopOut dataset
    add_text(s, 5.3, 1.05, 4.3, 0.4,
             "Dataset PopOut", size=15, bold=True, color=DEEP)
    add_bullets(s, 5.3, 1.50, 4.3, 2.0, [
        "Oracle: ReuseFlatNumbaSolverMCTS @ 100k iter",
        "193 058 posições · 50 features · 14 classes",
        "Mirroring horizontal (simetria)",
        "3 tiers de blunder (5/25/50 %) + opening forçada 7×7",
        "Flag is_proven distingue labels provados",
    ], size=12, color=DARK)

    # bottom stat callout
    add_box(s, 5.3, 3.55, 4.3, 1.45, ICE)
    add_box(s, 5.3, 3.55, 0.08, 1.45, DEEP)
    add_text(s, 5.5, 3.65, 4.0, 0.32,
             "Geração paralela", size=12, bold=True, color=NAVY)
    add_text(s, 5.5, 4.00, 4.0, 0.4,
             "16 workers · ~5 min", size=18, bold=True, color=DEEP)
    add_text(s, 5.5, 4.45, 4.0, 0.4,
             "39 % das posições têm prova matemática",
             size=10, color=GRAY, italic=True)

    slide_footer(s, 6)

    # ── Slide 7: DATA LEAKAGE — Methodological rigour ────────────────────────
    s = prs.slides.add_slide(blank)
    set_slide_bg(s, WHITE)
    slide_header(s, "Rigor Metodológico: Detecção do Data Leakage", chip="06")

    add_text(s, 0.5, 1.00, 9.0, 0.4,
             "Auditoria do pipeline expôs duas camadas de leakage train/test",
             size=14, color=DARK, italic=True)

    # Comparison table
    headers = ["Split", "State overlap", "Test acc", "Diagnóstico"]
    rows = [
        ("BUGGY (original)",    "94.9 %", "0.876", "Bug do reset_index"),
        ("RANDOM (row-level)",  "80.4 %", "0.752", "Duplicação intrínseca"),
        ("GROUPED (state)",     "0.0 %",  "0.441", "Generalização real"),
    ]
    colors = [CORAL, GRAY, GREEN]

    table_y = 1.55
    col_w   = [2.3, 1.7, 1.5, 3.5]
    col_x   = [0.5, 2.8, 4.5, 6.0]

    # header row
    add_box(s, 0.5, table_y, 9.0, 0.38, NAVY)
    for i, h in enumerate(headers):
        add_text(s, col_x[i] + 0.05, table_y, col_w[i] - 0.1, 0.38, h,
                 size=12, bold=True, color=WHITE,
                 align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.MIDDLE)

    # data rows
    for r_i, (a, b, c, d) in enumerate(rows):
        y = table_y + 0.45 + r_i * 0.45
        fill = ICE if r_i % 2 == 0 else WHITE
        add_box(s, 0.5, y, 9.0, 0.42, fill)
        # row label with status color
        add_box(s, 0.5, y, 0.08, 0.42, colors[r_i])
        add_text(s, col_x[0] + 0.05, y, col_w[0] - 0.1, 0.42, a,
                 size=12, bold=True, color=DARK, anchor=MSO_ANCHOR.MIDDLE)
        add_text(s, col_x[1] + 0.05, y, col_w[1] - 0.1, 0.42, b,
                 size=12, color=DARK, anchor=MSO_ANCHOR.MIDDLE)
        add_text(s, col_x[2] + 0.05, y, col_w[2] - 0.1, 0.42, c,
                 size=13, bold=True, color=colors[r_i],
                 anchor=MSO_ANCHOR.MIDDLE)
        add_text(s, col_x[3] + 0.05, y, col_w[3] - 0.1, 0.42, d,
                 size=11, color=GRAY, italic=True, anchor=MSO_ANCHOR.MIDDLE)

    # Conclusion
    add_text(s, 0.5, 3.90, 9.0, 0.35,
             "Generalização verdadeira: 44 %  (vs 88 % memorização)",
             size=16, bold=True, color=GREEN, align=PP_ALIGN.CENTER)
    add_text(s, 0.5, 4.30, 9.0, 0.4,
             "Reportamos todos os três no notebook — honestidade > apresentação inflada.",
             size=11, color=GRAY, italic=True, align=PP_ALIGN.CENTER)
    add_text(s, 0.5, 4.65, 9.0, 0.4,
             "Baselines de referência: random 7 %  ·  always-drop_3 = 28 %  ·  ID3 GROUPED = 44 %",
             size=10, color=DARK, align=PP_ALIGN.CENTER)

    slide_footer(s, 7)

    # ── Slide 8: GLOBAL RESULTS ──────────────────────────────────────────────
    s = prs.slides.add_slide(blank)
    set_slide_bg(s, WHITE)
    slide_header(s, "Resultados", chip="07")

    # Three KPI columns
    kpis = [
        ("13 M",      "iter/s\n(FlatNumba)",          TEAL),
        ("44 %",      "generalização\nID3 GROUPED",   DEEP),
        ("52 %",      "win-rate vs\nSolver 100k",     CORAL),
    ]
    for i, (val, lab, col) in enumerate(kpis):
        x = 0.5 + i * 3.05
        add_box(s, x, 1.05, 2.85, 1.55, ICE)
        add_box(s, x, 1.05, 2.85, 0.08, col)
        add_text(s, x, 1.25, 2.85, 0.85, val,
                 size=42, bold=True, color=col,
                 align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
        add_text(s, x, 2.05, 2.85, 0.55, lab,
                 size=12, color=GRAY, align=PP_ALIGN.CENTER,
                 anchor=MSO_ANCHOR.MIDDLE)

    # Speed-quality figure
    fig_path = FIG_DIR / "speed_quality_tradeoff.png"
    if fig_path.exists():
        s.shapes.add_picture(str(fig_path), Inches(0.5), Inches(2.85),
                             width=Inches(5.5))
    else:
        add_box(s, 0.5, 2.85, 5.5, 2.20, ICE)
        add_text(s, 0.5, 3.7, 5.5, 0.4, "(speed_quality_tradeoff.png)",
                 size=10, color=GRAY, align=PP_ALIGN.CENTER)

    # Notes column
    add_text(s, 6.3, 2.90, 3.3, 0.4,
             "Pontos a destacar", size=13, bold=True, color=NAVY)
    add_bullets(s, 6.3, 3.30, 3.3, 1.8, [
        "ID3 híbrido paridade com Solver 100k",
        "FlatNumba domina em alto budget",
        "ID3 ideal para inference em ms",
    ], size=11)

    slide_footer(s, 8)

    # ── Slide 9: DIFFICULTIES & SOLUTIONS ────────────────────────────────────
    s = prs.slides.add_slide(blank)
    set_slide_bg(s, WHITE)
    slide_header(s, "Dificuldades & Soluções", chip="08")

    pairs = [
        ("Tree reuse com flat arrays",
         "BFS compaction re-indexa pai/filhos para slots 0..k antes do kernel.",
         TEAL),
        ("Data leakage não-óbvio",
         "Auditoria com 3 splits paralelos → adopção de split agrupado por estado.",
         CORAL),
        ("Classes desequilibradas (89 % drops)",
         "Oversampling controlado de proven (×4) — validado empiricamente.",
         DEEP),
    ]
    for i, (problem, solution, col) in enumerate(pairs):
        y = 1.15 + i * 1.30
        # problem chip
        add_box(s, 0.5, y, 0.25, 1.05, col)
        add_text(s, 0.9, y + 0.05, 4.0, 0.45, problem,
                 size=14, bold=True, color=NAVY)
        add_text(s, 0.9, y + 0.50, 8.4, 0.55, "Solução: " + solution,
                 size=12, color=DARK)

    slide_footer(s, 9)

    # ── Slide 10: CONCLUSION + FUTURE WORK ───────────────────────────────────
    s = prs.slides.add_slide(blank)
    set_slide_bg(s, NAVY)

    add_box(s, 0.0, 0.0, 0.25, 5.625, TEAL)

    add_text(s, 1.0, 0.50, 8.5, 0.5,
             "Conclusão", size=36, bold=True, color=WHITE)
    add_box(s, 1.0, 1.05, 1.5, 0.05, TEAL)

    # Three columns: requisitos / técnica / honesty
    blocks = [
        ("Requisitos do guião",
         ["3 cenários (HvH/HvC/CvC)",
          "MCTS com UCT",
          "Variantes além do standard",
          "ID3 from scratch (sem sklearn)",
          "Iris + dataset PopOut"]),
        ("Contribuições técnicas",
         ["MCTS-Solver com proof propagation",
          "Tree reuse com BFS compaction",
          "Kernels Numba: 13 M iter/s",
          "2 200 linhas de testes",
          "Pipeline reprodutível"]),
        ("Rigor & honestidade",
         ["3 splits comparados",
          "Híbrido declarado explicitamente",
          "Limitações documentadas",
          "Baselines reportadas",
          "Notebook auto-contido"]),
    ]
    for i, (title, items) in enumerate(blocks):
        x = 1.0 + i * 2.85
        add_text(s, x, 1.30, 2.7, 0.35, title,
                 size=13, bold=True, color=TEAL)
        for j, item in enumerate(items):
            y = 1.70 + j * 0.30
            add_text(s, x, y, 0.18, 0.28, "✓",
                     size=12, bold=True, color=TEAL)
            add_text(s, x + 0.22, y, 2.5, 0.28, item,
                     size=11, color=ICE)

    # Future work bar
    add_box(s, 1.0, 4.30, 8.5, 0.85, DEEP)
    add_text(s, 1.15, 4.40, 8.2, 0.3,
             "Trabalho futuro", size=11, bold=True, color=ICE)
    add_text(s, 1.15, 4.70, 8.2, 0.4,
             "Zobrist hashing + transposition tables   ·   AMAF/RAVE   ·   Opening books",
             size=13, color=WHITE)

    add_text(s, 0.5, 5.35, 9.0, 0.2,
             "github.com/Difl4/popout-ai", size=9, color=ICE,
             align=PP_ALIGN.RIGHT, italic=True)

    # ── Save ─────────────────────────────────────────────────────────────────
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    prs.save(OUT_PATH)
    print(f"Saved: {OUT_PATH}")
    print(f"  {len(prs.slides)} slides  ·  {OUT_PATH.stat().st_size//1024} KB")


if __name__ == "__main__":
    build()

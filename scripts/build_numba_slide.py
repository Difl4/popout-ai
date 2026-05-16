"""Build a single standalone slide about Numba + FlatNumba.

Output: docs/numba_slide.pptx — one slide (16:9), ready to insert into any deck.
Uses the same "Ocean Gradient" design system as the main presentation for
visual continuity.

Usage:
    python scripts/build_numba_slide.py
"""
from __future__ import annotations

import math
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Inches, Pt

# ─── Paths ────────────────────────────────────────────────────────────────────
ROOT     = Path(__file__).resolve().parents[1]
OUT_PATH = ROOT / "docs/numba_slide.pptx"

# ─── Palette (matches main presentation) ──────────────────────────────────────
NAVY     = RGBColor(0x21, 0x29, 0x5C)
DEEP     = RGBColor(0x06, 0x5A, 0x82)
TEAL     = RGBColor(0x1C, 0x72, 0x93)
TEAL_LT  = RGBColor(0x5A, 0xA4, 0xB8)
ICE      = RGBColor(0xE6, 0xF1, 0xF5)
ICE_DK   = RGBColor(0xC4, 0xDB, 0xE3)
WHITE    = RGBColor(0xFF, 0xFF, 0xFF)
DARK     = RGBColor(0x12, 0x1E, 0x3F)
SLATE    = RGBColor(0x33, 0x44, 0x55)
GRAY     = RGBColor(0x6B, 0x7A, 0x88)
AMBER    = RGBColor(0xF4, 0xA3, 0x41)

HEADER_FONT = "Cambria"
BODY_FONT   = "Calibri"
MONO_FONT   = "Consolas"


# ─── Primitives ──────────────────────────────────────────────────────────────
def set_bg(slide, color):
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = color


def add_text(slide, x, y, w, h, text, *, size=14, bold=False, italic=False,
             color=DARK, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP,
             font=BODY_FONT):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = Inches(0.02)
    tf.margin_top = tf.margin_bottom = Inches(0.01)
    tf.vertical_anchor = anchor
    p = tf.paragraphs[0]
    p.alignment = align
    r = p.add_run()
    r.text = text
    f = r.font
    f.name = font
    f.size = Pt(size)
    f.bold = bold
    f.italic = italic
    f.color.rgb = color


def add_bullets(slide, x, y, w, h, items, *, size=12, color=SLATE,
                bullet_color=TEAL, spacing=5):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = Inches(0.02)
    tf.margin_top = Inches(0.0)
    for i, item in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = PP_ALIGN.LEFT
        p.space_after = Pt(spacing)
        b = p.add_run()
        b.text = "•  "
        b.font.name = BODY_FONT
        b.font.size = Pt(size)
        b.font.bold = True
        b.font.color.rgb = bullet_color
        r = p.add_run()
        r.text = item
        r.font.name = BODY_FONT
        r.font.size = Pt(size)
        r.font.color.rgb = color


def add_rect(slide, x, y, w, h, *, fill=None, line=None,
             shape=MSO_SHAPE.RECTANGLE):
    shp = slide.shapes.add_shape(shape, Inches(x), Inches(y),
                                 Inches(w), Inches(h))
    if fill is None:
        shp.fill.background()
    else:
        shp.fill.solid()
        shp.fill.fore_color.rgb = fill
    if line is None:
        shp.line.fill.background()
    else:
        shp.line.color.rgb = line
    shp.shadow.inherit = False
    return shp


# ─── Build slide ─────────────────────────────────────────────────────────────
def build():
    prs = Presentation()
    prs.slide_width  = Inches(10)
    prs.slide_height = Inches(5.625)
    blank = prs.slide_layouts[6]

    s = prs.slides.add_slide(blank)
    set_bg(s, WHITE)

    # ── Header ───────────────────────────────────────────────────────────────
    add_rect(s, 0.5, 0.32, 0.65, 0.33, fill=TEAL)
    add_text(s, 0.5, 0.32, 0.65, 0.33, "03",
             size=12, bold=True, color=WHITE,
             align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    add_text(s, 1.30, 0.28, 8.2, 0.45,
             "Numba JIT + FlatNumba",
             size=22, bold=True, color=NAVY, font=HEADER_FONT)
    add_text(s, 1.30, 0.72, 8.2, 0.28,
             "Como conseguimos ~1400× mais iterações por segundo sem sair do Python",
             size=11, italic=True, color=GRAY)
    add_rect(s, 0.5, 1.05, 9.0, 0.015, fill=ICE_DK)

    # ── Two cards: Numba | FlatNumba ─────────────────────────────────────────
    card_y = 1.25
    card_h = 2.40

    # Left — Numba JIT
    add_rect(s, 0.5, card_y, 4.40, card_h, fill=ICE)
    add_rect(s, 0.5, card_y, 4.40, 0.10, fill=TEAL)
    add_text(s, 0.65, card_y + 0.18, 4.10, 0.40,
             "Numba JIT", size=16, bold=True, color=TEAL,
             font=HEADER_FONT)
    add_text(s, 0.65, card_y + 0.55, 4.10, 0.30,
             "Compilação @njit  ·  Python → LLVM",
             size=10, italic=True, color=GRAY)
    add_bullets(s, 0.65, card_y + 0.90, 4.10, 1.45, [
        "Decorador @njit(cache=True) nos hot loops "
        "(select · expand · simulate · backprop).",
        "Mantém Python como linguagem principal — "
        "sem cadeia de build C/Cython.",
        "warmup() explícito antes de benchmarks: "
        "tira o custo da 1ª compilação.",
    ], size=11)

    # Right — FlatNumba
    add_rect(s, 5.10, card_y, 4.40, card_h, fill=ICE)
    add_rect(s, 5.10, card_y, 4.40, 0.10, fill=AMBER)
    add_text(s, 5.25, card_y + 0.18, 4.10, 0.40,
             "FlatNumba", size=16, bold=True, color=DEEP,
             font=HEADER_FONT)
    add_text(s, 5.25, card_y + 0.55, 4.10, 0.30,
             "Árvore em arrays NumPy pré-alocados",
             size=10, italic=True, color=GRAY)
    add_bullets(s, 5.25, card_y + 0.90, 4.10, 1.45, [
        "Sem objectos Python no caminho crítico — "
        "nós = índices inteiros em arrays planos.",
        "Cache-friendly, vetorizável, sem overhead "
        "do garbage collector.",
        "Tree reuse via BFS compaction: sub-árvore "
        "reindexada para slots 0..k−1 entre turnos.",
    ], size=11, bullet_color=AMBER)

    # ── Performance bar chart (log-scale, units: iter/s) ─────────────────────
    bars = [
        ("StandardUCT",      9_000,      TEAL_LT),
        ("NumbaMCTS",        50_000,     TEAL),
        ("FlatNumbaMCTS",    179_000,    DEEP),
        ("ReuseFlat (peak)", 13_000_000, AMBER),
    ]
    chart_y0   = 3.85
    label_x    = 0.50
    label_w    = 1.65
    chart_x    = label_x + label_w + 0.10
    chart_w    = 3.50
    bar_h      = 0.21
    bar_gap    = 0.06
    bar_max    = max(b[1] for b in bars)

    for i, (name, val, col) in enumerate(bars):
        y = chart_y0 + i * (bar_h + bar_gap)
        w = max(0.10,
                math.log10(val + 1) / math.log10(bar_max + 1) * chart_w)
        add_text(s, label_x, y, label_w, bar_h, name,
                 size=10, bold=True, color=DARK,
                 align=PP_ALIGN.RIGHT, anchor=MSO_ANCHOR.MIDDLE)
        add_rect(s, chart_x, y, w, bar_h, fill=col)
        if val >= 1_000_000:
            v_str = f"{val // 1_000_000} M iter/s"
        elif val >= 1000:
            v_str = f"{val // 1000} k iter/s"
        else:
            v_str = f"{val} iter/s"
        add_text(s, chart_x + w + 0.05, y, 1.10, bar_h, v_str,
                 size=10, color=DARK, anchor=MSO_ANCHOR.MIDDLE)

    add_text(s, label_x, chart_y0 + 4 * (bar_h + bar_gap) + 0.05,
             chart_x + chart_w, 0.20,
             "log-scale  ·  unidades: iterações por segundo",
             size=8, color=GRAY, italic=True)

    # ── Right column: big speedup callout ────────────────────────────────────
    add_rect(s, 7.30, 3.85, 2.20, 1.20, fill=NAVY)
    add_text(s, 7.30, 3.90, 2.20, 0.30,
             "SPEEDUP", size=10, bold=True, color=AMBER,
             align=PP_ALIGN.CENTER)
    add_text(s, 7.30, 4.20, 2.20, 0.65,
             "≈ 1400×",
             size=36, bold=True, color=WHITE, font=HEADER_FONT,
             align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    add_text(s, 7.30, 4.82, 2.20, 0.22,
             "vs Python puro", size=9, italic=True, color=TEAL_LT,
             align=PP_ALIGN.CENTER)

    # ── Bottom tagline ───────────────────────────────────────────────────────
    add_text(s, 0.5, 5.20, 9.0, 0.30,
             "Python legível em cima  ·  código nativo cache-friendly em baixo.",
             size=10, italic=True, color=GRAY, align=PP_ALIGN.CENTER)

    # ── Save ─────────────────────────────────────────────────────────────────
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    prs.save(OUT_PATH)
    print(f"Saved: {OUT_PATH}")
    print(f"  1 slide · {OUT_PATH.stat().st_size // 1024} KB")


if __name__ == "__main__":
    build()

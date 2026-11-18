"""Add a legend entry for the dashed exact-place-field curves to Figure 3c,d.

The printed `figures/figure3.pdf` is kept as the base layer and is not redrawn; one dashed
handle and one label are drawn below the existing legend entries of panels c and d.
Needs PyMuPDF (`fitz`) and matplotlib (for the DejaVu Sans font file).
"""
from pathlib import Path

import fitz
import matplotlib

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "paper/iclr_paper/figures/figure3.pdf"
OUTPUT = ROOT / "paper/iclr_paper/figures/rebuttal/figure3_legend/figure3.pdf"
FONT = Path(matplotlib.get_data_path()) / "fonts/ttf/DejaVuSans.ttf"
LABEL, SIZE, GRAY = "Exact field", 7.0, (0.13, 0.13, 0.13)
ANCHORS = ("Cell 2", "Narrow cell 4")  # last existing legend entry of panels c and d


def last_entries(page: fitz.Page) -> list[tuple[float, float, float]]:
    """Return (x, baseline, line spacing) of the last legend entry in panels c and d."""
    spans = {s["text"].strip(): s["origin"] for b in page.get_text("dict")["blocks"]
             for l in b.get("lines", []) for s in l["spans"]}
    spacing = spans["Cell 2"][1] - spans["Cell 1"][1]
    return [(*spans[name], spacing) for name in ANCHORS]


def main() -> None:
    document = fitz.open(SOURCE)
    page = document[0]
    for x, baseline, spacing in last_entries(page):
        y = baseline + spacing
        page.draw_line((x - 2.8 * SIZE, y - 0.35 * SIZE), (x - 0.8 * SIZE, y - 0.35 * SIZE),
                       color=GRAY, width=1.0, dashes="[3.7 1.6] 0")
        page.insert_text((x, y), LABEL, fontsize=SIZE, fontname="dejavu", fontfile=str(FONT),
                         color=(0, 0, 0))
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    document.save(OUTPUT, garbage=3, deflate=True)


if __name__ == "__main__":
    main()

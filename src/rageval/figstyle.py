"""Shared figure palette and axis styling.

Palette and mark specs follow a validated data-viz method: categorical slots
blue #2a78d6 / orange #eb6834 (adjacent-pair CVD deltaE 24.7, all checks pass
on a light surface), diverging poles blue/red, text in ink tokens rather than
series colors, recessive grid, no top/right spines.

Every figure in ``results/figures/`` goes through :func:`apply_style` and
:func:`strip_spines` so the set reads as one system.
"""

from __future__ import annotations

BLUE = "#2a78d6"  # categorical slot 1: the un-reranked configuration
ORANGE = "#eb6834"  # categorical slot 2: the reranked configuration
RED = "#e34948"  # diverging negative pole: queries the reranker hurt
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#8a8983"
SURFACE = "#ffffff"


def apply_style() -> None:
    """Set the shared rcParams. Call once, after selecting the Agg backend."""
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "text.color": INK,
            "axes.labelcolor": INK2,
            "xtick.color": INK2,
            "ytick.color": INK2,
            "axes.edgecolor": MUTED,
            "font.size": 10,
        }
    )


def strip_spines(ax) -> None:
    """Drop the top and right spines and thin the remaining two."""
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_linewidth(0.8)

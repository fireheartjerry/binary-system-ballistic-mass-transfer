from __future__ import annotations

import matplotlib.pyplot as plt


def apply_plot_style() -> None:
    """Apply consistent, paper-ready Matplotlib styling."""
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        # Slightly larger base size to improve PDF readability in 10pt papers
        "font.size": 20,
        "axes.labelsize": 20,
        "axes.titlesize": 20,
        "xtick.labelsize": 15,
        "ytick.labelsize": 15,
        "legend.fontsize": 12,
        "legend.frameon": False,
        "axes.linewidth": 1.0,
        "lines.linewidth": 2.5,
        "savefig.bbox": "tight",
    })

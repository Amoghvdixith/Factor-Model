"""Matplotlib helpers: one visual system and a mandatory provenance stamp per figure.

Every saved figure carries a footer naming the data mode (SYNTHETIC / REAL), universe,
dates, frequency, model/estimation window, in-sample/OOS status and cost assumptions,
and the numbers behind it are saved as CSV next to it (the "table view").

Palette: a validated categorical order (blue, orange, aqua, yellow, magenta, green,
violet, red), used in fixed order; diverging blue<->red with a grey midpoint; recessive
grid and axes; thin 2px lines.  Three light slots are below 3:1 contrast, so charts
carry legends/direct labels and a CSV table.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from cycler import cycler  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
DIVERGING = LinearSegmentedColormap.from_list(
    "blue_grey_red", ["#184f95", "#6da7ec", "#f0efec", "#ec8a8a", "#a32d2d"])
SEQUENTIAL = LinearSegmentedColormap.from_list("blue_seq", ["#cde2fb", "#3987e5", "#0d366b"])


def setup() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlecolor": INK,
        "axes.titlesize": 12, "axes.titleweight": "bold", "axes.labelsize": 10,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelsize": 9, "ytick.labelsize": 9,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.prop_cycle": cycler(color=SERIES),
        "lines.linewidth": 2.0, "legend.frameon": False, "legend.fontsize": 9,
        "font.size": 10, "figure.dpi": 110, "date.converter": "concise",
    })


def stamp(fig: plt.Figure, meta: dict[str, Any]) -> None:
    """Add the provenance footer (data mode first, in capitals)."""
    order = ["data_mode", "universe", "dates", "frequency", "model", "sample", "costs"]
    parts = [f"{k.replace('_', ' ')}: {meta[k]}" for k in order if meta.get(k)]
    extra = [f"{k}: {v}" for k, v in meta.items() if k not in order and v]
    text = " | ".join(parts + extra)
    fig.text(0.01, 0.005, text, fontsize=7.5, color=INK2, ha="left", va="bottom", wrap=True)


def save(fig: plt.Figure, path: Path, meta: dict[str, Any], data: pd.DataFrame | None = None) -> Path:
    """Stamp, save PNG and (optionally) the underlying data as CSV with the same stem."""
    stamp(fig, meta)
    import warnings

    with warnings.catch_warnings():
        # colorbars spanning several axes are not tight_layout-compatible; layout still fine
        warnings.filterwarnings("ignore", message=".*tight_layout.*")
        fig.tight_layout(rect=(0, 0.05, 1, 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    if data is not None:
        csv = path.parent.parent / "tables" / (path.stem + ".csv")
        csv.parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(csv)
    return path


def zero_line(ax: plt.Axes, axis: str = "y") -> None:
    (ax.axhline if axis == "y" else ax.axvline)(0, color=AXIS, linewidth=1.0, zorder=1)


setup()

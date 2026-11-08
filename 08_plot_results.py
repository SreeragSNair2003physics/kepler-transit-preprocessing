"""Plot the completed Kepler masking comparison from mask_results.csv.

Run in the project folder:
    py 08_plot_results.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SOURCE = Path("mask_results.csv")
OUTPUT = Path("mask_comparison_summary.png")
COLORS = {"PC": "#2563eb", "AFP": "#ea580c", "NTP": "#9333ea"}


def main():
    table = pd.read_csv(SOURCE)
    if not (table["status"] == "ok").all():
        raise ValueError("Some rows failed; inspect the error column before plotting")

    masked = table["depth_with_mask"].to_numpy()
    unmasked = table["depth_without_mask"].to_numpy()
    difference_ppm = (masked - unmasked) * 1_000_000

    if not np.isfinite(masked).all() or not np.isfinite(unmasked).all():
        raise ValueError("A measured depth is missing or invalid")
    if (masked <= 0).any() or (unmasked <= 0).any():
        raise ValueError("This log-scale figure requires positive depths")

    figure, (comparison, changes) = plt.subplots(
        2, 1, figsize=(11, 9), gridspec_kw={"height_ratios": [1.1, 1]},
        constrained_layout=True,
    )

    # The upper panel compares the two depth estimates for each signal.
    for label, color in COLORS.items():
        rows = table[table["av_training_set"] == label]
        comparison.scatter(
            rows["depth_without_mask"], rows["depth_with_mask"],
            s=74, color=color, edgecolor="white", linewidth=0.8, label=label,
        )

    lower = min(masked.min(), unmasked.min()) / 1.6
    upper = max(masked.max(), unmasked.max()) * 1.6
    comparison.plot([lower, upper], [lower, upper],
                    linestyle="--", color="#64748b", linewidth=1,
                    label="No change")
    comparison.set(xscale="log", yscale="log", xlim=(lower, upper),
                   ylim=(lower, upper),
                   xlabel="Dip depth without mask (relative brightness)",
                   ylabel="Dip depth with mask (relative brightness)",
                   title="A. Measured depth for each star")
    comparison.grid(alpha=0.2, which="both")
    comparison.legend(ncol=4, fontsize=9, loc="upper left")

    # The lower panel shows the signed difference in parts per million.
    positions = np.arange(len(table))
    bar_colors = [COLORS[label] for label in table["av_training_set"]]
    changes.bar(positions, difference_ppm, color=bar_colors, width=0.78)
    changes.axhline(0, color="#334155", linewidth=1)
    changes.axvline(4.5, color="#cbd5e1", linewidth=1)
    changes.axvline(9.5, color="#cbd5e1", linewidth=1)
    changes.set(
        xticks=positions,
        xticklabels=[str(star_id) for star_id in table["kepid"]],
        ylabel="With mask − without mask (ppm)",
        xlabel="Kepler star ID (KIC)",
        title="B. Change in measured depth for each star",
    )
    changes.tick_params(axis="x", labelrotation=60, labelsize=8)
    changes.grid(axis="y", alpha=0.2)

    figure.suptitle("Transit masking and measured dip depth: 15 Kepler signals",
                    fontsize=15, fontweight="bold")
    figure.savefig(OUTPUT, dpi=200, bbox_inches="tight")
    plt.close(figure)

    print("Stars plotted:", len(table))
    print("Mask gave a larger depth:", int((difference_ppm > 0).sum()))
    print("Mask gave a smaller depth:", int((difference_ppm < 0).sum()))
    print("Median absolute change (ppm):", round(float(np.median(np.abs(difference_ppm))), 2))
    print("Saved figure:", OUTPUT)


if __name__ == "__main__":
    main()

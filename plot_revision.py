"""Export the measured accuracy/cost tradeoff; never choose a test operating point."""
import argparse
from pathlib import Path
import os


def plot(comparison, output):
    import pandas as pd
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(output.resolve() / ".mplcache"))
    os.environ.setdefault("XDG_CACHE_HOME", str(output.resolve() / ".cache"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    table = pd.read_csv(comparison)
    colors = {"v1": "#64748b", "v2": "#007f80"}
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), layout="constrained")
    for version in ("v1", "v2"):
        clean = table[(table.version == version) & (table.scenario == "clean") & table.method.isin(["cascade", "fusion"])]
        if len(clean) != 2:
            raise ValueError("Need clean cascade and fusion results for both versions")
        axes[0].plot(clean.mean_cpu_ms, clean.all_window_stress_recall * 100, color=colors[version], alpha=.65)
        for row in clean.itertuples():
            axes[0].scatter(row.mean_cpu_ms, row.all_window_stress_recall * 100, s=65,
                            marker="o" if row.method == "cascade" else "s", color=colors[version])
            axes[0].annotate(f"{version} {row.method}", (row.mean_cpu_ms, row.all_window_stress_recall*100),
                             xytext=(5, 7 if version == "v2" else -13), textcoords="offset points", fontsize=8)
    axes[0].set(xlabel="Mean host CPU time per scored window (ms)",
                ylabel="All-window stress recall (%)", title="Clean held-out accuracy and cost")
    clean_recalls = table[(table.scenario == "clean") & table.method.isin(["cascade", "fusion"])].all_window_stress_recall * 100
    axes[0].set_ylim(clean_recalls.min()-.6, clean_recalls.max()+.5)
    scenarios = ["clean", "ecg_flatline", "ecg_noise_severe", "ecg_amplitude_scale", "eda_dropout", "alternate_dropout"]
    positions = np.arange(len(scenarios))
    for version, offset in (("v1", -.18), ("v2", .18)):
        group = table[(table.version == version) & (table.method == "cascade")].set_index("scenario").loc[scenarios]
        bars = axes[1].bar(positions+offset, group.stress_f1, width=.34, color=colors[version], label=version)
        for bar, coverage in zip(bars, group.coverage):
            axes[1].annotate(f"{coverage:.0%}", (bar.get_x()+bar.get_width()/2, bar.get_height()),
                             xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8)
    axes[1].set_xticks(positions, ["Clean", "ECG flatline", "Severe ECG noise", "ECG amplitude ×3", "EDA missing", "Alternates missing"], rotation=25, ha="right")
    axes[1].set(ylabel="Stress F1 among accepted decisions", ylim=(0, 1.09),
                title="Cascade reliability; labels show accepted coverage")
    axes[1].legend(frameon=False)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
    fig.suptitle("Revised adaptive multimodal stress monitoring · 15-subject LOSO", fontsize=12)
    fig.savefig(output / "revision_tradeoff.png", dpi=200, bbox_inches="tight")
    fig.savefig(output / "revision_tradeoff.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plot(args.comparison, args.output)


if __name__ == "__main__":
    main()

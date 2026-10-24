"""Export waveform overlays for manual checking of the shared ECG peak detector."""
import argparse
from pathlib import Path
import numpy as np
from pulsepoint.data import RawSource
from pulsepoint.features import RawFeatures, ecg_peak_positions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--subject", default="S2")
    parser.add_argument("--start-seconds", type=float, default=0.)
    parser.add_argument("--window-seconds", type=float, default=60.)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Refusing to overwrite --output")
    row = {"source_file": f"{args.subject}/{args.subject}.pkl", "sampling_rate": 700,
           "start_sample": int(args.start_seconds*700), "end_sample": int((args.start_seconds+args.window_seconds)*700)}
    signals, rate = RawSource(args.raw_root).window(row)
    import os
    os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".cache/matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = [name for name in ("ecg", "eda", "resp") if name in signals]
    fig, axes = plt.subplots(len(names), 1, figsize=(14, 3*len(names)), squeeze=False)
    for name, ax in zip(names, axes[:, 0]):
        values = np.asarray(signals[name]).reshape(-1)
        seconds = np.arange(len(values))/rate + args.start_seconds
        ax.plot(seconds, values, lw=.6)
        if name == "ecg":
            peaks = ecg_peak_positions(values, rate)
            ax.scatter(seconds[peaks], values[peaks], s=10, color="red", label="Energy peak positions; inspect QRS alignment")
            ax.legend()
        ax.set(xlabel="Source seconds", ylabel=name.upper())
    fig.tight_layout()
    fig.savefig(args.output, dpi=150)
    plt.close(fig)
    provider = RawFeatures(signals, rate)
    print(provider.all_features())
    print(f"Saved {args.output}. This is a manual diagnostic, not a detector accuracy evaluation.")


if __name__ == "__main__":
    main()

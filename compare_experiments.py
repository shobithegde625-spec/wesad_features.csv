"""Compare complete versions without selecting a new test-set operating point."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from pulsepoint.data import write_json


def compare(previous, current, output, seed=42):
    previous, current, output = map(Path, (previous, current, output))
    summaries = [json.loads((p / "summary.json").read_text()) for p in (previous, current)]
    for key in ("protocol", "seed", "dataset_sha256", "replay_sha256", "subjects", "windows"):
        if summaries[0].get(key) != summaries[1].get(key):
            raise ValueError(f"Experiments differ in {key}; a direct before/after comparison would be misleading")
    splits = []
    for path in (previous, current):
        manifest = json.loads((path / "split_manifest.json").read_text())
        splits.append(sorted((tuple(sorted(f["outer_train_subjects"])), tuple(sorted(f["outer_test_subjects"]))) for f in manifest))
    if splits[0] != splits[1]:
        raise ValueError("Outer participant splits differ")
    # Check identities and targets, not merely aggregate window counts.
    identities = []
    for path in (previous, current):
        rows = pd.read_csv(path / "predictions.csv", usecols=["row_id", "scenario", "method", "label"])
        rows = rows[rows.method == "cascade"].drop(columns="method").sort_values(["scenario", "row_id"]).reset_index(drop=True)
        if rows.duplicated(["scenario", "row_id"]).any():
            raise ValueError("Duplicate scenario/window identities")
        identities.append(rows)
    if not identities[0].equals(identities[1]):
        raise ValueError("Experiments must score identical source intervals, scenarios and labels")
    comparisons = []
    for version, path in zip(("v1", "v2"), (previous, current)):
        table = pd.read_csv(path / "comparison.csv")
        table = table[table.method.isin(["cascade", "fusion", "ecg", "legacy_svm"])].copy()
        table.insert(0, "version", version)
        comparisons.append(table)
    combined = pd.concat(comparisons, ignore_index=True)
    per_subject = [pd.read_csv(p / "per_subject.csv") for p in (previous, current)]
    paired = per_subject[1].merge(per_subject[0], on=["method", "scenario", "subject"], suffixes=("_v2", "_v1"), validate="one_to_one")
    rng = np.random.default_rng(seed)
    differences = []
    for (method, scenario), group in paired.groupby(["method", "scenario"]):
        for metric in ("all_window_stress_recall", "coverage", "stress_f1", "selective_error", "mean_cpu_ms"):
            if f"{metric}_v2" not in group:
                continue
            delta = (group[f"{metric}_v2"] - group[f"{metric}_v1"]).dropna().to_numpy()
            if not len(delta):
                continue
            draws = rng.choice(delta, size=(2000, len(delta)), replace=True).mean(axis=1)
            differences.append(dict(method=method, scenario=scenario, metric=metric, subjects=len(delta),
                                    mean_difference=float(delta.mean()), lower_95=float(np.quantile(draws, .025)),
                                    upper_95=float(np.quantile(draws, .975))))
    output.mkdir(parents=True, exist_ok=False)
    combined.to_csv(output / "comparison.csv", index=False)
    write_json(output / "paired_revision_differences.json", differences)
    targets = {version: json.loads((path / "targets.json").read_text()) for version, path in zip(("v1", "v2"), (previous, current))}
    write_json(output / "targets.json", targets)
    lines = ["# Revised-model comparison", "", "Both versions use identical source windows, target labels, scenarios and outer participant splits. This is an exploratory revision on an already observed cohort, not independent confirmation.", "",
             "| Version | Method | Scenario | Coverage | All-window stress recall | Accepted stress F1 | Mean CPU ms |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    for row in combined.to_dict("records"):
        def fmt(key):
            value = row.get(key)
            return f"{value:.4f}" if value is not None and pd.notna(value) else "unavailable"
        lines.append(f"| {row['version']} | {row['method']} | {row['scenario']} | {fmt('coverage')} | {fmt('all_window_stress_recall')} | {fmt('stress_f1')} | {fmt('mean_cpu_ms')} |")
    lines += ["", "## Unchanged combined target", ""]
    for version in ("v1", "v2"):
        item = targets[version]["compute_and_recall"]
        lines.append(f"- {version}: {item['status']}; CPU fraction saved {item.get('cpu_fraction_saved')}; recall loss {item.get('stress_recall_loss')}.")
    lines += ["", "CPU comparisons are paired against each version's fusion expert on that run's host. Different runs can have different background loads; cross-version absolute timing is descriptive. The dedicated bundle benchmark measures a different fitted artifact and must not be combined with these recall estimates as a deployment guarantee.", "",
              "Participant-paired bootstrap intervals are in paired_revision_differences.json. Intervals quantify participant variability within this cohort and do not remove adaptive reuse of the dataset. Accepted-only F1 must be read alongside coverage and all-window recall. A failed quality check is an unknown measurement, never evidence of baseline.", "",
              "No battery, clinical or physical edge-device result is implied. The complete per-version reports retain event onset and delay limitations."]
    (output / "FINDINGS.md").write_text("\n".join(lines) + "\n")
    return combined


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("--current", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    compare(args.previous, args.current, args.output)
    print(f"Saved {args.output / 'FINDINGS.md'}")


if __name__ == "__main__":
    main()

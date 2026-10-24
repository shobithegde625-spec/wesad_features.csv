"""Standalone experiment reports and participant-level paired intervals."""
import os
from pathlib import Path
import numpy as np
import pandas as pd
from .metrics import risk_coverage, reliability


def paired_comparisons(table, seed=42):
    results = []
    rng = np.random.default_rng(seed)
    for scenario, group in table.groupby("scenario"):
        for reference in ("fusion", "legacy_forest", "confidence_only"):
            ref = group[group.method == reference].set_index("subject")
            if ref.empty:
                continue
            for method in group.method.unique():
                if method == reference:
                    continue
                current = group[group.method == method].set_index("subject")
                for metric in ("all_window_stress_recall", "coverage", "selective_error", "mean_total_ms"):
                    if metric not in current or metric not in ref:
                        continue
                    paired = current[[metric]].merge(ref[[metric]], left_index=True, right_index=True,
                                                     suffixes=("_current", "_reference")).dropna()
                    differences = (paired.iloc[:, 0] - paired.iloc[:, 1]).to_numpy()
                    if not len(differences):
                        continue
                    draws = rng.choice(differences, size=(1000, len(differences)), replace=True).mean(axis=1)
                    results.append({"method": method, "reference": reference, "scenario": scenario, "metric": metric,
                                    "mean_difference": float(differences.mean()), "lower_95": float(np.quantile(draws, .025)),
                                    "upper_95": float(np.quantile(draws, .975)), "subjects": len(differences)})
    return results


def plot_report(predictions, output):
    cache = Path(output).resolve() / ".cache"
    cache.mkdir(exist_ok=True)
    os.environ.setdefault("XDG_CACHE_HOME", str(cache))
    os.environ.setdefault("MPLCONFIGDIR", str(Path(output).resolve() / ".mplcache"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    clean = predictions[predictions.scenario == "clean"]
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    for index, (method, frame) in enumerate(clean.groupby("method")):
        color = plt.get_cmap("tab10")(index % 10)
        curve = risk_coverage(frame)
        if len(curve):
            axes[0].plot(curve.coverage, curve.risk, label=method, color=color)
        diagram = reliability(frame).dropna()
        if len(diagram):
            axes[1].plot(diagram.mean_probability, diagram.stress_frequency, marker="o", label=method, color=color)
        valid = frame[frame.label.isin([0, 1])]
        axes[2].scatter(valid.total_ms.mean(), valid.accepted.mean(), label=method, color=color)
    axes[0].set(xlabel="Coverage", ylabel="Selective error", title="Confidence ranking; probabilities available only")
    axes[1].plot([0, 1], [0, 1], "k--", alpha=.4)
    axes[1].set(xlabel="Mean predicted probability", ylabel="Observed stress frequency", title="Held-out reliability")
    axes[2].set(xlabel="Mean processing time (ms)", ylabel="Accepted coverage", title="Cost and coverage")
    axes[2].legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(Path(output) / "evaluation.png", dpi=160)
    plt.close(fig)


def tradeoff_diagnostics(predictions, output):
    """Matched accepted coverage and fixed-policy cost frontiers, not test tuning."""
    from .metrics import window_metrics
    matched, budgets = [], []
    for scenario, group in predictions.groupby("scenario"):
        groups = {name: rows[rows.label.isin([0, 1])].copy() for name, rows in group.groupby("method")}
        probabilistic = {name: rows for name, rows in groups.items()
                         if (rows.accepted & rows.stress_probability.notna()).any()}
        if probabilistic:
            common = min(float((rows.accepted & rows.stress_probability.notna()).mean()) for rows in probabilistic.values())
            for target in sorted(set([min(.8, common), min(.9, common), common])):
                if target <= 0:
                    continue
                for name, rows in probabilistic.items():
                    ranked = rows[rows.accepted & rows.stress_probability.notna()].copy()
                    ranked["confidence"] = np.maximum(ranked.stress_probability, 1-ranked.stress_probability)
                    ranked = ranked.sort_values(["confidence", "row_id"], ascending=[False, True], kind="stable")
                    keep = ranked.head(int(np.floor(target*len(rows)))).index
                    rows["accepted"] = rows.index.isin(keep)
                    rows.loc[~rows.accepted, "prediction"] = np.nan
                    matched.append({"method": name, "scenario": scenario, "target_coverage": target,
                                    "kind": "posthoc_accepted_confidence_ranking_not_an_operating_policy", **window_metrics(rows)})
        points = {name: window_metrics(rows) for name, rows in groups.items()}
        for budget in sorted(set(p["mean_total_ms"] for p in points.values() if "mean_total_ms" in p)):
            eligible = {name: p for name, p in points.items() if p.get("mean_total_ms", float("inf")) <= budget}
            for name, point in eligible.items():
                budgets.append({"scenario": scenario, "budget_ms": budget, "method": name,
                                "kind": "measured_fixed_policy_candidate_no_test_selection", **point})
    pd.DataFrame(matched).to_csv(Path(output) / "matched_coverage.csv", index=False)
    pd.DataFrame(budgets).to_csv(Path(output) / "budget_frontier.csv", index=False)


def write_report(report, comparison, output, pairs):
    lines = ["# Subject-held-out stress monitoring evaluation", "", f"Protocol: {report['protocol']}; subjects: {report['subjects']}; labeled windows: {report['windows']}.", "",
             "Outer-fold results evaluate the training procedure. The separately saved deployment bundle has its own provenance; these metrics are not a direct test of that fitted artifact.", "",
             "| Method | Scenario | Coverage | All-window stress recall | Accepted stress F1 | Mean processing ms |",
             "| --- | --- | --- | --- | --- | --- |"]
    for row in comparison.to_dict("records"):
        def fmt(key):
            value = row.get(key)
            return f"{value:.3f}" if value is not None and pd.notna(value) else "unavailable"
        lines.append(f"| {row['method']} | {row['scenario']} | {fmt('coverage')} | {fmt('all_window_stress_recall')} | {fmt('stress_f1')} | {fmt('mean_total_ms')} |")
    lines += ["", "Accuracy/F1 describe accepted decisions. All-window stress recall counts abstentions as undetected stress. Probability metrics disclose their own probability coverage.", "",
              "Participant bootstrap intervals, paired differences, split roles, reliability bins, routed reliability and confidence-ranked risk curves are saved beside this report.", "",
              "Threshold sweeps are exploratory tradeoff curves, not independently retuned test operating points. Predeclared targets remain engineering targets; results do not establish clinical validity.", ""]
    lines.extend(f"- {item}" for item in report["limitations"])
    lines += ["", "Multimodal corruption and physical-device energy targets remain unassessed unless the corresponding waveform and hardware experiments have been run.", ""]
    Path(output, "REPORT.md").write_text("\n".join(lines))

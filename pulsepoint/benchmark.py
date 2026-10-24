"""Single-thread end-to-end measurements; no acquisition or energy claims."""
from pathlib import Path
from time import perf_counter
import os
import psutil
import numpy as np
from threadpoolctl import threadpool_limits, threadpool_info
from .data import load_dataset, RawSource, environment, write_json
from .replay import Runtime, chronological, source_interval
from .training import load_bundle
from .corruption import corrupt


def benchmark(bundle_path, data_path, output, raw_root=None, limit=100, repeats=3, scenarios=("clean",)):
    if min(limit, repeats) < 1:
        raise ValueError("Limit and repeats must be positive")
    if scenarios != ("clean",) and not raw_root:
        raise ValueError("Corruption benchmark requires raw waveform data")
    started = perf_counter()
    bundle = load_bundle(bundle_path)
    loading_ms = (perf_counter()-started)*1000
    frame = chronological(load_dataset(data_path, min_subjects=1))
    # Spread across participants and conditions rather than time one repeated row.
    positions = np.unique(np.linspace(0, len(frame)-1, min(limit, len(frame)), dtype=int))
    rows = frame.iloc[positions].to_dict("records")
    source = RawSource(raw_root) if raw_root else None
    modes = ["cascade", "ecg"] + (["fusion"] if "fusion" in bundle["experts"] else [])
    samples = []
    process = psutil.Process(os.getpid())
    memory_before = process.memory_info().rss
    with threadpool_limits(limits=1):
        threads = threadpool_info()
        runtimes = {(scenario, repeat, mode): Runtime(bundle, mode)
                    for scenario in scenarios for repeat in range(repeats) for mode in modes}
        # Warm up each path independently; reset before timed measurements.
        warm = rows[0]
        signals, rate = source.window(warm) if source else (None, 700)
        for (scenario, repeat, mode), runtime in runtimes.items():
            warm_signals = corrupt(signals, scenario, warm["row_id"]) if source else None
            runtime.predict(signals=warm_signals, rate=rate, features=None if source else warm, interval=source_interval(warm))
            runtime.reset()
        # Read each compressed participant once. Independent trackers preserve the
        # chronological sequence for every scenario/repeat despite interleaving.
        for index, row in enumerate(rows):
            original, rate = source.window(row) if source else (None, 700)
            for scenario_index, scenario in enumerate(scenarios):
                signals = corrupt(original, scenario, row["row_id"]) if source else None
                for repeat in range(repeats):
                    offset = (index+repeat+scenario_index) % len(modes)
                    order = modes[offset:] + modes[:offset]
                    for mode in order:
                        result = runtimes[(scenario, repeat, mode)].predict(signals=signals, rate=rate, features=None if source else row, interval=source_interval(row))
                        samples.append({"mode": mode, "scenario": scenario, "repeat": repeat, "row_id": row["row_id"], **result})
    import pandas as pd
    table = pd.DataFrame(samples)
    summary = []
    for (mode, scenario), group in table.groupby(["mode", "scenario"]):
        item = {"mode": mode, "scenario": scenario, "count": len(group), "coverage": float(group.accepted.mean()),
                "escalation_fraction": float(group.escalated.mean())}
        for field in ("quality_ms", "feature_ms", "inference_ms", "total_ms", "cpu_ms"):
            item[field] = {"mean": float(group[field].mean()), **{f"p{p}": float(np.percentile(group[field], p)) for p in (50, 95, 99)}}
        summary.append(item)
    report = {"artifact_id": bundle["artifact_id"], "loading_ms": loading_ms,
              "bundle_bytes": Path(bundle_path).stat().st_size, "rss_before_measurement_bytes": memory_before,
              "rss_after_measurement_bytes": process.memory_info().rss, "environment": environment(), "threadpools": threads,
              "input_kind": "waveform" if source else "stored_features", "results": summary,
              "excludes": ["file IO", "window acquisition duration", "sensor/radio energy"],
              "note": "Process RSS includes Python and all loaded experts; this is host performance, not a battery-life estimate."}
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    table.to_csv(output / "timings.csv", index=False)
    write_json(output / "benchmark.json", report)
    return report

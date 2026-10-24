"""Assess predeclared engineering targets without presenting missing evidence as success."""


def assess_targets(report):
    results = {(r["method"], r["scenario"]): r for r in report["results"]}
    cascade = results.get(("cascade", "clean"))
    fusion = results.get(("fusion", "clean"))
    single = results.get(("cascade_single_window", "clean"))
    output = {}
    if cascade:
        pooled = cascade["pooled"]
        output["clean_coverage"] = {"status": "met" if pooled["coverage"] >= .8 and (pooled["stress_coverage"] or 0) >= .8 else "not_met",
                                    "coverage": pooled["coverage"], "stress_coverage": pooled["stress_coverage"]}
    if cascade and fusion and report["waveform_processing_measured"]:
        current, reference = cascade["pooled"], fusion["pooled"]
        saving = 1-current["mean_cpu_ms"]/reference["mean_cpu_ms"] if reference["mean_cpu_ms"] else None
        recall_loss = reference["all_window_stress_recall"] - current["all_window_stress_recall"]
        output["compute_and_recall"] = {"status": "met_on_this_host" if saving is not None and saving >= .25 and recall_loss <= .03 else "not_met_on_this_host",
                                         "cpu_fraction_saved": saving, "stress_recall_loss": recall_loss}
    else:
        output["compute_and_recall"] = {"status": "pending_multimodal_waveform_experiment"}
    if cascade and single and cascade["events"]["status"] == "laboratory_replay":
        current, reference = cascade["events"], single["events"]
        count = reference["false_event_starts"]
        reduction = 1-current["false_event_starts"]/count if count else None
        output["false_events"] = {"status": ("met" if reduction is not None and reduction >= .2 else "not_met") if count else "zero_reference_events_percentage_undefined",
                                   "relative_reduction": reduction,
                                   "absolute_reduction": count-current["false_event_starts"],
                                   "cascade_event_recall": current["event_recall"], "single_window_event_recall": reference["event_recall"],
                                   "cascade_median_delay_seconds": current["median_onset_delay_seconds"],
                                   "single_window_median_delay_seconds": reference["median_onset_delay_seconds"]}
    else:
        output["false_events"] = {"status": "pending_chronological_waveform_evaluation"}
    output["target_device_latency"] = {"status": "pending_physical_target_device_benchmark"}
    output["energy"] = {"status": "not_measured_no_battery_life_claim"}
    output["interpretation"] = "Point targets are engineering checks, not statistical superiority or clinical guarantees. Inspect participant intervals and matched-coverage failure results."
    return output

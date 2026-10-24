# What the improved model supports

Suggested project claim:

> In a fifteen-fold participant-held-out WESAD evaluation, a quality-aware adaptive multimodal cascade reduced mean processing CPU time by 15.1% relative to always-on fusion while keeping all-window stress recall within 0.66 percentage points. Independent alternate-sensor inference retained useful classification under ECG failure, and total sensor loss produced explicit insufficient-evidence decisions.

This wording describes the measured tradeoff. The original combined target of at least 25% lower CPU cost with at most three recall points lost remains unmet: recall retention now meets its point target, but CPU saving falls short. The experiment does not establish battery savings, physical edge-device performance or clinical validity.

## Measured clean-window result

Fifteen outer leave-one-subject-out folds score the same 434 fully labeled baseline/stress replay windows. Fit, calibration and policy participants are disjoint within each fold. Labels do not enter runtime routing.

| Measure | Original cascade v1 | Revised cascade v2 | Revised always-on fusion |
| --- | ---: | ---: | ---: |
| Accepted coverage | 91.5% | 100.0% | 100.0% |
| All-window stress recall | 84.2% | 88.2% | 88.8% |
| Stress F1 among accepted windows | 0.831 | 0.810 | 0.846 |
| Mean processing CPU, ms/window | 12.26* | 14.24 | 16.77 |

*See the machine-readable v1 report for exact timing. Absolute times across separate runs are descriptive because background load can vary. The 15.1% saving compares v2 cascade with its own fusion reference in the same run. All-window recall counts abstentions as missed stress. V2's higher recall and coverage come with lower pooled precision/F1 than v1; it is not an improvement on every classification metric. V2 clean accuracy is 85.5%, compared with 88.7% for fusion.

## Sensor-failure evidence and remaining weaknesses

| Scenario | V1 cascade coverage / stress recall / accepted F1 | V2 cascade coverage / stress recall / accepted F1 |
| --- | --- | --- |
| ECG flatline or missing burst | 91.9% / 67.8% / 0.752 | 100% / 77.6% / 0.787 |
| ECG clipping | 96.1% / 61.2% / 0.446 | 100% / 70.4% / 0.639 |
| Moderate ECG noise | 94.5% / 91.4% / 0.610 | 100% / 94.7% / 0.696 |
| Severe ECG noise | 100% / 93.4% / 0.510 | 100% / 77.6% / 0.787 |
| ECG amplitude multiplied by three | 99.5% / 91.4% / 0.542 | 100% / 53.9% / 0.471 |
| EDA, respiration, or both alternates missing | 85.9% / 78.3% / 0.826 | 100% / 73.0% / 0.779 |
| Total sensor dropout | 0% / 0% / undefined | 0% / 0% / undefined |

The noise gate replaces severely noisy ECG with alternate evidence, substantially improving precision/F1 while reducing stress recall in that scenario. A high recall under noise can coexist with many false stress predictions. Amplitude scaling is a major regression; its confident ECG errors can bypass fusion even though fusion itself obtains 93.4% stress recall on scaled signals. Missing alternate channels retain full operational coverage but lose recall compared with v1. These cases limit the breadth of the robustness claim and remain priorities for a separately evaluated revision.

## Why the implementation is worthwhile

The contribution combines independent signal-quality checks, calibrated experts, separate escalation and acceptance rules, availability-aware fallback, explicit rejection and participant-held-out failure evaluation. The new selection rule minimizes escalation subject to fusion-relative recall and operational-error constraints on development participants. It no longer treats the original cascade's own errors as its performance reference. Model selection now considers grouped SVM hyperparameters and a compact forest, giving equal weight to fitting participants.

Adaptive sensing has prior art. This project contributes the specific implementation, reproducible comparison and measured tradeoffs; it should not claim to invent adaptive sensing.

## Scope and uncertainty

The paired participant-macro stress-recall change versus v1 is +3.33 points, with a bootstrap 95% interval from -9.33 to +20.02 points. V2 versus fusion is -0.79 points, interval -8.67 to +5.09. These broad intervals do not establish statistical superiority or formal noninferiority. The 0.66-point pooled recall difference is a descriptive engineering result. WESAD has already informed this revision, so the revised run is exploratory; independent confirmation requires a new cohort.

The event controller remains experimental: only 2 of 15 eligible stress episodes receive a new event start. The lower false-start count cannot be presented as reliable onset detection. Classifications use completed 60-second windows; millisecond processing times exclude acquisition duration and temporal confirmation.

The separately fitted dashboard artifact is `artifacts/multimodal-real-v2/model.joblib`, ID `dae3a7c563c9`, 76,313 bytes. It uses logistic ECG, a compact 64-tree alternate expert and a 100-tree fusion expert. The dashboard selects the newest complete artifact automatically.

Its separate waveform benchmark uses 100 representative replay windows, three repeats, four scenarios and independently timed cascade/ECG/fusion paths. On clean inputs from all conditions, the fitted cascade averages **19.83 ms CPU versus 19.65 ms for fusion**, with 83% escalation and p95 total processing 27.99 ms. Thus this particular artifact does **not** demonstrate CPU saving on that workload. Restricting the existing timing rows to binary-eligible intervals also does not establish saving (29 unique windows, 87 repeat samples per mode; 20.97 versus 20.07 ms CPU). The held-out procedure's 15.1% saving must not be attached to this model's artifact ID or presented as a deployment guarantee. The final model's policy participants and calibrator differ from those in each outer fold; the small cohort makes the resulting tradeoff unstable. Measured RSS includes the offline participant cache and Python allocator, rather than a pure streaming device footprint.

## Evidence for the report or presentation

- [Revised full evaluation](reports/multimodal-real-v2/REPORT.md)
- [Verified before/after tables](reports/multimodal-revision-comparison/FINDINGS.md)
- [Tradeoff figure, PNG](reports/multimodal-revision-comparison/revision_tradeoff.png) and [PDF](reports/multimodal-revision-comparison/revision_tradeoff.pdf)
- [Participant-paired revision intervals](reports/multimodal-revision-comparison/paired_revision_differences.json)
- [Unchanged target assessment](reports/multimodal-real-v2/targets.json)
- [Fitted-artifact host benchmark](reports/multimodal-real-benchmark-v2/benchmark.json)
- [Method fixed before the revised results](MODEL_IMPROVEMENT.md)

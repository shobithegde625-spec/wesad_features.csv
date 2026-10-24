# Implementation and real-data validation status

## Revised model v2

The requested model improvement is implemented and the full fifteen-fold, eleven-scenario waveform evaluation has completed. Read [RESEARCH_CLAIM.md](RESEARCH_CLAIM.md) for the supported wording, paired uncertainty, regressions and presentation figures. The original plan's thresholds remain unchanged.

- Separate ECG early-exit thresholds from final expert acceptance, with fusion-relative development constraints and availability-aware ECG fallback.
- Expand participant-weighted grouped selection to tuned SVMs and a compact forest; add a scale-invariant high-frequency noise quality gate for new bundles. Older bundles keep their previous behavior.
- Clean recall improves from 84.2% to 88.2%, with coverage increasing from 91.5% to 100%. Revised fusion reaches 88.8% recall. The revised cascade saves 15.1% of mean processing CPU versus that fusion reference: recall retention meets the three-point criterion, while the combined 25% saving target remains unmet.
- ECG-flatline recall improves from 67.8% to 77.6%; severe-noise accepted F1 improves from 0.510 to 0.787. Pooled clean F1 declines from 0.831 to 0.810, amplitude-scaling performance regresses substantially, and alternate-dropout recall declines. These limitations are preserved in the comparison.
- Final artifact: `artifacts/multimodal-real-v2/model.joblib`, ID `dae3a7c563c9`, 76,313 bytes; logistic ECG, compact forest alternate and forest fusion.
- Revised outputs: `reports/multimodal-real-v2/`, `reports/multimodal-revision-comparison/` and `reports/multimodal-real-benchmark-v2/`. The dashboard selects the newest complete artifact automatically.
- The separate final-artifact waveform benchmark completed: 100 representative source windows, three repeats and four scenarios. Clean cascade mean CPU is 19.83 ms versus fusion's 19.65 ms; cascade p95 total is 27.99 ms. This fitted model does not demonstrate CPU saving on that workload, so the held-out procedure's 15.1% saving is not a deployment guarantee.
- **41 tests passed** after the implementation changes; dependency consistency and whitespace checks passed. Revised cohort results remain exploratory because WESAD has already informed development.

## Original implementation and v1 evidence

The planned software changes are implemented. The official WESAD archive has been downloaded and checksum-verified; real multimodal training, the full corruption experiment, a dedicated host benchmark and all tests have completed. Physical target-device and energy claims remain untested.

| Plan step | Completed evidence | Remaining research/hardware work |
| --- | --- | --- |
| 1. Baseline/provenance | Shared factories/schemas, import-safe CLIs, declared dependencies/lock file, unique artifacts; all historical scores reproduced exactly | None for supplied CSV reproduction |
| 2. Subject separation | Fifteen nested LOSO folds, disjoint fit/calibration/policy roles, saved splits/models/predictions, participant bootstrap and paired intervals | Independent cohorts for broader generalization |
| 3. Multimodal data | Verified official archive; direct ZIP adapter; 445 pure-condition windows and 1,442 chronological replay windows; source intervals and identities retained | Additional real-world recordings |
| 4. Features/quality | Shared lazy ECG/RR, EDA and respiration features; movement context; quality checks; known-rate synthetic tests and a qualitative S9 waveform diagnostic | Broader detector validation and annotated signal-quality data |
| 5. Calibrated experts | Real ECG, alternate and fusion experts with grouped selection and participant-disjoint sigmoid calibration | Stronger generalization and calibration under noise |
| 6. Routing | Quality/confidence gates, escalation, alternate routing, explicit abstention and development-only corruption tuning; real ECG-flatline fallback demonstrated | Better rejection of severe noisy ECG |
| 7. Events | Source-time confirmation/recovery/cooldown and boundary resets; held-out false-start, new-onset and active-event exposure analyses | Reliable event-onset detection across unsupported states |
| 8. Ablations/failures | All 15 folds across eleven scenarios; original-feature baselines, ECG, static fusion, confidence-only, no-abstention and single-window variants | New cohorts and unseen real-world failure modes |
| 9. Edge measurements | Dedicated waveform benchmark: 100 source windows, three repeats, three scenarios; stage/total CPU and latency, model size and process RSS | Physical edge device and power instrumentation |
| 10. Dashboard | Shared runtime; automatic real-model/replay/ZIP defaults; participant/session playback, rejected decisions, quality and exported history | Live wearable adapter |
| 11. Targets/report | Automated target assessment, paired uncertainty and explicit negative findings | Combined compute/recall target remains unmet |

Validation completed:

- **34 tests passed**, including archive ingestion, downloader resume/checksum behavior, the complete synthetic waveform pipeline, repeat-isolated benchmark state, and every dashboard page against the real multimodal artifact.
- Dependency consistency (`pip check`) and `git diff --check` passed.
- The original five-fold SVM, 200-tree forest and logistic regression metrics match the saved comparison exactly.
- All 445 original windows, participant order, labels and seven ECG amplitude statistics were reproduced from the official raw archive (numerical tolerance 1e-10).
- The actual full replay contains 434 pure binary-scored windows; 1,008 mixed/unsupported windows are processed chronologically but remain unscored for binary classification.
- Original feature CSV, saved model, historical comparison and original figures are unchanged.

Final local outputs:

- `data/downloads/WESAD.zip`: 2,249,444,501-byte official archive, SHA-256 `5e15d2606adf16d819ba535c1786d92e94bae5ff5392f7ae01fb63f9938fd71c`.
- `data/wesad-documentation/acquisition.json`: verified host checksum, source, recording identities, documentation and research-use terms.
- `data/wesad-real-v1/`: real classification/replay tables and provenance.
- `artifacts/multimodal-real-v1/model.joblib`: final real-data prototype, artifact ID `180f6c507c3a`; ECG logistic regression plus alternate/fusion Random Forest experts.
- `reports/multimodal-real-v1/REPORT.md` and `FINDINGS.md`: complete held-out results, target assessment and interpretation.
- `reports/multimodal-real-benchmark-v1/benchmark.json`: artifact-specific waveform timings on this host.
- `reports/raw-data-validation.json` and `reports/multimodal-diagnostics/S9-baseline-waveforms.png`: acquisition/feature verification and a qualitative waveform check.

On clean held-out replay windows, the quality-aware cascade has 91.5% accepted coverage, 84.2% all-window stress recall and accepted stress F1 0.831. Always-on fusion achieves 88.8% recall and F1 0.846 at full coverage. The cascade saves about 24.7% of mean processing CPU time in this evaluation but loses 4.6 recall points, missing the combined 25%/3-point target. Under ECG flatline it retains 91.9% coverage through the alternate expert; severe ECG noise remains poorly rejected.

Temporal confirmation reduces new baseline event starts, but new stress-onset detection remains weak. An event already active before stress is not counted as a new onset. FINDINGS.md separately reports active-event exposure so low start counts cannot be mistaken for low false-alert duration. Onset delay is referenced to the first fully labeled stress window, not the exact protocol transition. These results do not establish clinical validity or reliable field alerting.

The separate fitted-artifact benchmark records a clean cascade mean CPU time of approximately 11.1 ms and p95 total processing of 20.4 ms. It samples all conditions and a different fitted artifact than the outer folds; do not combine that timing with held-out recall as a deployment guarantee. Acquisition time, IO and sensor/radio energy are excluded. Process RSS includes the offline participant cache and allocator retention.

Launch the dashboard:

```bash
.venv/bin/python -m streamlit run app.py
```
